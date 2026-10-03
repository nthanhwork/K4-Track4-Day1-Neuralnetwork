"""One training pipeline for all experiments; eval is never used for selection."""
import math
import os
import random
import resource
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params, activation_stats
from optimizer import build_optimizer, build_scheduler, clip_gradients

DEFAULT_CFG = dict(exp_id='base-s1', group='baseline', description='Baseline M-base',
                   loss='ce', optimizer='sgd_momentum', lr=0.05, weight_decay=0.0,
                   momentum=0.9, batch=512, epochs=20, hidden=(256, 128), dropout=0.0,
                   init='he', clip_norm=None, precision='fp32', seed=1,
                   train_probe_size=50000, scheduler=None)

def _progress(message):
    print(message, flush=True)
    if os.environ.get('LAB_PROGRESS_LOG'):
        with open(os.environ['LAB_PROGRESS_LOG'], 'a') as f:
            f.write(message + '\n')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def macro_f1_from_confusion(cm):
    tp = np.diag(cm).astype(float)
    denominator = cm.sum(0) + cm.sum(1)
    return float(np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0).mean())

def compute_loss(logits, y, loss_name):
    if loss_name == 'ce':
        return F.cross_entropy(logits, y)
    if loss_name == 'mse':
        # Mean over all B*7 elements: RAW logits versus one-hot; no factor 1/2.
        return F.mse_loss(logits, F.one_hot(y, 7).to(logits.dtype))
    raise ValueError(f'Unknown loss: {loss_name}')

@torch.no_grad()
def predict(model, X, batch_size=8192):
    model.eval()
    return torch.cat([model(X[i:i + batch_size]).argmax(1) for i in range(0, len(X), batch_size)])

@torch.no_grad()
def evaluate(model, X, y, loss_name='ce', batch_size=8192):
    model.eval()
    cm = torch.zeros(49, dtype=torch.int64, device=X.device)
    loss_sum = torch.zeros((), device=X.device)
    for xb, yb in iterate_batches(X, y, batch_size, shuffle=False):
        logits = model(xb)
        loss_sum += compute_loss(logits, yb, loss_name) * len(yb)
        cm += torch.bincount(yb * 7 + logits.argmax(1), minlength=49)
    cm = cm.reshape(7, 7).cpu().numpy()
    return dict(loss=float(loss_sum / len(y)), acc=float(np.trace(cm) / cm.sum()),
                macro_f1=macro_f1_from_confusion(cm))

def run_experiment(cfg, data):
    cfg = {**DEFAULT_CFG, **cfg}
    device = data['X_tr'].device
    if cfg['precision'] not in ('fp32', 'fp16', 'bf16'):
        raise ValueError('Unknown precision')
    if cfg['precision'] == 'fp16' and device.type != 'cuda':
        raise ValueError('FP16 experiment requires CUDA; use CPU BF16 or FP32')
    set_seed(cfg['seed'])
    model = MLP(cfg['hidden'], cfg['dropout'], cfg['init']).to(device)
    assert count_params(model) == EXPECTED_PARAMS[tuple(cfg['hidden'])]
    optimizer = build_optimizer(cfg['optimizer'], model.parameters(), cfg['lr'], cfg['weight_decay'], cfg['momentum'])
    steps = math.ceil(len(data['X_tr']) / cfg['batch'])
    scheduler = build_scheduler(optimizer, cfg['scheduler'], steps * cfg['epochs'])
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['precision'] == 'fp16')
    dtype = torch.float16 if cfg['precision'] == 'fp16' else torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(cfg['seed'])
    # Fixed representative train probe, independent of the model seed.
    probe_gen = torch.Generator(device=device).manual_seed(42)
    ix = torch.randperm(len(data['X_tr']), generator=probe_gen, device=device)[:cfg['train_probe_size']]
    Xp, yp = data['X_tr'][ix], data['y_tr'][ix]
    step0 = evaluate(model, data['X_val'], data['y_val'], cfg['loss'])['loss']
    stds = activation_stats(model, data['X_val'][:512])
    history = {k: [] for k in ('epoch', 'train_loss', 'val_loss', 'val_acc', 'val_macro_f1',
                              'grad_norm', 'grad_norm_max', 'clip_fraction', 'epoch_time_s', 'steps', 'scaler_skipped_steps')}
    step_norms = []
    best_loss, best_epoch, best_state = math.inf, 0, None
    diverged = False
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    _progress(f"{cfg['exp_id']}: lr={cfg['lr']} precision={cfg['precision']} step0={step0:.6f}")
    for epoch in range(1, cfg['epochs'] + 1):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        model.train()
        norms, recorded_norms, skipped_steps = [], [], 0
        for xb, yb in iterate_batches(data['X_tr'], data['y_tr'], cfg['batch'], generator):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=dtype, enabled=cfg['precision'] != 'fp32'):
                loss = compute_loss(model(xb), yb, cfg['loss'])
            if not torch.isfinite(loss):
                diverged = True
                break
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # Before measuring/clipping even if clip_norm is None.
            gn = clip_gradients(model.parameters(), cfg['clip_norm'])
            if not math.isfinite(gn):
                if scaler.is_enabled():
                    # Overflow is handled by GradScaler: skip update and lower scale.
                    # None records an unrepresentable norm; never invent a finite value.
                    scaler.step(optimizer)
                    scaler.update()
                    recorded_norms.append(None)
                    skipped_steps += 1
                    continue
                diverged = True
                break
            norms.append(gn)
            recorded_norms.append(gn)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
        step_norms.extend(recorded_norms)
        if diverged:
            _progress('Diverged: nonfinite loss/gradient; stopped.')
            break
        tr = evaluate(model, Xp, yp, cfg['loss'])
        val = evaluate(model, data['X_val'], data['y_val'], cfg['loss'])
        if not all(math.isfinite(v) for v in (*tr.values(), *val.values())):
            diverged = True
            break
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        values = dict(epoch=epoch, train_loss=tr['loss'], val_loss=val['loss'], val_acc=val['acc'],
                      val_macro_f1=val['macro_f1'], grad_norm=float(np.mean(norms)),
                      grad_norm_max=max(norms), steps=len(recorded_norms), scaler_skipped_steps=skipped_steps, epoch_time_s=elapsed,
                      clip_fraction=float(np.mean(np.array(norms) > cfg['clip_norm'])) if cfg['clip_norm'] else 0.0)
        for k, v in values.items():
            history[k].append(v)
        if val['loss'] < best_loss:
            best_loss, best_epoch = val['loss'], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        _progress(f"  epoch {epoch:02d}/{cfg['epochs']}: train={tr['loss']:.4f} val={val['loss']:.4f} "
              f"acc={val['acc']:.4f} F1={val['macro_f1']:.4f} grad={np.mean(norms):.3f} scaler_skips={skipped_steps} {elapsed:.1f}s")
    b = best_epoch - 1
    summary = dict(step0_loss=step0, best_val_loss=best_loss if best_epoch else None,
                   best_epoch=best_epoch, final_train_loss=history['train_loss'][-1] if best_epoch else None,
                   final_val_loss=history['val_loss'][-1] if best_epoch else None,
                   val_acc=history['val_acc'][b] if best_epoch else None,
                   val_macro_f1=history['val_macro_f1'][b] if best_epoch else None,
                   time_per_epoch_s=float(np.mean(history['epoch_time_s'])) if best_epoch else None,
                   peak_mem_MB=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda'
                               else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                   memory_kind='CUDA allocated peak' if device.type == 'cuda' else 'process RSS high-water mark',
                   diverged=diverged, completed_epochs=len(history['epoch']),
                   activation_stds=stds, device=str(device),
                   grad_norm_p75=float(np.percentile([v for v in step_norms if v is not None], 75)) if any(v is not None for v in step_norms) else None,
                   scaler_skipped_steps=sum(history['scaler_skipped_steps']))
    return dict(cfg=cfg, history=history, summary=summary, grad_norm_steps=step_norms, best_state=best_state)

def write_predictions(row_id, preds, path):
    import pandas as pd
    row_id, preds = np.asarray(row_id), np.asarray(preds)
    if row_id.ndim != 1 or preds.shape != row_id.shape:
        raise ValueError('Prediction lengths/shapes differ')
    if len(np.unique(row_id)) != len(row_id) or not np.isin(preds, range(7)).all():
        raise ValueError('Duplicate IDs or invalid classes')
    if not np.array_equal(row_id, row_id.astype(np.int64)) or not np.array_equal(preds, preds.astype(np.int64)):
        raise ValueError('IDs and predictions must be integers')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(row_id=row_id.astype(np.int64), pred=preds.astype(np.int64))).to_csv(path, index=False)

def final_eval(cfg, result, data, pred_path):
    if result['best_state'] is None:
        raise ValueError('No finite validation checkpoint')
    model = MLP(cfg['hidden'], cfg['dropout'], cfg['init']).to(data['X_eval'].device)
    model.load_state_dict(result['best_state'])
    write_predictions(data['eval_row_id'], predict(model, data['X_eval']).cpu().numpy(), pred_path)
