"""Generate the report and workbook strictly from measured experiment records."""
import json
import platform
from pathlib import Path
import numpy as np
import torch
from results_table import to_row, write_xlsx

def table(headers, rows):
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |']
    for row in rows:
        lines.append('| ' + ' | '.join(str(x).replace('|','/') for x in row) + ' |')
    return '\n'.join(lines)

def f(x):
    return 'N/A' if x is None else f'{x:.6f}'

def make_deliverables(results, base, final, scores, health, data, out_dir, repo_root, total_time):
    out_dir, repo_root = Path(out_dir), Path(repo_root)
    template_path = repo_root/'templates/experiment_table_template.xlsx'
    if not template_path.exists():
        template_path = Path(__file__).with_name('experiment_table_template.xlsx')
    lookup = {r['cfg']['exp_id']:r for r in results}
    seeds = [r for r in results if r['cfg']['group']=='baseline']
    f1s = np.array([r['summary']['val_macro_f1'] for r in seeds])
    accs = np.array([r['summary']['val_acc'] for r in seeds])
    mean, std = f1s.mean(), f1s.std(ddof=1)
    noise = 2*std
    b, fs = base['summary'], final['summary']
    bid, fid = base['cfg']['exp_id'], final['cfg']['exp_id']
    official = scores[fid]
    baseline_eval = scores[bid]
    annotations = [[bid,'parameters',health['parameters']], [bid,'He health step0',health['step0_loss']],
                   [bid,'uniform step0 check',health['uniform_step0_loss']],
                   [bid,'overfit20 CE',health['tiny_metrics']['loss']],
                   [bid,'overfit20 accuracy',health['tiny_metrics']['acc']],
                   [bid,'majority validation accuracy',data['majority_val_acc']],
                   [bid,'train samples',len(data['X_tr'])], [bid,'validation samples',len(data['X_val'])],
                   [fid,'eval samples',official['n_eval']], [fid,'total wall time seconds',total_time],
                   [bid,'Python',platform.python_version()], [bid,'PyTorch',torch.__version__],
                   [bid,'device',torch.cuda.get_device_name(0) if data['X_tr'].is_cuda else 'CPU'],
                   [bid,'split seed',data['split_seed']]]
    for name, value in health['gradient_norms'].items():
        annotations.append([bid,'health gradient '+name,value])
    for r in results:
        rid, h, s = r['cfg']['exp_id'], r['history'], r['summary']
        annotations += [[rid,'activation stds after Linear',json.dumps(s['activation_stds'])],
                        [rid,'mean fraction clipped',float(np.mean(h['clip_fraction'])) if h['epoch'] else None],
                        [rid,'max preclip gradient',max(h['grad_norm_max'],default=None)],
                        [rid,'gradient norm p75',s['grad_norm_p75']],
                        [rid,'first epoch train loss',h['train_loss'][0] if h['epoch'] else None],
                        [rid,'first epoch validation loss',h['val_loss'][0] if h['epoch'] else None],
                        [rid,'total attempted update steps',sum(h['steps'])],
                        [rid,'GradScaler skipped overflow steps',s.get('scaler_skipped_steps',0)]]
    annotations += [[fid,'final eval per class: class','support','precision','recall','F1']]
    for c in official['per_class']:
        annotations.append([fid,'class '+str(c['cls']),c['support'],c['precision'],c['recall'],c['f1']])
    annotations += [[fid,'confusion matrix rows=true, columns=pred',0,1,2,3,4,5,6]]
    for i, row in enumerate(official['confusion_matrix']):
        annotations.append([fid,'true class '+str(i),*row])
    rows=[]
    for r in results:
        delta = r['summary']['val_macro_f1']-b['val_macro_f1'] if r['summary']['val_macro_f1'] is not None else None
        notes = f"delta vs base-s1={f(delta)}; 2sigma={noise:.6f}; selected final={r is final}"
        rows.append(to_row(r,scores.get(r['cfg']['exp_id']),notes))
    write_xlsx(rows,template_path,out_dir/'experiments.xlsx',annotations)
    def run_table(group, include_base=True):
        chosen=([base] if include_base else [])+[r for r in results if r is not base and r['cfg']['group']==group]
        return table(['exp_id','lr','val accuracy','val macro-F1','best epoch'],
                     [[r['cfg']['exp_id'],r['cfg']['lr'],f(r['summary']['val_acc']),f(r['summary']['val_macro_f1']),
                       r['summary']['best_epoch']] for r in chosen])
    def comparison(r):
        delta=r['summary']['val_macro_f1']-b['val_macro_f1']
        direction='cao hơn' if delta>0 else 'thấp hơn' if delta<0 else 'bằng'
        conclusion='vượt' if abs(delta)>noise else 'chưa vượt'
        return (f"`{r['cfg']['exp_id']}` có val macro-F1 {f(r['summary']['val_macro_f1'])}, {direction} "
                f"`{bid}` {abs(delta):.6f}; chênh lệch {conclusion} ngưỡng 2σ={noise:.6f}. "
                'Đây là so sánh một seed của kỹ thuật với baseline; chưa chứng minh tính ổn định qua nhiều seed.')
    dropout=lookup['drop-01']
    gap_base=b['final_val_loss']-b['final_train_loss']
    ds=dropout['summary']; gap_drop=ds['final_val_loss']-ds['final_train_loss']
    mse=lookup['loss-mse']
    high_delta=lookup['highlr-clip']['summary']['val_macro_f1']-lookup['highlr-no-clip']['summary']['val_macro_f1']
    clipping=[lookup[i] for i in ('clip-normal','highlr-no-clip','highlr-clip')]
    amp=[r for r in results if r['cfg']['group']=='amp']
    inits=[base]+[r for r in results if r['cfg']['group']=='init']
    optimizers=[r for r in results if r['cfg']['optimizer']=='adam']
    adam=max(optimizers,key=lambda r:r['summary']['val_macro_f1'])
    sgds=[r for r in results if r['cfg']['group']=='optimizer' and r['cfg']['optimizer']=='sgd_momentum']+[base]
    sgd=max(sgds,key=lambda r:r['summary']['val_macro_f1'])
    winner=adam if adam['summary']['val_macro_f1']>sgd['summary']['val_macro_f1'] else sgd
    worst=min(official['per_class'],key=lambda c:c['f1'])
    cm=np.array(official['confusion_matrix'])
    err=cm[worst['cls']].copy();err[worst['cls']]=0
    confused=int(err.argmax())
    train_counts=torch.bincount(data['y_tr'],minlength=7).cpu().tolist()
    environment=torch.cuda.get_device_name(0) if data['X_tr'].is_cuda else 'CPU (4 threads)'
    from evidence import compact_report
    measured_environment = dict(conda='base', python=platform.python_version(),
                                torch=torch.__version__, device=environment,total_time_s=total_time)
    report = compact_report(results, health, measured_environment,
                            dict(final_exp_id=fid), official, baseline_eval)
    for r in amp:
        annotations.append([r['cfg']['exp_id'],'FP32/AMP epoch time ratio',
                            b['time_per_epoch_s']/r['summary']['time_per_epoch_s']])
    zero_probe = out_dir/'zero_init_gradient_check.json'
    if zero_probe.exists():
        for name,value in json.loads(zero_probe.read_text())['initial_gradient_norms'].items():
            annotations.append(['init-zeros','initial gradient '+name,value])
    # Additional evidence used in report stays traceable inside the original four-sheet workbook.
    for c,count in enumerate(train_counts):
        annotations.append([fid,'train class '+str(c)+' support',count])
    write_xlsx(rows,template_path,out_dir/'experiments.xlsx',annotations)
    (out_dir/'REPORT.md').write_text(report)
    (out_dir/'environment.json').write_text(json.dumps(dict(conda='base',python=platform.python_version(),
                 torch=torch.__version__,device=environment,total_time_s=total_time),indent=2))
