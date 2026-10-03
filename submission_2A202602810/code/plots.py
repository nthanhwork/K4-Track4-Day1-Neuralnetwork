"""Per-run curves and controlled comparisons, saved as PNG."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_run(result, path):
    cfg, h = result['cfg'], result['history']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for key in ('train_loss', 'val_loss'):
        axes[0].plot(h['epoch'], h[key], label=key)
    for key in ('val_acc', 'val_macro_f1'):
        axes[1].plot(h['epoch'], h[key], label=key)
    for key in ('grad_norm', 'grad_norm_max'):
        axes[2].plot(h['epoch'], h[key], label=key)
    if cfg['clip_norm']:
        axes[2].axhline(cfg['clip_norm'], color='red', linestyle=':', label='clip threshold')
    for ax, ylabel in zip(axes, ('Loss (eval mode)', 'Validation metric', 'Global L2 norm before clipping')):
        ax.set(xlabel='Epoch', ylabel=ylabel)
        ax.axvline(result['summary']['best_epoch'], linestyle='--', color='gray', label='best val loss')
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"{cfg['exp_id']} | {cfg['optimizer']} lr={cfg['lr']} batch={cfg['batch']} "
                 f"{cfg['loss']} dropout={cfg['dropout']} init={cfg['init']} {cfg['precision']}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_compare(results, metric, path, title=''):
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in results:
        ax.plot(r['history']['epoch'], r['history'][metric], label=r['cfg']['exp_id'])
    ax.set(xlabel='Epoch', ylabel=metric, title=title)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
