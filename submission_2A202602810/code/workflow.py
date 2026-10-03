"""Notebook helpers: save every run, compare against measured seed variation."""
from pathlib import Path
import numpy as np
import pandas as pd
from train import run_experiment
from results_table import save_result
from plots import plot_run

def persist(result, out_dir):
    save_result(result, Path(out_dir) / 'results')
    plot_run(result, Path(out_dir) / 'figures' / (result['cfg']['exp_id'] + '.png'))
    return result

def run_store(cfg, data, out_dir):
    return persist(run_experiment(cfg, data), out_dir)

def overview(results):
    return pd.DataFrame([dict(exp_id=r['cfg']['exp_id'], **{k:r['summary'][k] for k in
                       ('step0_loss','best_epoch','val_acc','val_macro_f1','time_per_epoch_s','diverged')}) for r in results])

def contrast(result, baseline, noise):
    s, b = result['summary'], baseline['summary']
    if s['val_macro_f1'] is None:
        return 'Run diverged before a finite validation epoch; no metric comparison.'
    delta = s['val_macro_f1'] - b['val_macro_f1']
    exceeds = abs(delta) > noise
    return (f"{result['cfg']['exp_id']}: val F1={s['val_macro_f1']:.6f}; delta against base-s1={delta:+.6f}; "
            f"2σ={noise:.6f}; exceeds noise={exceeds}. Best epoch by minimum validation loss={s['best_epoch']}. "
            'Single-seed technique comparisons remain provisional.')
