"""Fixed split and training-only standardization; eval loads only after selection."""
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10

def _read(path):
    with np.load(path) as a:
        X, y = a['X'], a['y']
        assert X.shape == (len(y), 54) and X.dtype == np.float32 and y.dtype == np.int64
        assert np.isfinite(X).all() and np.isin(y, np.arange(7)).all()
        return X, y, a['row_id']

def load_split(processed_dir='data/processed'):
    X, y, _ = _read(Path(processed_dir) / 'train.npz')
    Xe, ye, ids = _read(Path(processed_dir) / 'eval.npz')
    return X, y, Xe, ye, ids

def make_val_split(X, y, val_fraction=0.2, seed=42):
    a, b, c, d = train_test_split(X, y, test_size=val_fraction, random_state=seed, stratify=y)
    return a, c, b, d

def fit_standardizer(X_tr):
    numeric = X_tr[:, :N_NUMERIC].astype(np.float64)
    mean, std = numeric.mean(0), numeric.std(0)
    return mean, np.where(std == 0, 1, std)

def apply_standardizer(X, mean, std):
    out = X.copy()
    out[:, :N_NUMERIC] = (out[:, :N_NUMERIC] - mean) / std
    return out

def prepare_data(device, val_fraction=0.2, seed=42, processed_dir='data/processed', include_eval=False):
    # Default path does not even open eval.npz.
    X, y, _ = _read(Path(processed_dir) / 'train.npz')
    Xt, yt, Xv, yv = make_val_split(X, y, val_fraction, seed)
    mean, std = fit_standardizer(Xt)
    data = dict(mean=mean, std=std, split_seed=seed, val_fraction=val_fraction,
                processed_dir=str(Path(processed_dir).resolve()))
    for name, features, labels in [('tr', Xt, yt), ('val', Xv, yv)]:
        data['X_' + name] = torch.as_tensor(apply_standardizer(features, mean, std), device=device)
        data['y_' + name] = torch.as_tensor(labels, device=device)
    data['majority_val_acc'] = float((yv == np.bincount(yt).argmax()).mean())
    print('Train:', Xt.shape, 'Validation:', Xv.shape, 'Majority val accuracy:', data['majority_val_acc'])
    if include_eval:
        attach_eval(data)
    return data

def attach_eval(data):
    """Call only after selecting the model; use the already fitted standardizer."""
    X, y, ids = _read(Path(data['processed_dir']) / 'eval.npz')
    device = data['X_tr'].device
    data['X_eval'] = torch.as_tensor(apply_standardizer(X, data['mean'], data['std']), device=device)
    data['y_eval'] = torch.as_tensor(y, device=device)
    data['eval_row_id'] = ids

def iterate_batches(X, y, batch_size, generator=None, shuffle=True):
    if batch_size < 1 or len(X) != len(y):
        raise ValueError('Invalid batch size or lengths')
    if shuffle:
        indices = torch.randperm(len(X), generator=generator, device=X.device)
        for start in range(0, len(X), batch_size):
            ix = indices[start:start + batch_size]
            yield X[ix], y[ix]
    else:
        for start in range(0, len(X), batch_size):
            yield X[start:start + batch_size], y[start:start + batch_size]
