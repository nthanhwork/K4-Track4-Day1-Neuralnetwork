"""Optimizer construction and the global gradient norm before clipping."""
import math
import torch

OPTIMIZERS = ('sgd', 'sgd_momentum', 'adam', 'adamw')

def build_optimizer(name, params, lr, weight_decay=0.0, momentum=0.9, betas=(0.9, 0.999), eps=1e-8):
    if name not in OPTIMIZERS or lr is None or lr <= 0:
        raise ValueError('Invalid optimizer or learning rate')
    kwargs = dict(lr=lr, weight_decay=weight_decay)
    if name in ('sgd', 'sgd_momentum'):
        return torch.optim.SGD(params, momentum=momentum if name == 'sgd_momentum' else 0, **kwargs)
    cls = torch.optim.Adam if name == 'adam' else torch.optim.AdamW
    return cls(params, betas=betas, eps=eps, **kwargs)

def build_scheduler(optimizer, name, total_steps, **kwargs):
    if name is None:
        return None
    if name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, **kwargs)
    raise ValueError(f'Unknown scheduler: {name}')

def clip_gradients(params, max_norm):
    if max_norm is not None and max_norm <= 0:
        raise ValueError('clip_norm must be positive')
    return float(torch.nn.utils.clip_grad_norm_(params, math.inf if max_norm is None else max_norm))
