# 学习率调整器

import torch.optim.lr_scheduler as lr_scheduler


def build_lr_scheduler(optimizer, args):
    if args['name'] == 'StepLR':
        return lr_scheduler.StepLR(optimizer=optimizer, step_size=int(args['step_size']), gamma=float(args['gamma']))
    elif args['name'] == 'ExponentialLR':
        return lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=float(args['gamma']))
    elif args['name'] == 'CosineAnnealingLR':
        return lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=int(args['T_max']), eta_min=float(args['eta_min']))
    elif args['name'] == 'ReduceLROnPlateau':
        return lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode=args['mode'], patience=int(args['patience']), threshold=float(args['threshold']))
    elif args['name'] == 'CosineAnnealingWarmRestarts':
        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=int(args['T_0']), T_mult=int(args['T_mult']))
    else:
        pass


