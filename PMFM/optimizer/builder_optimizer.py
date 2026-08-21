import torch.optim as optim


def build_optimizer(model, args):
    if args["name"] == 'Adam':
        return optim.Adam(model, lr=float(args['lr']), weight_decay=float(args['weight_decay']))
    elif args["name"] == 'SGD':
        return optim.SGD(model, lr=float(args['lr']), momentum=float(args['momentum']), weight_decay=float(args['weight_decay']))
    elif args["name"] == 'AdamW':
        return optim.AdamW(model, lr=float(args['lr']), weight_decay=float(args['weight_decay']))
    elif args["name"] == 'RMSprop':
        return optim.RMSprop(model, lr=float(args['lr']), weight_decay=float(args['weight_decay']), momentum=float(args['momentum']))
