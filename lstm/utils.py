import torch
from adabound import AdaBound

def getOptimizer(args, model_params):
    args.optim = args.optim.lower()
    if args.optim == 'sgdm':
        return torch.optim.SGD(model_params, lr=args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        return torch.optim.Adam(model_params, lr=args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, eps=args.eps)
    elif args.optim == 'adadelta':
        return torch.optim.Adadelta(model_params,lr=args.lr,weight_decay=args.weight_decay)
    elif args.optim == 'amsgrad':
        return torch.optim.Adam(model_params, lr=args.lr, betas=(args.beta1, args.beta2),
                                weight_decay=args.weight_decay, eps=args.eps, amsgrad=True)
    elif args.optim == 'adamw':
        return torch.optim.AdamW(model_params, lr=args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, eps=args.eps)
    elif args.optim == 'adabelief':
        return torch.optim.AdaBelief(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, eps=args.eps)
    elif args.optim == 'adabound':
        return AdaBound(model_params, args.lr, betas=(args.beta1, args.beta2),
                        final_lr=args.final_lr, gamma=args.gamma,
                        weight_decay=args.weight_decay)
    elif args.optim == 'lookahead':
        #sgd = torch.optim.SGD(model_params, lr=args.lr, momentum=args.momentum,
        #                 weight_decay=args.weight_decay)
        adam = torch.optim.Adam(model_params, lr=args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, eps=args.eps)
        return torch.optim.Lookahead(adam,la_steps=args.hist_length,la_alpha=args.alpha,pullback_momentum=args.pullback_momentum)
    elif args.optim == 'adasam':
        #sgd = torch.optim.SGD(model_params,lr=args.lr,weight_decay=args.weight_decay)
        adam = torch.optim.Adam(model_params, lr=args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, eps=args.eps)
        return torch.optim.AdaSAM(adam,period=args.period,hist_length=args.hist_length,
                               damp=args.damp,beta=args.beta,alpha=args.alpha,loss_r=1e-3)
    elif args.optim == 'padasam':
        #sgd = torch.optim.SGD(model_params,lr=args.lr,weight_decay=args.weight_decay)
        adam = torch.optim.Adam(model_params, lr=args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, eps=args.eps)
        return torch.optim.pAdaSAM(adam,period=args.period,hist_length=args.hist_length,
                               damp=args.damp,beta=args.beta,alpha=args.alpha,precision=0)
    elif args.optim == 'oaar':
        sgd = torch.optim.SGD(model_params, lr=args.lr, weight_decay=args.weight_decay)
        return torch.optim.oAAR(sgd, period=args.period, hist_length=args.hist_length,
                               damp=args.damp, beta=args.beta, alpha=args.alpha)
    elif args.optim == 'aenr':
        sgd = torch.optim.SGD(model_params, lr=args.lr, weight_decay=args.weight_decay)
        return torch.optim.AENR(sgd, period=args.period, hist_length=args.hist_length,
                               damp=args.damp, beta=args.beta, alpha=args.alpha)
    elif args.optim == 'cr':
        cr = torch.optim.CR(model_params,lr=args.lr,weight_decay=args.weight_decay,period=args.period,
                                damp=args.damp,mix=args.beta,mu=args.alpha,loss_r=0.01)
        return cr
    else:
        print('Optimizer not found')

def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target
