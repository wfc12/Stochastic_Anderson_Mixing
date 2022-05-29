import torch
from functools import reduce
from .optimizer import Optimizer
from torch import pinverse as pinv




class AdaSAMVR(Optimizer):

    def __init__(self, optimizer=None,beta=1.0,
                 hist_length=10,period=1,damp=0.01,gamma=0,precision=0):
        if optimizer is None:
            raise ValueError("optimizer cannot be None")        
        if beta < 0.0:
            raise ValueError("Invalid Anderson beta parameter: {}".format(beta))
        if hist_length < 0:
            raise ValueError("Invalid history size: {}".format(hist_length))
        if period <= 0:
            raise ValueError("Invalid period size: {}".format(period))

        self.optimizer = optimizer
        self.beta = beta
        self.damp = damp
        self.hist_length = hist_length
        self.period = period
        self.gamma = gamma

        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state
        self.defaults = self.optimizer.defaults

        if len(self.param_groups) != 1:
            raise ValueError("AdaSAMVR doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None

        N = self._numel()
        device = self._params[0].device        
        self.dtype = dtype = torch.float64 if precision==1 else self._params[0].dtype
        state = self.state
        state.setdefault('step', 0)
        state.setdefault('Xk', torch.zeros((N, hist_length), device=device, dtype=dtype))
        state.setdefault('Rk', torch.zeros((N, hist_length), device=device, dtype=dtype))
        state.setdefault('x_prev', torch.zeros(N, dtype=dtype, device=device))
        state.setdefault('res_prev', torch.zeros(N, dtype=dtype, device=device))
        state.setdefault('d_xk_avg',torch.zeros(N, dtype=dtype, device=device))
        state.setdefault('d_res_avg',torch.zeros(N, dtype=dtype, device=device))

    def __setstate__(self, state):
        super(AdaSAMVR, self).__setstate__(state)

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _gather_flat_data(self):
        views = []
        for p in self._params:
            views.append(p.data.view(-1))
        return torch.cat(views,0)

    def _store_data(self,other):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.copy_(other[offset:offset + numel].view_as(p))
            offset += numel
        assert offset == self._numel()

    def _store_grad(self,other):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.grad.copy_(other[offset:offset + numel].view_as(p))
            offset += numel
        assert offset == self._numel()

    
    def setfullgrad(self,length):
        self.fullgrad = self._gather_flat_grad().div_(length)

    def settmpx(self):
        self.xk = self._gather_flat_data()

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        if closure is None:
            raise ValueError("closure cannot be None!")
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        beta = self.beta
        damp = self.damp
        hist_length = self.hist_length
        period = self.period
        optimizer=self.optimizer
        N = self._numel()               
        gamma = self.gamma
        state = self.state

        loss = closure()
        

        xk = self._gather_flat_data().to(self.dtype)
        flat_grad = self._gather_flat_grad()
        self._store_data(self.xk)
        closure()
        fullgrad = self.fullgrad
        tmp_grad = self._gather_flat_grad()
        flat_grad = flat_grad - tmp_grad + fullgrad
        self._store_data(xk)
        self._store_grad(flat_grad)

        weight_decay = group['weight_decay']
        res = flat_grad.add(alpha=weight_decay,other=xk).neg().to(self.dtype)

        Xk, Rk = state['Xk'], state['Rk']
        res_prev, x_prev = state['res_prev'], state['x_prev']

        cnt = state['step']
        if cnt > 0:
            d_xk_avg, d_res_avg = state['d_xk_avg'], state['d_res_avg']
            d_xk_avg.mul_(gamma).add_(xk - x_prev, alpha=1 - gamma)
            d_res_avg.mul_(gamma).add_(res - res_prev, alpha=1 - gamma)
            k = (cnt - 1) % hist_length
            Xk[:, k] = d_xk_avg
            Rk[:, k] = d_res_avg
            delta_x = Xk[:, k]

        state['x_prev'].copy_(xk)
        state['res_prev'].copy_(res)

        eps = 1e-8
        alpha = 1
        optimizer.step(None)
        if cnt>0 and cnt % period == 0:
            delta = damp * (res * res).sum() / ((delta_x * delta_x).sum() + eps)            
            x_delta = beta * res - (alpha * Xk + alpha * beta * Rk) @ (
                        pinv((Rk.t() @ Rk) + delta * (Xk.t() @ Xk)) @ (Rk.t() @ res))
            tmp = x_delta.dot(res)
            if tmp <=0:
                print("**Notice: (dir,res) = {} <= 0".format(float(tmp)))
                pass
            else:
                xk += x_delta
                self._store_data(xk)

        cnt = cnt+1
        state['step'] = cnt

        return loss
