# Adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/kfac.py
import math

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


# TODO: In order to make this code faster:
# 1) Implement _extract_patches as a single cuda kernel
# 2) Compute QR decomposition in a separate process
# 3) Actually make a general KFAC optimizer so it fits PyTorch


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


class AddWn(nn.Module):
    def __init__(self, g):
        super(AddWn, self).__init__()
        self._g = nn.Parameter(g.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            g = self._g.t().view(1, -1)
        else:
            g = self._g.t().view(1, -1, 1, 1)
        return x * g


def _extract_patches(x, kernel_size, stride, padding):
    if padding[0] + padding[1] > 0:
        # Actually check dims
        x = F.pad(x, (padding[1], padding[1], padding[0], padding[0])).data

    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    d0, d1, d2, d3, d4, d5 = x.size()
    x = x.view(d0, d1, d2, d3 * d4 * d5)
    return x


def compute_cov_a(a, classname, layer_info, fast_cnn):
    batch_size = a.size(0)

    if 'Conv2d' in classname:
        if fast_cnn:
            a = _extract_patches(a, *layer_info)
            a = a.view(a.size(0), -1, a.size(-1))
            a = a.mean(1)
        else:
            a = _extract_patches(a, *layer_info)
            a = a.view(-1, a.size(-1)).div_(a.size(1)).div_(a.size(2))
    elif classname == 'AddBias' or classname == 'AddWn':
        is_cuda = a.is_cuda
        a = torch.ones(a.size(0), 1)
        if is_cuda:
            a = a.cuda()

    return torch.matmul(a.t(), a / batch_size)


def compute_cov_g(g, classname, layer_info, fast_cnn):
    batch_size = g.size(0)

    if 'Conv2d' in classname:
        if fast_cnn:
            g = g.view(g.size(0), g.size(1), -1)
            g = g.sum(-1)
        else:
            g = g.transpose(1, 2).transpose(2, 3).contiguous()
            g = g.view(-1, g.size(-1)).mul_(g.size(1)).mul_(g.size(2))
    elif classname == 'AddBias' or classname == 'AddWn':
        g = g.view(g.size(0), g.size(1), -1)
        g = g.sum(-1)

    g_ = g * batch_size
    return torch.matmul(g_.t(), g_ / g.size(0))


def update_running_stat(aa, m_aa, momentum):
    # Do the trick to keep aa unchanged and not create any additional tensors
    m_aa *= momentum / (1 - momentum)
    m_aa += aa
    m_aa *= (1 - momentum)


class Split(nn.Module):
    def __init__(self, module):
        super(Split, self).__init__()
        self.module = module
        self.add_bias = AddBias(module.bias.data)
        self.module.bias = None

        if hasattr(module, 'g'):
            self.add_wn = AddWn(module.g.data)
            self.module.g = None

    def forward(self, input):
        x = self.module(input)
        if hasattr(self, 'add_wn'):
            x = self.add_wn(x)

        x = self.add_bias(x)
        return x


class KFACOptimizer(optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.25,
                 momentum=0.9,
                 stat_decay=0.99,
                 kl_clip=0.001,
                 damping=1e-2,
                 weight_decay=0,
                 fast_cnn=False,
                 Ts=1,
                 Tf=10):
        defaults = dict()

        def split(module):
            for mname, child in module.named_children():
                if hasattr(child, 'bias') or hasattr(child, 'g'):
                    module._modules[mname] = Split(child)
                else:
                    split(child)

        split(model)

        super(KFACOptimizer, self).__init__(model.parameters(), defaults)

<<<<<<< HEAD
        self.known_modules = {'Linear', 'Conv2d', 'AddBias', 'AddWn'}
=======
        self.known_modules = {'Linear', 'Conv2d', 'WnLinear', 'WnConv2d', 'AddBias', 'AddWn'}
>>>>>>> wn-kfac

        self.modules = []
        self.grad_outputs = {}

        self.model = model
        self._prepare_model()

        self.steps = 0

        self.m_aa, self.m_gg = {}, {}
        self.Q_a, self.Q_g = {}, {}
        self.d_a, self.d_g = {}, {}

        self.momentum = momentum
        self.stat_decay = stat_decay

        self.lr = lr
        self.kl_clip = kl_clip
        self.damping = damping
        self.weight_decay = weight_decay

        self.fast_cnn = fast_cnn

        self.Ts = Ts
        self.Tf = Tf

        self.optim = optim.SGD(
            model.parameters(),
            lr=self.lr * (1 - self.momentum),
            momentum=self.momentum)

    def _save_input(self, module, input):
        if input[0].volatile == False and self.steps % self.Ts == 0:
            classname = module.__class__.__name__
            layer_info = None
            if 'Conv2d' in classname:
                layer_info = (module.kernel_size, module.stride, module.padding)

            aa = compute_cov_a(input[0].data, classname, layer_info, self.fast_cnn)

            # Initialize buffers
            if self.steps == 0:
                self.m_aa[module] = aa.clone()

            update_running_stat(aa, self.m_aa[module], self.stat_decay)

    def _save_grad_output(self, module, grad_input, grad_output):
        if self.acc_stats and self.steps % self.Ts == 0:
            classname = module.__class__.__name__
            layer_info = None
            if 'Conv2d' in classname:
                layer_info = (module.kernel_size, module.stride, module.padding)

            gg = compute_cov_g(
                grad_output[0].data, classname, layer_info, self.fast_cnn)

            # Initialize buffers
            if self.steps == 0:
                self.m_gg[module] = gg.clone()

            update_running_stat(gg, self.m_gg[module], self.stat_decay)

    def _prepare_model(self):
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                assert not (
                    (classname in ['Linear', 'Conv2d', 'WnLinear', 'WnConv2d'])
                    and module.bias is not None
                ), "You must have a bias as a separate layer"

                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)

    def step(self):
        # Add weight decay
        if self.weight_decay > 0:
            for p in self.model.parameters():
                p.grad.data.add_(self.weight_decay, p.data)

        updates = {}
        for i, m in enumerate(self.modules):
            assert len(list(m.parameters())) == 1, \
                "Can handle only one parameter at the moment"
            classname = m.__class__.__name__
            p = next(m.parameters())

            la = self.damping + self.weight_decay

            if self.steps % self.Tf == 0:
                # My asynchronous implementation exists, I will add it later.
                # Experimenting with different ways to this in PyTorch.
                self.d_a[m], self.Q_a[m] = torch.symeig(
                    self.m_aa[m].double(), eigenvectors=True)
                self.d_g[m], self.Q_g[m] = torch.symeig(
                    self.m_gg[m].double(), eigenvectors=True)
                self.d_a[m] = self.d_a[m].float()
                self.Q_a[m] = self.Q_a[m].float()
                self.d_g[m] = self.d_g[m].float()
                self.Q_g[m] = self.Q_g[m].float()
                # if self.m_aa[m].is_cuda:
                #     self.d_a[m], self.Q_a[m] = self.d_a[m].cuda(), self.Q_a[m].cuda()
                #     self.d_g[m], self.Q_g[m] = self.d_g[m].cuda(), self.Q_g[m].cuda()

                self.d_a[m].mul_((self.d_a[m] > 1e-6).float())
                self.d_g[m].mul_((self.d_g[m] > 1e-6).float())

            if 'Conv2d' in classname:
                p_grad_mat = p.grad.data.view(p.grad.data.size(0), -1)
            else:
                p_grad_mat = p.grad.data

            # print classname
            # print p_grad_mat.size()
            # print self.Q_a[m].size()
            # print self.Q_g[m].t().size()

            v1 = torch.matmul(self.Q_g[m].t(), torch.matmul(p_grad_mat, self.Q_a[m]))
            v2 = v1 / (self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + la)
            v = torch.matmul(self.Q_g[m], torch.matmul(v2, self.Q_a[m].t()))

            v = v.view(p.grad.data.size())
            updates[p] = v

        vg_sum = 0
        for p in self.model.parameters():
            v = updates[p]
            vg_sum += (v * p.grad.data * self.lr * self.lr).sum()

        nu = min(1, math.sqrt(self.kl_clip / vg_sum))

        for p in self.model.parameters():
            v = updates[p]
            p.grad.data.copy_(v)
            p.grad.data.mul_(nu)

        self.optim.step()
        self.steps += 1
