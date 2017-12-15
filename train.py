import torch
import torch.nn as nn
import numpy as np
import module
from dataset_wrapper import MnistWrapper
import module
from kfac import KFACOptimizer
import time


def normal_grad(loss, model):
    loss.backward()


def newton_grad(loss, model):
    loss.backward(retain_graph=True)
    gs = model.get_param_g()
    grad = torch.zeros(len(gs))
    for i, g in enumerate(gs):
        grad[i] = g.grad.data[0]
    hessian = torch.zeros(len(gs), len(gs))
    # compute second order grad
    dl_dgs = torch.autograd.grad(loss, gs, create_graph=True)
    for i, dl_dg in enumerate(dl_dgs):
        ddg = torch.autograd.grad(dl_dg, gs, retain_graph=True)
        hessian[i][i] = ddg[i].data[0]

    invh_grad = torch.mv(torch.inverse(hessian), grad)
    for i, g in enumerate(gs):
        g.grad.data[0] = invh_grad[i]
    nn.utils.clip_grad_norm(gs, 1)


def train(model, dataset, grad_func, lr, kfac, num_epochs):
    batch_size = 100

    if kfac:
        optim = KFACOptimizer(model)
        optim.acc_stats = True
    else:
        optim = torch.optim.SGD(model.parameters(), lr)

    train_loss = np.zeros(num_epochs)
    train_acc = np.zeros(num_epochs)
    test_acc = np.zeros(num_epochs)
    times = np.zeros(num_epochs)

    t = time.time()
    for epoch in range(num_epochs):
        dataset.reset_and_shuffle(batch_size)
        losses = np.zeros(len(dataset))

        for i in range(len(dataset)):
            x, y = dataset.next_batch()
            x = torch.autograd.Variable(torch.from_numpy(x)).cuda()
            y = torch.autograd.Variable(torch.from_numpy(y).long()).cuda()
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            losses[i] = loss.data[0]

            grad_func(loss, model)
            optim.step()
            optim.zero_grad()

        train_loss[epoch] = losses.mean()
        train_acc[epoch] = eval_(model, dataset.train_xs, dataset.train_ys)
        test_acc[epoch] = eval_(model, dataset.test_xs, dataset.test_ys)
        times[epoch] = time.time() - t
        print 'epoch: %i, loss: %.4f' % (epoch+1, train_loss[epoch])
        print 'accumulate time', times[epoch]
        print 'train acc:', train_acc[epoch]
        print 'eval acc:', test_acc[epoch]
        print '----------------'

    return train_loss, train_acc, test_acc, times


def eval_(model, x, y):
    x = torch.autograd.Variable(torch.from_numpy(x), volatile=True).cuda()
    logits = model(x)
    _, pred = torch.max(logits, dim=1)
    pred = pred.data.cpu().numpy()
    acc = (pred == y).sum() / float(len(y))
    return acc
