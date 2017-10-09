import torch
import torch.nn as nn
import numpy as np
import module
from dataset_wrapper import MnistWrapper
import module


def normal_grad(loss, model):
    loss.backward()


def newton_grad(loss, model):
    loss.backward(retain_graph=True)
    gs = model.get_param_g()
    # print 'gs:', gs
    grad = torch.zeros(len(gs))
    # print gs[0].grad.data[0]
    for i, g in enumerate(gs):
        grad[i] = g.grad.data[0]
    hessian = torch.zeros(len(gs), len(gs))
    # compute second order grad
    dl_dgs = torch.autograd.grad(loss, gs, create_graph=True)
    # ddl_ddg = torch.autograd.grad(dl_dgs, gs)
    # print '>>>>>>>>>>>>>>', ddl_ddg
    for i, dl_dg in enumerate(dl_dgs):
        ddg = torch.autograd.grad(dl_dg, gs, retain_graph=True)
        # print 'ddg', ddg
        hessian[i][i] = ddg[i].data[0]
        # for j in range(len(ddg)):
        #     hessian[i][j] = ddg[j].data[0]
    # print hessian
    invh_grad = torch.mv(torch.inverse(hessian), grad)
    for i, g in enumerate(gs):
        g.grad.data[0] = invh_grad[i]
    nn.utils.clip_grad_norm(gs, 1)

    # print gs[0].grad.data[0]
    # print '--------'

    # print hessian, grad, invh_grad
    # print ddl_ddg.data[0]
    # print g.grad.data[0]
    # g.grad.data /= ddl_ddg.data
    # print g.grad.data[0]
    # dl_dg.backward()


def train(model, dataset, grad_func, lr):
    batch_size = 100
    num_epochs = 20
    # optim = torch.optim.RMSprop(model.parameters(), lr)
    optim = torch.optim.SGD(model.parameters(), lr)

    train_loss = np.zeros(num_epochs)
    test_acc = np.zeros(num_epochs)

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
            # break

        # break
        train_loss[epoch] = losses.mean()
        test_acc[epoch] = eval_(model, dataset.test_xs, dataset.test_ys)
        print 'epoch: %i, loss: %.4f' % (epoch+1, train_loss[epoch])
        print 'eval acc:', test_acc[epoch]
    return train_loss, test_acc


def eval_(model, x, y):
    x = torch.autograd.Variable(torch.from_numpy(x), volatile=True).cuda()
    logits = model(x)
    _, pred = torch.max(logits, dim=1)
    pred = pred.data.cpu().numpy()
    acc = (pred == y).sum() / float(len(y))
    return acc
