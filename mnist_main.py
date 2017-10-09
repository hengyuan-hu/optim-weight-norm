import time
import torch
import torch.nn as nn
import numpy as np
import module
from train import train, eval_, normal_grad, newton_grad
from dataset_wrapper import MnistWrapper
import module
import matplotlib.pyplot as plt
import random


def create_model1():
    model = module.Sequential()
    model.add_module('l1', module.WnLinear(784, 200))
    model.add_module('relu1', nn.ReLU())
    model.add_module('l2', module.WnLinear(200, 10))
    return model


def set_all_seeds(rand_seed):
    def large_randint():
        return random.randint(int(1e5), int(1e6))

    random.seed(rand_seed)
    np.random.seed(large_randint())
    torch.manual_seed(large_randint())
    torch.cuda.manual_seed(large_randint())


if __name__ == '__main__':
    set_all_seeds(100009)

    dataset = MnistWrapper.load_default()
    dataset.reshape((-1,))
    dataset.train_ys = dataset.train_ys.astype(np.int32).reshape((-1,))
    dataset.test_ys = dataset.test_ys.astype(np.int32).reshape((-1,))
    print dataset.train_xs.min(), dataset.train_xs.max()
    print dataset.train_xs.shape

    # model = module.Sequential(module.WnLinear(784, 10)).cuda()
    model = create_model1().cuda()
    print model
    t = time.time()
    newton_loss, newton_acc = train(model, dataset, newton_grad, 0.1)
    print 'time:', time.time() - t

    model = create_model1().cuda()
    # model = module.Sequential(module.WnLinear(784, 10)).cuda()
    t = time.time()
    normal_loss, normal_acc = train(model, dataset, normal_grad, 0.1)
    print 'time:', time.time() - t

    x = range(len(newton_loss))
    plt.figure()
    plt.plot(x, newton_loss, 'r-', label='Newton Loss')
    plt.plot(x, normal_loss, 'b-', label='Normal Loss')
    plt.legend(loc='upper right')
    plt.savefig('train_loss.png')

    plt.figure()
    plt.plot(x, newton_acc, 'r-', label='newton Acc')#  t, t**3, 'g^')
    plt.plot(x, normal_acc, 'b-', label='normal Acc')#  t, t**3, 'g^')
    plt.legend(loc='lower right')
    plt.savefig('test_acc.png')
