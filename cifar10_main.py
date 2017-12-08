import time
import torch
import torch.nn as nn
import numpy as np
import module
from train import train, eval_, normal_grad, newton_grad
from dataset_wrapper import Cifar10Wrapper
import module
import matplotlib.pyplot as plt
import random


class Cifar10Model1(nn.Module):
    def __init__(self):
        super(Cifar10Model1, self).__init__()

        model = module.Sequential()
        model.add_module('c1', module.WnConv2d(3, 32, 3, 2, 1))
        model.add_module('relu1', nn.ReLU())
        model.add_module('c2', module.WnConv2d(32, 64, 3, 2, 1))
        model.add_module('relu2', nn.ReLU())
        model.add_module('l3', module.WnConv2d(64, 128, 3, 2, 1))
        model.add_module('relu3', nn.ReLU())
        model.add_module('avg_pool', nn.AvgPool2d(4))
        self.main = model
        self.fc = module.WnLinear(128, 10)

    def get_param_g(self):
        return self.main.get_param_g() + [self.fc.get_param_g()]

    def forward(self, x):
        y = self.main(x).squeeze()
        y = self.fc(y)
        return y


class Cifar10Model2(nn.Module):
    def __init__(self):
        super(Cifar10Model2, self).__init__()

        model = module.Sequential()
        model.add_module('c1', nn.Conv2d(3, 32, 3, 2, 1))
        model.add_module('relu1', nn.ReLU())
        model.add_module('c2', nn.Conv2d(32, 64, 3, 2, 1))
        model.add_module('relu2', nn.ReLU())
        model.add_module('l3', nn.Conv2d(64, 128, 3, 2, 1))
        model.add_module('relu3', nn.ReLU())
        model.add_module('avg_pool', nn.AvgPool2d(4))
        self.main = model
        self.fc = nn.Linear(128, 10)

    def get_param_g(self):
        return self.main.get_param_g() + [self.fc.get_param_g()]

    def forward(self, x):
        y = self.main(x).squeeze()
        y = self.fc(y)
        return y


def set_all_seeds(rand_seed):
    def large_randint():
        return random.randint(int(1e5), int(1e6))

    random.seed(rand_seed)
    np.random.seed(large_randint())
    torch.manual_seed(large_randint())
    torch.cuda.manual_seed(large_randint())


if __name__ == '__main__':
    set_all_seeds(100009)

    dataset = Cifar10Wrapper.load_default()
    dataset.train_ys = dataset.train_ys.astype(np.int32).reshape((-1,))
    dataset.test_ys = dataset.test_ys.astype(np.int32).reshape((-1,))
    print dataset.train_ys.min()
    print dataset.train_xs.min(), dataset.train_xs.max()
    print dataset.train_xs.shape

    # model = Cifar10Model1().cuda()
    # print model
    # t = time.time()
    # newton_loss, newton_acc = train(model, dataset, newton_grad, 0.1)
    # print 'time:', time.time() - t

    # model = Cifar10Model1().cuda()
    model = Cifar10Model2().cuda()
    t = time.time()
    normal_loss, normal_acc = train(model, dataset, normal_grad, 0.1)
    print 'time:', time.time() - t

    x = range(len(newton_loss))
    plt.figure()
    # plt.plot(x, newton_loss, 'r-', label='Newton Loss')
    plt.plot(x, normal_loss, 'b-', label='Normal Loss')
    plt.legend(loc='upper right')
    plt.savefig('cifar10_train_loss.png')

    plt.figure()
    # plt.plot(x, newton_acc, 'r-', label='newton Acc')#  t, t**3, 'g^')
    plt.plot(x, normal_acc, 'b-', label='normal Acc')#  t, t**3, 'g^')
    plt.legend(loc='lower right')
    plt.savefig('cifar10_test_acc.png')
