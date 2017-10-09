"""Wrapper classes for original and encoded datasets."""
from __future__ import print_function
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import h5py
# import utils
# import stl_dataset
import torch
import torchvision
from PIL import Image
from multiprocessing import Pool
from numba import autojit, prange


class DatasetWrapper(object):
    idx2cls = []

    def __init__(self, train_xs, train_ys, test_xs, test_ys):
        """DO NOT do any normalization in this function"""
        self.train_xs = train_xs.astype(np.float32)
        self.train_ys = train_ys
        self.test_xs = test_xs.astype(np.float32)
        self.test_ys = test_ys
        self.batch_size = None
        self.batch_idx = None

    def __len__(self):
        return int(np.ceil(1.0 * len(self.train_xs) / self.batch_size))

    @property
    def x_shape(self):
        return self.train_xs.shape[1:]

    @property
    def cls2idx(self):
        return {cls: idx for (idx, cls) in enumerate(idx2cls)}

    @classmethod
    def load_from_h5(cls, h5_path):
        with h5py.File(h5_path, 'r') as hf:
            train_xs = np.array(hf.get('train_xs'))
            train_ys = np.array(hf.get('train_ys')) if 'train_ys' in hf else None
            test_xs = np.array(hf.get('test_xs'))
            test_ys = np.array(hf.get('test_ys')) if 'test_ys' in hf else None
        print('Dataset loaded from %s' % h5_path)
        return cls(train_xs, train_ys, test_xs, test_ys)

    @classmethod
    def load_default(cls, regenerate):
        """Load the dataset in default format.

        value: (uint8, 0-255)
        shape: (num_imgs, nc, img_size, img_size)
        """
        raise NotImplementedError

    def dump_to_h5(self, h5_path):
        with h5py.File(h5_path, 'w') as hf:
            hf.create_dataset('train_xs', data=self.train_xs)
            if self.train_ys is not None:
                hf.create_dataset('train_ys', data=self.train_ys)
            hf.create_dataset('test_xs', data=self.test_xs)
            if self.test_ys is not None:
                hf.create_dataset('test_ys', data=self.test_ys)
        print('Dataset written to %s' % h5_path)

    # def need_reset(self):
    #     return self.batch_idx is None or self.batch_idx >= len(self)

    def reset_and_shuffle(self, batch_size, shuffle=True):
        assert self.train_xs.min() >= -1.0, 'may not be suitable for training'
        assert self.train_xs.max() <= 1.0, 'may not be suitable for training'

        self.batch_size = batch_size
        self.batch_idx = 0
        if not shuffle:
            return
        shuffled_order = range(len(self.train_xs))
        np.random.shuffle(shuffled_order)
        self.train_xs = self.train_xs[shuffled_order]
        if self.train_ys is not None:
            self.train_ys = self.train_ys[shuffled_order]

    def next_batch(self):
        self.batch_idx += 1
        batch = self.train_xs[
            (self.batch_idx-1)*self.batch_size : self.batch_idx*self.batch_size]
        label = self.train_ys[
            (self.batch_idx-1)*self.batch_size : self.batch_idx*self.batch_size]
        return batch, label

    def transform(self, func):
        self.train_xs = func(self.train_xs)
        self.test_xs = func(self.test_xs)
        return self

    def reshape(self, new_shape):
        num_imgs = self.train_xs.shape[0]
        self.train_xs = self.train_xs.reshape((num_imgs,) + new_shape)
        num_imgs = self.test_xs.shape[0]
        self.test_xs = self.test_xs.reshape((num_imgs,) + new_shape)
        assert self.train_xs.shape[1:] == self.test_xs.shape[1:]

    def resize(self, new_shape):
        self.train_xs = utils.resize_nparray(self.train_xs, new_shape)
        self.test_xs = utils.resize_nparray(self.test_xs, new_shape)

    # def rescale(self, low, high):
    #     old_low = min(self.train_xs.min(), self.test_xs.min())
    #     old_range = max(self.train_xs.max(), self.test_xs.max()) - old_low
    #     new_range = high - low
    #     transform = lambda x: (x - old_low) / old_range * new_range + low
    #     self.train_xs = transform(self.train_xs).astype(np.float32, copy=False)
    #     self.test_xs = transform(self.test_xs).astype(np.float32, copy=False)

    def plot_data_dist(self, fig_path, num_bins=50):
        xs = np.vstack((self.train_xs, self.test_xs))
        if len(xs.shape) > 2:
            num_imgs = len(xs)
            xs = xs.reshape((num_imgs, -1))
        w, h = plt.figaspect(1.)
        plt.figure(figsize=(w, h))
        plt.hist(xs, num_bins)
        if fig_path:
            plt.savefig(fig_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    # def get_subset(self, subset, subclass=None):
    #     """get a subset.
    #     subset: 'train' or 'test'
    #     subclass: name of the subclass of interest
    #     """
    #     xs = self.train_xs if subset == 'train' else self.test_xs
    #     ys = self.train_ys if subset == 'train' else self.test_ys
    #     assert len(xs) == len(ys)
    #     if subclass:
    #         idx = self.cls2idx[subclass]
    #         loc = np.where(ys == idx)[0]
    #         xs = xs[loc]
    #         ys = ys[loc]
    #     return xs, ys


class MnistWrapper(DatasetWrapper):
    dataroot = '/home/hengyuah/datasets/mnist'

    @classmethod
    def load_default(cls, regenerate=False):
        """Load Mnist, rescale to [0, 1] and reshape to (1, 28, 28)."""
        h5_path = os.path.join(cls.dataroot, 'mnist.h5')
        if not regenerate and os.path.exists(h5_path):
            return cls.load_from_h5(h5_path)

        train_dset = torchvision.datasets.MNIST(
            cls.dataroot, train=True, download=True)
        test_dset = torchvision.datasets.MNIST(
            cls.dataroot, train=False, download=True)

        train_xs = train_dset.train_data.float().numpy()
        train_ys = train_dset.train_labels.float().numpy().reshape((-1, 1))
        test_xs = test_dset.test_data.float().numpy()
        test_ys = test_dset.test_labels.float().numpy().reshape((-1, 1))

        assert train_xs.min() == 0, train_xs.max() == 255
        train_xs = train_xs.reshape(-1, 1, 28, 28) / 255.0
        test_xs = test_xs.reshape(-1, 1, 28, 28) / 255.0
        dataset = cls(train_xs, train_ys, test_xs, test_ys)
        dataset.dump_to_h5(h5_path)
        return dataset


class StaticBinaryMnistWrapper(DatasetWrapper):
    dataroot = '/home/hengyuah/datasets/BinaryMNIST'
    # dataroot = '/Users/hhu/Developer/datasets/BinaryMNIST'

    @classmethod
    def load_default(cls, regenerate=False):
        h5_path = os.path.join(cls.dataroot, 'binary_mnist.h5')
        if not regenerate and os.path.exists(h5_path):
            return cls.load_from_h5(h5_path)

        def lines_to_np_array(lines):
            return np.array([[int(i) for i in line.split()] for line in lines])

        datafiles = ['binarized_mnist_train.amat',
                     'binarized_mnist_valid.amat',
                     'binarized_mnist_test.amat']
        data_arrays = []
        for dfile in datafiles:
            with open(os.path.join(cls.dataroot, dfile), 'r') as f:
                lines = f.readlines()
            data_arrays.append(lines_to_np_array(lines))
        train, val, test = data_arrays
        train_xs = np.concatenate([train, val]).reshape(-1, 1, 28, 28)
        test_xs = test.reshape(-1, 1, 28, 28)
        dataset = cls(train_xs, None, test_xs, None)
        dataset.dump_to_h5(h5_path)
        return dataset


class Cifar10Wrapper(DatasetWrapper):
    idx2cls = ['airplane' , 'automobile', 'bird'  , 'cat' , 'deer',
               'dog'      , 'frog'      ,  'horse', 'ship', 'truck']
    dataroot = '/home/hengyuah/datasets/cifar10'

    @classmethod
    def load_default(cls, regenerate=False):
        """Load Cifar10, rescale to [-1, 1], default shape: (1, 28, 28)."""
        h5_path = os.path.join(cls.dataroot, 'cifar10.h5')
        if not regenerate and os.path.exists(h5_path):
            return cls.load_from_h5(h5_path)

        train_dset = torchvision.datasets.CIFAR10(
            cls.dataroot, train=True, download=True)
        test_dset = torchvision.datasets.CIFAR10(
            cls.dataroot, train=False, download=True)

        train_xs = train_dset.train_data.transpose((0, 3, 1, 2))
        train_ys = np.array(train_dset.train_labels).reshape((-1, 1))
        test_xs = test_dset.test_data.transpose((0, 3, 1, 2))
        test_ys = np.array(test_dset.test_labels).reshape((-1, 1))

        assert train_xs.min() == 0, train_xs.max() == 255
        assert test_xs.min() == 0, test_xs.max() == 255
        train_xs = train_xs / 255.0 * 2.0 - 1.0
        test_xs = test_xs / 255.0 * 2.0 - 1.0
        dataset = cls(train_xs, train_ys, test_xs, test_ys)
        dataset.dump_to_h5(h5_path)
        return dataset


def load_folder(folder, suffix):
    imgs = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(suffix):
            imgs.append(os.path.join(folder, f))
    return imgs


def load_data(files):
    files = files
    imgs = np.zeros((len(files), 64, 64, 3), dtype=np.float32)

    for i in prange(len(files)):
        img = np.array(Image.open(files[i]), dtype=np.float32)
        imgs[i] = img
    imgs = imgs.transpose((0, 3, 1, 2))
    imgs = imgs / 255.0 * 2. - 1.
    return imgs


class ImageNet64Wrapper(object):
    dataroot = '/home/hengyuah/datasets/imagenet64'

    def __init__(self):
        train_root = os.path.join(self.dataroot, 'train')
        test_root = os.path.join(self.dataroot, 'valid/valid_64x64')
        # train_dset = torchvision.datasets.ImageFolder(train_root)
        # test_dset = torchvision.datasets.ImageFolder(test_root)

        data_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5])
        ])
        self.test_xs = np.array(load_data(load_folder(test_root, 'png')))
        self.train_dset = torchvision.datasets.ImageFolder(
            train_root, transform=data_transform)
        # self.train_xs = np.array(load_folder(train_root+'/train_64x64', 'png'))

    def get_train_dataloader(self, batch_size, drop_last=True):
        loader = torch.utils.data.DataLoader(
            self.train_dset,
            batch_size=batch_size, shuffle=True,
            num_workers=4, drop_last=drop_last)
        return loader


    def __len__(self):
        return int(np.ceil(1.0 * len(self.train_xs) / self.batch_size))

    def reset_and_shuffle(self, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.batch_idx = 0
        if not shuffle:
            return
        shuffled_order = range(len(self.train_xs))
        np.random.shuffle(shuffled_order)
        # self.train_xs = self.train_xs[shuffled_order]

    # def next_batch(self):
    #     self.batch_idx += 1
    #     batch_file = self.train_xs[
    #         (self.batch_idx-1)*self.batch_size : self.batch_idx*self.batch_size]
    #     batch = np.zeros((self.batch_size, 64, 64, 3), dtype=np.float32)
    #     for i, f in enumerate(batch_file):
    #         img = np.array(Image.open(f), dtype=np.float32)
    #         batch[i] = img
    #     batch = batch.transpose((0, 3, 1, 2))
    #     batch = batch / 255.0 * 2. - 1.
    #     return batch


# class STL10Wrapper(DatasetWrapper):
#     @classmethod
#     def load_default(cls):
#         train_xs = stl_dataset.read_all_images(stl_dataset.UNLABELED_DATA_PATH)
#         train_ys = np.zeros(len(train_xs), dtype=np.uint8)
#         test_xs = stl_dataset.read_all_images(stl_dataset.DATA_PATH)
#         test_ys = stl_dataset.read_labels(stl_dataset.LABEL_PATH)

#         assert train_xs.min() == 0, train_xs.max() == 255
#         assert test_xs.min() == 0, test_xs.max() == 255
#         train_xs = train_xs / 255.0 * 2.0 - 1.0
#         test_xs = test_xs / 255.0 * 2.0 - 1.0
#         return cls(train_xs, train_ys, test_xs, test_ys)


def test_dataset(output='tests'):
    output = os.path.join(output, 'test_dataset_wrapper')
    if not os.path.exists(output):
        os.makedirs(output)
    datasets = {}
    datasets['mnist'] = MnistWrapper.load_default(regenerate=True)
    datasets['mnist_round'] = MnistWrapper.load_default().transform(np.around)
    datasets['binary_mnist'] = StaticBinaryMnistWrapper.load_default()
    datasets['cifar10'] = Cifar10Wrapper.load_default(regenerate=True)
    batch_size = 100
    for name, dset in datasets.iteritems():
        img_path = os.path.join(output, '%s.png' % name)
        dset.reset_and_shuffle(batch_size, shuffle=False)
        imgs = dset.next_batch()
        torchvision.utils.save_image(
            torch.from_numpy(imgs), img_path, nrow=10, normalize=True)


if __name__ == '__main__':
    # binary_mnist = StaticBinaryMnistWrapper.load_default()
    # test_dataset()
    import time

    imgnet = ImageNet64Wrapper()
    dataloader = imgnet.get_train_dataloader(100)
    t = time.time()
    for b, (x, _) in enumerate(dataloader):
        y = x
        if b >= len(dataloader) / 10:
            break
    print('time: %.2f' % (time.time() - t))

    imgnet.reset_and_shuffle(100)
    print('num batches:', len(imgnet))
    t = time.time()
    for i in xrange(len(imgnet)/10):
        imgnet.next_batch()
    print('time: %.2f' % (time.time() - t))
    # _load_data(imgnet.train_xs)
