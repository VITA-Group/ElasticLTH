# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
from PIL import Image
import sys
import torchvision
from torchvision import transforms

from datasets import base
from platforms.platform import get_platform

normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                std=[0.2023, 0.1994, 0.2010])

class CIFAR10(torchvision.datasets.CIFAR10):
    """A subclass to suppress an annoying print statement in the torchvision CIFAR-10 library.

    Not strictly necessary - you can just use `torchvision.datasets.CIFAR10 if the print
    message doesn't bother you.
    """

    def download(self):
        if get_platform().is_primary_process:
            with get_platform().open(os.devnull, 'w') as fp:
                sys.stdout = fp
                super(CIFAR10, self).download()
                sys.stdout = sys.__stdout__
        get_platform().barrier()


class Dataset(base.ImageDataset):
    """The CIFAR-10 dataset."""

    @staticmethod
    def num_train_examples(): return 50000

    @staticmethod
    def num_test_examples(): return 10000

    @staticmethod
    def num_classes(): return 10

    @staticmethod
    def get_train_set(use_augmentation, resize):
        augment = [
            # torchvision.transforms.RandomHorizontalFlip(),
            # torchvision.transforms.RandomCrop(32, 4)
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            # normalize
        ]
        train_set = CIFAR10(train=True, root=os.path.join(get_platform().dataset_root, 'cifar10'), download=True)
        return Dataset(train_set.data, np.array(train_set.targets), augment if use_augmentation else [])

    @staticmethod
    def get_test_set(resize):
        augment = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # transforms.ToTensor(),
            # normalize
        ]
        test_set = CIFAR10(train=False, root=os.path.join(get_platform().dataset_root, 'cifar10'), download=True)
        return Dataset(test_set.data, np.array(test_set.targets), augment)

    def __init__(self,  examples, labels, image_transforms=None):
        super(Dataset, self).__init__(examples, labels, image_transforms or [], [normalize])
            # [torchvision.transforms.Normalize(
            #     [0.485, 0.456, 0.406],
            #     [0.229, 0.224, 0.225]
            # )])

    def example_to_image(self, example):
        return Image.fromarray(example)


DataLoader = base.DataLoader
