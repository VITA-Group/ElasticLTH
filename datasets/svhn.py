# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
from PIL import Image
import sys
import torchvision

from datasets import base
from platforms.platform import get_platform


class SVHN(torchvision.datasets.SVHN):
    """A subclass to suppress an annoying print statement in the torchvision CIFAR-10 library.

    Not strictly necessary - you can just use `torchvision.datasets.CIFAR10 if the print
    message doesn't bother you.
    """

    def download(self):
        if get_platform().is_primary_process:
            with get_platform().open(os.devnull, 'w') as fp:
                sys.stdout = fp
                super(SVHN, self).download()
                sys.stdout = sys.__stdout__
        get_platform().barrier()


class Dataset(base.ImageDataset):
    """The SVHN dataset."""

    @staticmethod
    def num_train_examples(): return 73257

    @staticmethod
    def num_test_examples(): return 26032

    @staticmethod
    def num_classes(): return 10

    @staticmethod
    def get_train_set(use_augmentation, resize):
        augment = [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomCrop(32, 4)]
        train_set = SVHN(split='train', root=os.path.join(get_platform().dataset_root, 'svhn'), download=True)
        return Dataset(train_set.data, np.array(train_set.labels), augment if use_augmentation else [])

    @staticmethod
    def get_test_set(resize):
        test_set = SVHN(split='test', root=os.path.join(get_platform().dataset_root, 'svhn'), download=True)
        return Dataset(test_set.data, np.array(test_set.labels))

    def __init__(self,  examples, labels, image_transforms=None):
        super(Dataset, self).__init__(examples, labels, image_transforms or [],
                                      [torchvision.transforms.Normalize([0.438, 0.444, 0.473], [0.198, 0.201, 0.197])])

    def example_to_image(self, example):
        # return Image.fromarray(example)
        return Image.fromarray(np.transpose(example, (1, 2, 0)))


DataLoader = base.DataLoader
