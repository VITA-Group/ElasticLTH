# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F

from foundations import hparams
from lottery.desc import LotteryDesc
from models import base
from pruning import sparse_global


class Model(base.Model):
    """A MobileNet-V1 as originally designed for CIFAR-10."""

    class Block(nn.Module):
        """A MobileNet-V1 block."""

        # def __init__(self, f_in: int, f_out: int, downsample=False):
        def __init__(self, f_in: int, f_out: int, stride=1):
            super(Model.Block, self).__init__()

            self.conv1 = nn.Conv2d(f_in, f_in, kernel_size=3, stride=stride, padding=1, groups=f_in, bias=False)
            self.bn1 = nn.BatchNorm2d(f_in)
            self.conv2 = nn.Conv2d(f_in, f_out, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn2 = nn.BatchNorm2d(f_out)

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            return F.relu(out)

    def __init__(self, initializer, num_512_blocks=5, outputs=None):
        super(Model, self).__init__()
        outputs = outputs or 10

        # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
        # cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
        cfg_part1 = [64, (128,2), 128, (256,2), 256, (512,2)]
        cfg_part2 = [512] * num_512_blocks
        cfg_part3 = [(1024,2), 1024]

        # Initial convolution.
        self.conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)

        # The subsequent layers of MobileNet-V1.
        self.layers_part1 = self._make_layers(in_planes=32 , config=cfg_part1)
        self.layers_part2 = self._make_layers(in_planes=512, config=cfg_part2)
        self.layers_part3 = self._make_layers(in_planes=512, config=cfg_part3)

        # Final fc layer. Size = number of filters in last segment.
        self.fc = nn.Linear(1024, outputs)
        self.criterion = nn.CrossEntropyLoss()

        # Initialize.
        self.apply(initializer)

    def _make_layers(self, in_planes, config):
        layers = []
        for x in config:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Model.Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.layers_part1(out)
        out = self.layers_part2(out)
        out = self.layers_part3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    @property
    def output_layer_names(self):
        return ['fc.weight', 'fc.bias']

    @staticmethod
    def is_valid_model_name(model_name):
        return (model_name.startswith('cifar_mobilenetv1') and
                3 >= len(model_name.split('_')) >= 2 and
                all([x.isdigit() and int(x) > 0 for x in model_name.split('_')[2:]]) and
                (len(model_name.split('_')) == 2 or int(model_name.split('_')[2]) >= 1))

    @staticmethod
    def get_model_from_name(model_name, initializer,  outputs=10):
        """The naming scheme for a MobileNetV1 is 'cifar_mobilenetv1[_N]'.

        The name of a MobileNetV1 is 'cifar_mobilenetv1[_N]'.
        N is the total number of blocks with 512 input and output channels and stride 1.
        The default value of W is 5 if it isn't provided.
        """

        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        name = model_name.split('_')
        N = 5 if len(name) == 2 else int(name[2])

        return Model(initializer, N, outputs)

    @property
    def loss_criterion(self):
        return self.criterion

    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='cifar_mobilenetv1',
            model_init='kaiming_normal',
            batchnorm_init='uniform',
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='cifar10',
            batch_size=128,
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='sgd',
            momentum=0.9,
            milestone_steps='80ep,120ep',
            lr=0.1,
            gamma=0.1,
            weight_decay=1e-4,
            training_steps='160ep',
        )

        pruning_hparams = sparse_global.PruningHparams(
            pruning_strategy='sparse_global',
            pruning_fraction=0.2
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams)
