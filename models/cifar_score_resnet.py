# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from foundations import hparams
from lottery.desc import LotteryDesc
from distill.desc import DistillDesc
from models import base
from pruning import sparse_global


class ScoreConv2d(nn.Conv2d):
    """docstring for SeedConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ScoreConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        
        self.score = nn.Parameter(torch.ones_like(self.weight))

        # Disable the learning of the weight and bias
        self.disable_weight_training()

    def disable_weight_training(self):
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def enable_weight_training(self):
        self.weight.requires_grad_(True)
        if self.bias is not None:
            self.bias.requires_grad_(True)

    def forward(self, input):
        weight = self.weight * self.score
        output = F.conv2d(input, weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)
        return output

class ScoreLinear(nn.Linear):
    """[summary]

    Args:
        nn ([type]): [description]

    Raises:
        ValueError: [description]
        ValueError: [description]

    Returns:
        [type]: [description]
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias=bias)

        self.score = nn.Parameter(torch.ones_like(self.weight))

        # Disable the learning of the weight and bias
        self.disable_weight_training()

    def disable_weight_training(self):
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def enable_weight_training(self):
        self.weight.requires_grad_(True)
        if self.bias is not None:
            self.bias.requires_grad_(True)

    def forward(self, input):
        weight = self.weight * self.score
        output = F.linear(input, weight, self.bias)
        return output


class Model(base.Model):
    """A residual neural network as originally designed for CIFAR-10."""

    class Block(nn.Module):
        """A ResNet block."""

        def __init__(self, f_in: int, f_out: int, downsample=False):
            super(Model.Block, self).__init__()

            stride = 2 if downsample else 1
            self.conv1 = ScoreConv2d(f_in, f_out, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(f_out)
            self.conv2 = ScoreConv2d(f_out, f_out, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(f_out)

            # No parameters for shortcut connections.
            if downsample or f_in != f_out:
                self.shortcut = nn.Sequential(
                    ScoreConv2d(f_in, f_out, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(f_out)
                )
            else:
                self.shortcut = nn.Sequential()

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            return F.relu(out)

    def __init__(self, plan, initializer, outputs=None):
        super(Model, self).__init__()
        outputs = outputs or 10

        # Initial convolution.
        current_filters = plan[0][0]
        self.conv = ScoreConv2d(3, current_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(current_filters)

        # The subsequent blocks of the ResNet.
        blocks = []
        for segment_index, (filters, num_blocks) in enumerate(plan):
            for block_index in range(num_blocks):
                downsample = segment_index > 0 and block_index == 0
                blocks.append(Model.Block(current_filters, filters, downsample))
                current_filters = filters

        self.blocks = nn.Sequential(*blocks)

        # Final fc layer. Size = number of filters in last segment.
        self.fc = ScoreLinear(plan[-1][0], outputs)
        self.criterion = nn.CrossEntropyLoss()

        # Initialize.
        self.apply(initializer)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.blocks(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def copy_score_to_weight(self):
        for m in self.modules():
            if isinstance(m, (ScoreConv2d, ScoreLinear)):
                with torch.no_grad():
                    m.weight.data = m.score.data.detach()

    def apply_score_to_weight(self):
        for m in self.modules():
            if isinstance(m, (ScoreConv2d, ScoreLinear)):
                with torch.no_grad():
                    m.weight.data *= m.score.data.detach()

    @property
    def output_layer_names(self):
        return ['fc.weight', 'fc.bias']

    @staticmethod
    def is_valid_model_name(model_name):
        return (model_name.startswith('cifar_score-resnet_') and
                5 > len(model_name.split('_')) > 2 and
                all([x.isdigit() and int(x) > 0 for x in model_name.split('_')[2:]]) and
                (int(model_name.split('_')[2]) - 2) % 6 == 0 and
                int(model_name.split('_')[2]) > 2)

    @staticmethod
    def get_model_from_name(model_name, initializer,  outputs=10):
        """The naming scheme for a ResNet is 'cifar_resnet_N[_W]'.

        The ResNet is structured as an initial convolutional layer followed by three "segments"
        and a linear output layer. Each segment consists of D blocks. Each block is two
        convolutional layers surrounded by a residual connection. Each layer in the first segment
        has W filters, each layer in the second segment has 32W filters, and each layer in the
        third segment has 64W filters.

        The name of a ResNet is 'cifar_resnet_N[_W]', where W is as described above.
        N is the total number of layers in the network: 2 + 6D.
        The default value of W is 16 if it isn't provided.

        For example, ResNet-20 has 20 layers. Exclusing the first convolutional layer and the final
        linear layer, there are 18 convolutional layers in the blocks. That means there are nine
        blocks, meaning there are three blocks per segment. Hence, D = 3.
        The name of the network would be 'cifar_resnet_20' or 'cifar_resnet_20_16'.
        """

        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        name = model_name.split('_')
        W = 16 if len(name) == 3 else int(name[3])
        D = int(name[2])
        if (D - 2) % 3 != 0:
            raise ValueError('Invalid ResNet depth: {}'.format(D))
        D = (D - 2) // 6
        plan = [(W, D), (2*W, D), (4*W, D)]

        return Model(plan, initializer, outputs)

    @property
    def loss_criterion(self):
        return self.criterion

    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='cifar_score-resnet_20',
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
            pruning_fraction=0.9
        )

        distill_hparams = hparams.DistillHparams(
            teacher_model_name='cifar_resnet_20',
            teacher_ckpt=None,
            teacher_mask=None,
            alpha_ce=1.0,
            alpha_mse=1.0,
            alpha_cls=0.0,
            alpha_cos=0.0,
            temperature=2.0
        )

        return DistillDesc(model_hparams, dataset_hparams, training_hparams,
                           pruning_hparams, distill_hparams)
