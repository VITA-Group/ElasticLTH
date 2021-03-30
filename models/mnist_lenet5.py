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
    '''A LeNet-5 model for MNIST-10'''

    def __init__(self, initializer, outputs=10):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(1,6,kernel_size=5, bias=False)
        self.conv2 = nn.Conv2d(6,16,kernel_size=5, bias=False)
        self.conv3 = nn.Conv2d(16,120,kernel_size=5, bias=False)
        self.fc1 = nn.Conv2d(120,84,kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(84,outputs,kernel_size=1, bias=False)
        self.tanh = nn.Tanh()
        self.avg1 = nn.AvgPool2d((2,2))
        self.avg2 = nn.AvgPool2d((2,2))

        self.criterion = nn.CrossEntropyLoss()

        self.apply(initializer)

    def forward(self, x):
        out = self.conv1(x)
        out = self.tanh(out)
        out = self.avg1(out)

        out = self.conv2(out)
        out = self.tanh(out)
        out = self.avg2(out)

        out = self.conv3(out)
        out = self.tanh(out)

        out = self.fc1(out)
        out = self.tanh(out)
        out = self.fc2(out)

        return out.view(out.size(0), -1)

    @property
    def output_layer_names(self):
        return ['fc2.weight', 'fc2.bias']

    @staticmethod
    def is_valid_model_name(model_name):
        return model_name == 'mnist_lenet5'
        # return (model_name.startswith('mnist_lenet5') and
        #         len(model_name.split('_')) == 2 and
        #         all([x.isdigit() and int(x) > 0 for x in model_name.split('_')[2:]]))

    @staticmethod
    def get_model_from_name(model_name, initializer, outputs=None):
        """The name of a model is mnist_lenet_N1[_N2...].

        N1, N2, etc. are the number of neurons in each fully-connected layer excluding the
        output layer (10 neurons by default). A LeNet with 300 neurons in the first hidden layer,
        100 neurons in the second hidden layer, and 10 output neurons is 'mnist_lenet_300_100'.
        """

        outputs = outputs or 10

        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        # plan = [int(n) for n in model_name.split('_')[2:]]
        return Model(initializer, outputs)

    @property
    def loss_criterion(self):
        return self.criterion

    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='mnist_lenet5',
            model_init='kaiming_normal',
            batchnorm_init='uniform'
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='mnist',
            batch_size=128,
            # resize_input=False
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='adam',
            lr=0.001,
            training_steps='20ep',
        )

        pruning_hparams = sparse_global.PruningHparams(
            pruning_strategy='sparse_global',
            pruning_fraction=0.2,
            # pruning_layers_to_ignore='fc.weight',
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams)
