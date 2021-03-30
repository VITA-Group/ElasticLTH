# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os


def checkpoint(root, suffix=""):
    return os.path.join(root, 'checkpoint{}.pth'.format(suffix))


def logger(root, suffix=""):
    return os.path.join(root, 'logger' + suffix)


def mask(root, suffix=""):
    return os.path.join(root, 'mask{}.pth'.format(suffix))


def sparsity_report(root, suffix=""):
    return os.path.join(root, 'sparsity_report{}.json'.format(suffix))


def model(root, step, suffix=""):
    return os.path.join(root, 'model_ep{}_it{}{}.pth'.format(step.ep, step.it, suffix))


def hparams(root, suffix=""):
    return os.path.join(root, 'hparams{}.log'.format(suffix))
