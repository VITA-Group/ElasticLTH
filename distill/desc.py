# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from dataclasses import dataclass
import os

from cli import arg_utils
from datasets import registry as datasets_registry
from foundations import desc
from foundations import hparams
from foundations.step import Step
from platforms.platform import get_platform
import pruning.registry


@dataclass
class DistillDesc(desc.Desc):
    """The hyperparameters necessary to describe a training run."""

    model_hparams: hparams.ModelHparams
    dataset_hparams: hparams.DatasetHparams
    training_hparams: hparams.TrainingHparams
    pruning_hparams: hparams.PruningHparams
    distill_hparams: hparams.DistillHparams

    @staticmethod
    def name_prefix(): return 'distill'

    # @staticmethod
    # def _add_distill_argument(parser):
    #     # help_text = \
    #     #     'Perform a pre-training phase prior to running the main lottery ticket process. Setting this argument '\
    #     #     'will enable arguments to control how the dataset and training during this pre-training phase. Rewinding '\
    #     #     'is a specific case of of pre-training where pre-training uses the same dataset and training procedure '\
    #     #     'as the main training run.'
    #     parser.add_argument('--teacher-model-name', type=str, help='Model name of the teacher model')
    #     parser.add_argument('--teacher-ckpt', type=str, help='Path to the teacher model\'s checkpoint')
    #     parser.add_argument('--teacher-mask', type=str, help='Path to the teacher model\'s mask')
    #     parser.add_argument('--alpha-ce', type=float, default=1.0)
    #     parser.add_argument('--alpha-mse', type=float, default=1.0)
    #     parser.add_argumetn('--alpha-cls', type=float, default=1.0)
    #     parser.add_argumetn('--alpha-cos', type=float, default=1.0)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, defaults: 'DistillDesc' = None):
        # Get the proper pruning hparams.
        pruning_strategy = arg_utils.maybe_get_arg('pruning_strategy')
        if defaults and not pruning_strategy: pruning_strategy = defaults.pruning_hparams.pruning_strategy
        if pruning_strategy:
            pruning_hparams = pruning.registry.get_pruning_hparams(pruning_strategy)
            if defaults and defaults.pruning_hparams.pruning_strategy == pruning_strategy:
                def_ph = defaults.pruning_hparams
            else:
                def_ph = None
        else:
            pruning_hparams = hparams.PruningHparams
            def_ph = None

        # Add the main arguments.
        hparams.DatasetHparams.add_args(parser, defaults=defaults.dataset_hparams if defaults else None)
        hparams.ModelHparams.add_args(parser, defaults=defaults.model_hparams if defaults else None)
        hparams.TrainingHparams.add_args(parser, defaults=defaults.training_hparams if defaults else None)
        pruning_hparams.add_args(parser, defaults=def_ph if defaults else None)
        hparams.DistillHparams.add_args(parser, defaults=defaults.distill_hparams if defaults else None)

    @classmethod
    def create_from_args(cls, args: argparse.Namespace) -> 'DistillDesc':
        dataset_hparams = hparams.DatasetHparams.create_from_args(args)
        model_hparams = hparams.ModelHparams.create_from_args(args)
        training_hparams = hparams.TrainingHparams.create_from_args(args)
        pruning_hparams = pruning.registry.get_pruning_hparams(args.pruning_strategy).create_from_args(args)
        distill_hparams = hparams.DistillHparams.create_from_args(args)

        # Create the desc.
        desc = cls(model_hparams, dataset_hparams, training_hparams, pruning_hparams, distill_hparams)

        return desc

    def str_to_step(self, s: str, pretrain: bool = False) -> Step:
        dataset_hparams = self.pretrain_dataset_hparams if pretrain else self.dataset_hparams
        iterations_per_epoch = datasets_registry.iterations_per_epoch(dataset_hparams)
        return Step.from_str(s, iterations_per_epoch)

    @property
    def start_step(self):
        return self.str_to_step('0it')

    @property
    def end_step(self):
        iterations_per_epoch = datasets_registry.iterations_per_epoch(self.dataset_hparams)
        return Step.from_str(self.training_hparams.training_steps, iterations_per_epoch)

    @property
    def train_start_step(self):
        return self.str_to_step('0it')

    @property
    def train_end_step(self):
        return self.str_to_step(self.training_hparams.training_steps)

    @property
    def train_outputs(self):
        datasets_registry.num_classes(self.dataset_hparams)

    def run_path(self, replicate, experiment='main'):

        if not isinstance(replicate, int) or replicate <= 0:
            raise ValueError('Bad replicate: {}'.format(replicate))

        return os.path.join(get_platform().root, self.hashname, f'replicate_{replicate}', experiment)

    @property
    def display(self):
        ls = [self.dataset_hparams.display, self.model_hparams.display,
              self.training_hparams.display, self.pruning_hparams.display,
              self.distill_hparams.display]
        return '\n'.join(ls)
