# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import torch

from lottery.branch import base
import models.registry
import pruning.registry
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel
from training import train
from platforms.platform import get_platform
from utils.tensor_utils import vectorize, unvectorize, shuffle_tensor, shuffle_state_dict


class Branch(base.Branch):
    def branch_function(self,
                        seed: int,
                        strategy: str = 'sparse_global',
                        start_at: str = 'rewind',
                        layers_to_ignore: str = ''):
        # Reset the masks of any layers that shouldn't be pruned.
        if layers_to_ignore:
            for k in layers_to_ignore.split(','): mask[k] = torch.ones_like(mask[k])

        # Determine the start step.
        if start_at == 'init':
            start_step = self.lottery_desc.str_to_step('0ep')
            state_step = start_step
        elif start_at == 'end':
            start_step = self.lottery_desc.str_to_step('0ep')
            state_step = self.lottery_desc.train_end_step
        elif start_at == 'rewind':
            start_step = self.lottery_desc.train_start_step
            state_step = start_step
        else:
            raise ValueError(f'Invalid starting point {start_at}')

        # Train the model with the new mask.
        model = models.registry.load(self.pretrain_root, state_step, self.lottery_desc.model_hparams)

        # Get the current level mask and get the target pruning ratio
        mask = Mask.load(self.level_root)
        sparsity_ratio = mask.get_sparsity_ratio()
        target_pruning_fraction = 1.0 - sparsity_ratio

        # Run pruning
        pruning_hparams = copy.deepcopy(self.lottery_desc.pruning_hparams)
        pruning_hparams.pruning_strategy = strategy
        pruning_hparams.pruning_fraction = target_pruning_fraction
        new_mask = pruning.registry.get(pruning_hparams)(
            model, Mask.ones_like(model),
            self.lottery_desc.training_hparams,
            self.lottery_desc.dataset_hparams, seed
        )
        new_mask.save(self.branch_root)

        repruned_model = PrunedModel(model.to(device=get_platform().cpu_device), new_mask)

        # Run training
        train.standard_train(repruned_model, self.branch_root, self.lottery_desc.dataset_hparams,
                             self.lottery_desc.training_hparams, start_step=start_step, verbose=self.verbose)

    @staticmethod
    def description():
        return "Reprune the model using early pruning methods."

    @staticmethod
    def name():
        return 'prune_early'

