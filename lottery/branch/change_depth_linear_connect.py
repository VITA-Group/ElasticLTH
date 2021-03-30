# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import copy
import numpy as np
import torch
from lottery.branch import base
import models.registry
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel
from training import train
from lottery.branch.morphism import change_depth


def linear_interpolate(pruned_model_a, pruned_model_b, alpha):
    model_c = copy.deepcopy(pruned_model_a)
    sd_c = model_c.state_dict()
    for (ka, va), (kb, vb) in zip(pruned_model_a.state_dict().items(), pruned_model_b.state_dict().items()):
        assert ka == kb
        if 'mask' in ka:
            assert torch.all(va == vb)
            sd_c[ka].data = va.detach().clone()
        else:
            sd_c[ka].data = (va * alpha + vb * (1 - alpha)).detach().clone()
    model_c.load_state_dict(sd_c)
    return model_c


def parse_block_mapping_for_stage(string):
    mapping = dict()
    mapping_strs = string.split(';')
    try:
        for s in mapping_strs:
            src_id_str, tgt_ids_str = s.split(':')
            src_id = int(src_id_str)
            tgt_ids = [int(t) for t in tgt_ids_str.split(',')]
            mapping[src_id] = tgt_ids
        return mapping
    except:
        raise RuntimeError('Invalid block mapping string.')


class Branch(base.Branch):
    def branch_function(
        self,
        target_model_name: str = None,
        block_mapping: str = None,
        start_at_step_zero: bool = False,
        data_seed: int = 118
    ):
        # Process the mapping
        # A valid string format of a mapping is like:
        #   `0:0;1:1,2;2:3,4;3:5,6;4:7,8`
        if 'cifar' in target_model_name and 'resnet' in target_model_name:
            mappings = parse_block_mapping_for_stage(block_mapping)
        elif 'imagenet' in target_model_name and 'resnet' in target_model_name:
            mappings = list(map(parse_block_mapping_for_stage, block_mapping.split('|')))
        elif 'cifar' in target_model_name and 'vggnfc' in target_model_name:
            mappings = parse_block_mapping_for_stage(block_mapping)
        elif 'cifar' in target_model_name and 'vgg' in target_model_name:
            mappings = list(map(parse_block_mapping_for_stage, block_mapping.split('|')))
        elif 'cifar' in target_model_name and 'mobilenetv1' in target_model_name:
            mappings = parse_block_mapping_for_stage(block_mapping)
        elif 'mnist' in target_model_name and 'lenet' in target_model_name:
            mappings = parse_block_mapping_for_stage(block_mapping)
        else:
            raise NotImplementedError('Other mapping cases not implemented yet')

        # Load source model at `train_start_step`
        src_mask = Mask.load(self.level_root)
        start_step = self.lottery_desc.str_to_step('0it') if start_at_step_zero else self.lottery_desc.train_start_step
        # model = PrunedModel(models.registry.get(self.lottery_desc.model_hparams), src_mask)
        src_model = models.registry.load(self.level_root, start_step, self.lottery_desc.model_hparams)

        # Create target model
        target_model_hparams = copy.deepcopy(self.lottery_desc.model_hparams)
        target_model_hparams.model_name = target_model_name
        target_model = models.registry.get(target_model_hparams)
        target_ones_mask = Mask.ones_like(target_model)

        # Do the morphism
        target_sd = change_depth(target_model_name, src_model.state_dict(), target_model.state_dict(), mappings)
        target_model.load_state_dict(target_sd)
        target_mask = change_depth(target_model_name, src_mask, target_ones_mask, mappings)
        target_model_a = PrunedModel(target_model, target_mask)
        target_model_b = copy.deepcopy(target_model_a)

        # Save and run a standard train on model a
        seed_a = data_seed + 9999
        training_hparams_a = copy.deepcopy(self.lottery_desc.training_hparams)
        training_hparams_a.data_order_seed = seed_a
        output_dir_a = os.path.join(self.branch_root, f'seed_{seed_a}')
        target_mask.save(output_dir_a)
        train.standard_train(target_model_a, output_dir_a, self.lottery_desc.dataset_hparams,
                             training_hparams_a, start_step=start_step, verbose=self.verbose)

        # Save and run a standard train on model b
        seed_b = data_seed + 10001
        training_hparams_b = copy.deepcopy(self.lottery_desc.training_hparams)
        training_hparams_b.data_order_seed = seed_b
        output_dir_b = os.path.join(self.branch_root, f'seed_{seed_b}')
        target_mask.save(output_dir_b)
        train.standard_train(target_model_b, output_dir_b, self.lottery_desc.dataset_hparams,
                             training_hparams_b, start_step=start_step, verbose=self.verbose)

        # Linear connectivity between model_a and model_b
        training_hparams_c = copy.deepcopy(self.lottery_desc.training_hparams)
        training_hparams_c.training_steps = '1ep'
        for alpha in np.linspace(0, 1.0, 21):
            model_c = linear_interpolate(target_model_a, target_model_b, alpha)
            output_dir_c = os.path.join(self.branch_root, f'alpha_{alpha}')
            # Measure acc of model_c
            train.standard_train(model_c, output_dir_c, self.lottery_desc.dataset_hparams,
                                 training_hparams_c, start_step=None, verbose=self.verbose)

    @staticmethod
    def description():
        return "Change the depth of the source network and do linear connectivity exp."

    @staticmethod
    def name():
        return 'change_depth_linear_connect'

