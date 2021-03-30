# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from lottery.branch import base
import models.registry
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel
from training import train
from lottery.branch.morphism import change_depth

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
        start_at_step_zero: bool = False
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
        target_model = PrunedModel(target_model, target_mask)

        # Save and run a standard train
        target_mask.save(self.branch_root)
        train.standard_train(target_model, self.branch_root, self.lottery_desc.dataset_hparams,
                             self.lottery_desc.training_hparams, start_step=start_step, verbose=self.verbose)

    @staticmethod
    def description():
        return "Change the depth of the source network."

    @staticmethod
    def name():
        return 'change_depth'

