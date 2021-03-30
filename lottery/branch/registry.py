# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from lottery.branch.base import Branch
from lottery.branch import randomly_prune
from lottery.branch import randomly_reinitialize
from lottery.branch import retrain
from lottery.branch import change_depth
from lottery.branch import change_depth_linear_connect
from lottery.branch import change_depth_dataset
from lottery.branch import change_depth_random_mask
from lottery.branch import prune_early

registered_branches = {
    'randomly_prune': randomly_prune.Branch,
    'randomly_reinitialize': randomly_reinitialize.Branch,
    'retrain': retrain.Branch,
    'change_depth': change_depth.Branch,
    'change_depth_linear_connect': change_depth_linear_connect.Branch,
    'change_depth_dataset': change_depth_dataset.Branch,
    'change_depth_random_mask': change_depth_random_mask.Branch,
    'prune_early': prune_early.Branch,
}


def get(branch_name: str) -> Branch:
    if branch_name not in registered_branches:
        raise ValueError('No such branch: {}'.format(branch_name))
    else:
        return registered_branches[branch_name]
