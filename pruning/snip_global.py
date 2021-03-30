# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import numpy as np

from foundations import hparams
import models.base
from training import train
from pruning import base
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel
from platforms.platform import get_platform


@dataclasses.dataclass
class PruningHparams(hparams.PruningHparams):
    pruning_fraction: float = 0.2
    pruning_layers_to_ignore: str = None

    _name = 'Hyperparameters for Sparse Global Pruning'
    _description = 'Hyperparameters that modify the way pruning occurs.'
    _pruning_fraction = 'The fraction of additional weights to prune from the network.'
    _layers_to_ignore = 'A comma-separated list of addititonal tensors that should not be pruned.'


class Strategy(base.Strategy):
    @staticmethod
    def get_pruning_hparams() -> type:
        return PruningHparams

    @staticmethod
    def prune(pruning_hparams: PruningHparams,
              trained_model: models.base.Model,
              current_mask: Mask = None,
              training_hparams: hparams.TrainingHparams = None,
              dataset_hparams: hparams.DatasetHparams = None,
              data_order_seed: int = None):
        current_mask = Mask.ones_like(trained_model) if current_mask is None else current_mask
        current_mask_numpy = current_mask.numpy()

        # Determine the number of weights that need to be pruned.
        number_of_remaining_weights = np.sum([np.sum(v) for v in current_mask_numpy.values()])
        number_of_weights_to_prune = np.ceil(
            pruning_hparams.pruning_fraction * number_of_remaining_weights).astype(int)

        # Determine which layers can be pruned.
        prunable_tensors = set(trained_model.prunable_layer_names)
        if pruning_hparams.pruning_layers_to_ignore:
            prunable_tensors -= set(pruning_hparams.pruning_layers_to_ignore.split(','))

        # Get the model score.
        scores = Strategy.get_score(trained_model, current_mask, prunable_tensors,
                                    training_hparams, dataset_hparams, data_order_seed)

        # Get the model weights.
        # weights = {k: v.clone().cpu().detach().numpy()
        #            for k, v in trained_model.state_dict().items()
        #            if k in prunable_tensors}

        # Create a vector of all the unpruned weights in the model.
        # weight_vector = np.concatenate([v[current_mask[k] == 1] for k, v in weights.items()])
        score_vector = np.concatenate([v[current_mask_numpy[k] == 1] for k, v in scores.items()])
        threshold = np.sort(np.abs(score_vector))[number_of_weights_to_prune]

        new_mask = Mask({k: np.where(np.abs(v) > threshold, current_mask_numpy[k], np.zeros_like(v))
                         for k, v in scores.items()})
        for k in current_mask_numpy:
            if k not in new_mask:
                new_mask[k] = current_mask_numpy[k]

        return new_mask

    @staticmethod
    def get_score(trained_model: models.base.Model,
                  current_mask: Mask,
                  prunable_tensors: set,
                  training_hparams: hparams.TrainingHparams,
                  dataset_hparams: hparams.DatasetHparams,
                  data_order_seed: int = None):
        pruned_model = PrunedModel(trained_model, current_mask).to(device=get_platform().torch_device)
        pruned_model._clear_grad()
        # pruned_model._enable_mask_gradient()

        # Calculate the gradient
        train.accumulate_gradient(
            training_hparams, pruned_model,
            dataset_hparams, data_order_seed, verbose=False
        )

        # Calculate the score
        scores = dict()
        for name, param in pruned_model.model.named_parameters():
            if hasattr(pruned_model, PrunedModel.to_mask_name(name)) and name in prunable_tensors:
                scores[name] = (param.grad * param).abs().clone().cpu().detach().numpy()

        score_vector = np.concatenate([v.reshape(-1) for k, v in scores.items()])
        norm = np.sum(score_vector)
        for k in scores.keys():
            scores[k] /= norm

        # Clean up
        pruned_model._clear_grad()
        # model._disable_mask_gradient()

        return scores

