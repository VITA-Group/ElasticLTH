# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import numpy as np
from utils.tensor_utils import shuffle_tensor


def change_depth(model_name, src_sd, dst_sd, mappings, seed=None, permute_copy=False):
    if 'resnet' in model_name and 'cifar' in model_name:
        return change_depth_cifar_resnet(src_sd, dst_sd, mappings, seed=seed, permute_copy=permute_copy)
    elif 'resnet' in model_name and 'imagenet' in model_name:
        return change_depth_imagenet_resnet(src_sd, dst_sd, mappings)
    elif 'vggnfc' in model_name and 'cifar' in model_name:
        return change_depth_cifar_vggnfc(src_sd, dst_sd, mappings)
    elif 'vgg' in model_name and 'cifar' in model_name:
        return change_depth_cifar_vgg(src_sd, dst_sd, mappings)
    elif 'mobilenetv1' in model_name and 'cifar' in model_name:
        return change_depth_cifar_mobilenetv1(src_sd, dst_sd, mappings)
    elif 'lenet' in model_name and 'mnist' in model_name:
        return change_depth_mnist_lenet(src_sd, dst_sd, mappings)
    else:
        raise NotImplementedError(f'Depth morphism method is not implemeted yet for {model_name}')


def change_depth_mnist_lenet(src_sd, dst_sd, mappings):
    dst_sd = copy.deepcopy(dst_sd)

    overwritten_keys = []

    for k,v in src_sd.items():
        if 'layers' not in k:  # final linear layer
            dst_sd[k] = v.clone()
            overwritten_keys.append(k)
            print('{} not classifier, skipped'.format(k))
            continue

        # Lenet FC layers except final fc
        split_k = k.split('.')
        src_fc_id = int(split_k[1])
        dst_fc_id_list = mappings.get(src_fc_id, [])
        for dst_fc_id in dst_fc_id_list:
            new_split_k = copy.deepcopy(split_k)
            new_split_k[1] = str(dst_fc_id)
            dst_k = '.'.join(new_split_k)
            dst_sd[dst_k] = v.clone()
            overwritten_keys.append(dst_k)
            print('{} -> {}'.format(k, dst_k))

    all_overwritten = True
    for k in dst_sd.keys():
        all_overwritten = all_overwritten and k in overwritten_keys
    assert overwritten_keys

    return dst_sd


def change_depth_cifar_vggnfc(src_sd, dst_sd, mappings):
    dst_sd = copy.deepcopy(dst_sd)

    overwritten_keys = []

    for k,v in src_sd.items():
        if 'classifier' not in k:  # final linear layer
            dst_sd[k] = v.clone()
            overwritten_keys.append(k)
            print('{} not classifier, skipped'.format(k))
            continue

        # VGG FC Layers in classifier
        split_k = k.split('.')
        src_fc_id = int(split_k[1])
        dst_fc_id_list = mappings.get(src_fc_id, [])
        for dst_fc_id in dst_fc_id_list:
            new_split_k = copy.deepcopy(split_k)
            new_split_k[1] = str(dst_fc_id)
            dst_k = '.'.join(new_split_k)
            dst_sd[dst_k] = v.clone()
            overwritten_keys.append(dst_k)
            print('{} -> {}'.format(k, dst_k))

    all_overwritten = True
    for k in dst_sd.keys():
        all_overwritten = all_overwritten and k in overwritten_keys
    assert overwritten_keys

    return dst_sd


def change_depth_cifar_vgg(src_sd, dst_sd, mappings):
    dst_sd = copy.deepcopy(dst_sd)

    src_layer_ids = set()
    for k in src_sd.keys():
        if 'layers' in k:
            src_layer_ids.add(int(k.split('.')[1]))
    src_max_layer_id = max(src_layer_ids)

    dst_layer_ids = set()
    for k in dst_sd.keys():
        if 'layers' in k:
            dst_layer_ids.add(int(k.split('.')[1]))
    dst_max_layer_id = max(dst_layer_ids)

    src_stage_mapping = {i:[] for i in range(5)}
    cur_stage_id = 0
    for i in range(src_max_layer_id + 1):
        if f'layers.{i}.conv.weight' in src_sd:
            src_stage_mapping[cur_stage_id].append(i)
        else:
            cur_stage_id += 1
    print('src_stage_mapping:', src_stage_mapping)

    dst_stage_mapping = {i:[] for i in range(5)}
    cur_stage_id = 0
    for i in range(dst_max_layer_id + 1):
        if f'layers.{i}.conv.weight' in dst_sd:
            dst_stage_mapping[cur_stage_id].append(i)
        else:
            cur_stage_id += 1
    print('dst_stage_mapping:', dst_stage_mapping)

    def get_stage_local_id(stage_mapping, layer_id):
        for stage_id, stage_layer_ids in stage_mapping.items():
            if layer_id in stage_layer_ids:
                return stage_id, layer_id - stage_layer_ids[0]
    def get_layer_id(stage_mapping, stage_id, local_id):
        return stage_mapping[stage_id][0] + local_id

    overwritten_keys = []

    for k,v in src_sd.items():
        if 'layers' not in k:  # final linear layer
            dst_sd[k] = v.clone()
            overwritten_keys.append(k)
            continue

        # VGG Conv Layers
        split_k = k.split('.')
        src_layer_id = int(split_k[1])
        stage_id, src_local_id = get_stage_local_id(src_stage_mapping, src_layer_id)
        dst_local_id_list = mappings[stage_id].get(src_local_id, [])
        for dst_local_id in dst_local_id_list:
            dst_layer_id = get_layer_id(dst_stage_mapping, stage_id, dst_local_id)
            new_split_k = copy.deepcopy(split_k)
            new_split_k[1] = str(dst_layer_id)
            dst_k = '.'.join(new_split_k)
            dst_sd[dst_k] = v.clone()
            overwritten_keys.append(dst_k)

    all_overwritten = True
    for k in dst_sd.keys():
        all_overwritten = all_overwritten and k in overwritten_keys
    assert overwritten_keys

    return dst_sd


def change_depth_cifar_mobilenetv1(src_sd, dst_sd, mappings):
    dst_sd = copy.deepcopy(dst_sd)

    overwritten_keys = []

    for k,v in src_sd.items():
        if 'layers_part2' not in k:  # not the intermediate 512 layers
            dst_sd[k] = v.clone()
            overwritten_keys.append(k)
            print('{} not part 2 layers, skipped'.format(k))
            continue

        # VGG FC Layers in classifier
        split_k = k.split('.')
        src_fc_id = int(split_k[1])
        dst_fc_id_list = mappings.get(src_fc_id, [])
        for dst_fc_id in dst_fc_id_list:
            new_split_k = copy.deepcopy(split_k)
            new_split_k[1] = str(dst_fc_id)
            dst_k = '.'.join(new_split_k)
            dst_sd[dst_k] = v.clone()
            overwritten_keys.append(dst_k)
            print('{} -> {}'.format(k, dst_k))

    all_overwritten = True
    for k in dst_sd.keys():
        all_overwritten = all_overwritten and k in overwritten_keys
    assert overwritten_keys

    return dst_sd


def change_depth_cifar_resnet(src_sd, dst_sd, mappings, seed=None, permute_copy=False):
    dst_sd = copy.deepcopy(dst_sd)

    # get the milestone for stages
    src_milestones = set([0])
    dst_milestones = set([0])
    for k in src_sd.keys():
        if 'blocks' in k and 'shortcut' in k:
            src_milestones.add(int(k.split('.')[1]))
    for k in dst_sd.keys():
        if 'blocks' in k and 'shortcut' in k:
            dst_milestones.add(int(k.split('.')[1]))
    src_milestones = sorted(src_milestones)
    dst_milestones = sorted(dst_milestones)
    src_stage_len = src_milestones[1]
    dst_stage_len = dst_milestones[1]
    print(src_stage_len, dst_stage_len)
    shallow_to_deep = src_stage_len <= dst_stage_len

    overwritten_keys = []

    permute_counter = 0

    for k,v in src_sd.items():
        if 'blocks' in k:
            splitted_key = k.split('.')
            src_block_id = int(splitted_key[1])
            for i in range(len(src_milestones)-1, -1, -1):
                if src_block_id >= src_milestones[i]:
                    stage_id = i
                    break
            src_local_id = src_block_id - src_milestones[stage_id]

            dst_local_id_list = mappings.get(src_local_id, [])

            for j, dst_local_id in enumerate(dst_local_id_list):
                dst_block_id = stage_id * dst_stage_len + dst_local_id
                dst_key = copy.deepcopy(splitted_key)
                dst_key[1] = str(dst_block_id)
                dst_key = '.'.join(dst_key)
                assert dst_key in dst_sd
                if j > 0 and permute_copy:
                    dst_sd[dst_key] = shuffle_tensor(v.clone(), seed=seed + permute_counter)
                    print(f'permuted with seed {seed+permute_counter}!')
                    permute_counter += 1
                else:
                    dst_sd[dst_key] = v.clone()
                overwritten_keys.append(dst_key)
        else:
            # directly copy the first conv and bn layer and the last linear layer
            assert k in dst_sd
            dst_sd[k] = v
            overwritten_keys.append(k)

    all_overwritten = True
    for k in dst_sd.keys():
        all_overwritten = all_overwritten and k in overwritten_keys
    assert all_overwritten

    return dst_sd


def change_depth_imagenet_resnet(src_sd, dst_sd, mappings):
    dst_sd = copy.deepcopy(dst_sd)

    overwritten_keys = []

    for k,v in src_sd.items():
        if 'layer' in k:
            splitted_key = k.split('.')
            src_layer_id = int(splitted_key[1][-1])  # in range [1-4]
            src_block_id = int(splitted_key[2])

            dst_block_id_list = mappings[src_layer_id-1].get(src_block_id, [])

            for dst_block_id in dst_block_id_list:
                dst_key = copy.deepcopy(splitted_key)
                dst_key[2] = str(dst_block_id)
                dst_key = '.'.join(dst_key)
                assert dst_key in dst_sd
                dst_sd[dst_key] = v.clone()
                overwritten_keys.append(dst_key)
        else:
            # directly copy the first conv and bn layer and the last linear layer
            assert k in dst_sd
            dst_sd[k] = v
            overwritten_keys.append(k)

    all_overwritten = True
    for k in dst_sd.keys():
        all_overwritten = all_overwritten and k in overwritten_keys
    assert all_overwritten

    return dst_sd

