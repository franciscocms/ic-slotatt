import torch
import numpy as np

import os
import sys
sys.path.append(os.path.abspath(__file__+'/../../../'))

import logging
logger = logging.getLogger("eval")


def process_preds(preds):
    # preds must have shape (max_objects, n_features)
    assert len(preds.shape) == 2

    coords = preds[:, :3]
    object_size = torch.argmax(preds[:, 3:5], dim=-1)
    material = torch.argmax(preds[:, 5:7], dim=-1)
    shape = torch.argmax(preds[:, 7:10], dim=-1)
    color = torch.argmax(preds[:, 10:18], dim=-1)
    real_obj = preds[:, 18] if all([p in [0., 1.] for p in preds[:, 18]]) else torch.distributions.Bernoulli(preds[:, 18]).sample()
    return coords, object_size, material, shape, color, real_obj


def compute_AP(preds, targets, threshold_dist, print_ap=False):

    """
    adapted from 'https://github.com/google-research/google-research/blob/master/slot_attention/utils.py'
    """

    sizes = ['small', 'large']
    materials = ['rubber', 'metal']
    shapes = ['cube', 'sphere', 'cylinder']
    colors = ['gray', 'blue', 'brown', 'yellow', 'red', 'green', 'purple', 'cyan']

    # preds have shape (max_objects, n_features)
    # targets have shape (max_objects, n_features)

    # if threshold_dist == -1:
    #     logger.info(f"\ncomputing AP...")
    #     logger.info(f"preds: {preds}")
    #     logger.info(f"targets: {targets}")

    assert preds.shape == targets.shape
    assert preds.shape[0] != 1
    
    # if threshold_dist == -1.:
    #     logger.info(f"\npred coords and real flag: {torch.cat((preds[:, :3], preds[:, -1].unsqueeze(-1)), dim=-1)}")  
    #     logger.info(f"\ntarget coords and real flag: {torch.cat((targets[:, :3], targets[:, -1].unsqueeze(-1)), dim=-1)}")

    #logger.info(f"\npredictions matrix: ")
    coords, size, mat, shape, color, pred_real_obj = process_preds(preds)
    #logger.info(f"\ntarget matrix: ")
    target_coords, target_size, target_mat, target_shape, target_color, target_real_obj = process_preds(targets)

    # shape, size, ...  has shape (17)

    max_objects = coords.shape[0]
    
    tp = np.zeros(1)
    fp = np.zeros(1)
    
    found_objects = []
    for o in range(max_objects):
        if pred_real_obj[o]:
            
            #logger.info(shape[o])
            #logger.info(target_shape[0])

            #logger.info(f'{locx[o]} - {locy[o]}')
            #logger.info(f'{target_locx[o]} - {target_locy[o]}')

            # tries to find a match between predicted object 'o' and any target object
            # returns the best distance match (if more than one occurs)

            found = False
            found_idx = -1 
            best_distance = 1000
            
            for j in range(max_objects):
                if target_real_obj[j]:
                    if [shape[o], size[o], color[o], mat[o]] == [target_shape[j], target_size[j], target_color[j], target_mat[j]]: 
                        dist = np.linalg.norm((target_coords[j].numpy() - coords[o].numpy()) * 3.)
                        if dist < best_distance and j not in found_objects:
                            #logger.info(f'found at best distance {dist}')
                            found = True
                            best_distance = dist
                            found_idx = j # stores the best match between an object and all possible targets

                            # if threshold_dist == -1:
                            # logger.info(f"object {j} found to have the best distance {best_distance} matching with object {o}")
                            # logger.info(f"object {o} coords: {coords[o]} - object {j} coords: {target_coords[j]}\n")
            
            if found:
                if dist <= threshold_dist or threshold_dist == -1:
                    found_objects.append(found_idx)
                    
                    #logger.info(f"found match between pred object {o} and real object {found_idx} below distance threshold!")

                    #logger.info(f"PREDS: {[shapes[shape[o]], sizes[size[o]], colors[color[o]], materials[mat[o]]]}")
                    #logger.info(f"TARGET: {[shapes[target_shape[found_idx]], sizes[target_size[found_idx]], colors[target_color[found_idx]], materials[target_mat[found_idx]]]}")
                    
                    tp += 1
            else: fp += 1

            #logger.info(found_objects)
    
    precision = tp / (tp+fp)
    recall = tp / np.sum(np.asarray(target_real_obj.cpu()))

    #logger.info(f'precision: {precision}')
    #logger.info(f'recall: {recall}')

    recall = recall.tolist()
    precision = precision.tolist()
    recall = [0] + recall + [1]
    precision = [0] + precision + [0]

    for i in range(len(precision) - 1, -0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    indices_recall = [
        i for i in range(len(recall) - 1) if recall[1:][i] != recall[:-1][i]
    ]

    average_precision = 0.
    for i in indices_recall:
        average_precision += precision[i + 1] * (recall[i + 1] - recall[i])

    if print_ap:
        logger.info(f'ap: {average_precision}')
    return average_precision