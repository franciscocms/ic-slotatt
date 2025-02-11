import torch
import numpy as np

import os
import sys
sys.path.append(os.path.abspath(__file__+'/../../../'))

import logging
logger = logging.getLogger("eval")

def transform_coords(coords):
    #return (coords/2) + 0.5
    """
    re-scale coordinates (both predicted and target) from [-1., 1.] to [-3., 3.]
    """
    return coords*3.

def process_preds(preds):
    # preds must have shape (max_objects, n_features)
    assert len(preds.shape) == 2

    shape = torch.argmax(preds[:, :3], dim=-1)
    color = torch.argmax(preds[:, 3:11], dim=-1)
    size = torch.argmax(preds[:, 11:13], dim=-1)
    mat = torch.argmax(preds[:, 13:15], dim=-1)
    x, y = transform_coords(preds[:, 15]), transform_coords(preds[:, 16])
    real_obj = preds[:, 17]
    return shape, size, color, mat, x, y, real_obj

def distance(loc1, loc2):
    return torch.sqrt(torch.square(loc1[0]-loc2[0]) + torch.square(loc1[1]-loc2[1]))

def compute_AP(preds, targets, threshold_dist, print_ap=False):

    """
    adapted from 'https://github.com/google-research/google-research/blob/master/slot_attention/utils.py'
    """

    # preds have shape (max_objects, n_features)
    # targets have shape (max_objects, n_features)

    # if threshold_dist == -1:
    #     logger.info(f"\ncomputing AP...")
    #     logger.info(f"preds: {preds}")
    #     logger.info(f"targets: {targets}")

    #logger.info(f"\npredictions matrix: ")
    shape, size, color, mat, x, y, pred_real_obj = process_preds(preds)
    #logger.info(f"\ntarget matrix: ")
    target_shape, target_size, target_color, target_mat, target_x, target_y, target_real_obj = process_preds(targets)

    # shape, size, ...  has shape (17)

    max_objects = shape.shape[0]
    
    tp = np.zeros(1)
    fp = np.zeros(1)
    
    found_objects = []
    for o in range(max_objects):
        if torch.round(pred_real_obj[o]):

            #logger.info(f'{o} - {pred_real_obj[o]}')
            
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
                        dist = distance((x[o], y[o]), (target_x[j], target_y[j]))
                        if dist < best_distance and j not in found_objects:
                            #logger.info(f'found at best distance {dist}')
                            found = True
                            best_distance = dist
                            found_idx = j # stores the best match between an object and all possible targets

                            # if threshold_dist == -1:
                            #     logger.info(f"object {j} found to have the best distance {best_distance} matching with object {o}")
            
            if found:
                if distance((x[o], y[o]), (target_x[found_idx], target_y[found_idx])) <= threshold_dist or threshold_dist == -1:
                    found_objects.append(found_idx)
                    # if threshold_dist == -1: logger.info("found match below distance threshold!")
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