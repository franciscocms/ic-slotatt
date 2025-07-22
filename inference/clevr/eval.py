import pyro
import pyro.poutine as poutine
import torch
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
from scipy.special import softmax

import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import json
import shutil
import time

# add project path to sys to import relative modules
import sys
sys.path.append(os.path.abspath(__file__+'/../../../'))

import logging

from main.clevr_model import clevr_gen_model, preprocess_clevr
from main.modifiedCSIS import CSIS
from main.modifiedImportance import vectorized_importance_weights
from guide import InvSlotAttentionGuide, visualize
from utils.distributions import Empirical
from eval_utils import compute_AP, transform_coords
from utils.guide import load_trained_guide_clevr
from main.setup import params, JOB_SPLIT
from dataset import CLEVRDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_default_device(device)
main_dir = os.path.abspath(__file__+'/../../../')

properties_json_path = os.path.join(main_dir, "main", "clevr_data", "properties.json")
dataset_path = "/nas-ctm01/datasets/public/CLEVR/CLEVR_v1.0"

# Load the property file
with open(properties_json_path, 'r') as f:
    properties = json.load(f)
    color_name_to_rgba = {}
    for name, rgb in properties['colors'].items():
        rgba = [float(c) / 255.0 for c in rgb] + [1.0]
        color_name_to_rgba[name] = rgba
        material_mapping = [(v, k) for k, v in properties['materials'].items()]
    object_mapping = [(v, k) for k, v in properties['shapes'].items()]
    size_mapping = list(properties['sizes'].items())
    color_mapping = list(color_name_to_rgba.items())


PRINT_INFERENCE_TIME = False

def average_precision_clevr(pred, attributes, distance_threshold):
  """Computes the average precision for CLEVR.
  This function computes the average precision of the predictions specifically
  for the CLEVR dataset. First, we sort the predictions of the model by
  confidence (highest confidence first). Then, for each prediction we check
  whether there was a corresponding object in the input image. A prediction is
  considered a true positive if the discrete features are predicted correctly
  and the predicted position is within a certain distance from the ground truth
  object.
  Args:
    pred: Tensor of shape [batch_size, num_elements, dimension] containing
      predictions. The last dimension is expected to be the confidence of the
      prediction.
    attributes: Tensor of shape [batch_size, num_elements, dimension] containing
      ground-truth object properties.
    distance_threshold: Threshold to accept match. -1 indicates no threshold.
  Returns:
    Average precision of the predictions.
  """

  # pred[:, :, :3] = (pred[:, :, :3] + 1) / 2
  # attributes[:, :, :3] = (attributes[:, :, :3] + 1) / 2

  [batch_size, _, element_size] = attributes.shape
  [_, predicted_elements, _] = pred.shape

  def unsorted_id_to_image(detection_id, predicted_elements):
    """Find the index of the image from the unsorted detection index."""
    return int(detection_id // predicted_elements)

  flat_size = batch_size * predicted_elements
  flat_pred = np.reshape(pred, [flat_size, element_size])
  sort_idx = np.argsort(flat_pred[:, -1], axis=0)[::-1]  # Reverse order.

  sorted_predictions = np.take_along_axis(
      flat_pred, np.expand_dims(sort_idx, axis=1), axis=0)
  idx_sorted_to_unsorted = np.take_along_axis(
      np.arange(flat_size), sort_idx, axis=0)

  def process_targets(target):
    """Unpacks the target into the CLEVR properties."""
    coords = target[:3]
    object_size = np.argmax(target[3:5])
    material = np.argmax(target[5:7])
    shape = np.argmax(target[7:10])
    color = np.argmax(target[10:18])
    real_obj = target[18]
    return coords, object_size, material, shape, color, real_obj

  true_positives = np.zeros(sorted_predictions.shape[0])
  false_positives = np.zeros(sorted_predictions.shape[0])

  detection_set = set()

  for detection_id in range(sorted_predictions.shape[0]):
    # Extract the current prediction.
    current_pred = sorted_predictions[detection_id, :]
    # Find which image the prediction belongs to. Get the unsorted index from
    # the sorted one and then apply to unsorted_id_to_image function that undoes
    # the reshape.
    original_image_idx = unsorted_id_to_image(
        idx_sorted_to_unsorted[detection_id], predicted_elements)
    # Get the ground truth image.
    gt_image = attributes[original_image_idx, :, :]

    # Initialize the maximum distance and the id of the groud-truth object that
    # was found.
    best_distance = 10000
    best_id = None

    # Unpack the prediction by taking the argmax on the discrete attributes.
    (pred_coords, pred_object_size, pred_material, pred_shape, pred_color,
     _) = process_targets(current_pred)

    # Loop through all objects in the ground-truth image to check for hits.
    for target_object_id in range(gt_image.shape[0]):
      target_object = gt_image[target_object_id, :]
      # Unpack the targets taking the argmax on the discrete attributes.
      (target_coords, target_object_size, target_material, target_shape,
       target_color, target_real_obj) = process_targets(target_object)
      # Only consider real objects as matches.
      if target_real_obj:
        # For the match to be valid all attributes need to be correctly
        # predicted.
        pred_attr = [pred_object_size, pred_material, pred_shape, pred_color]
        target_attr = [
            target_object_size, target_material, target_shape, target_color]
        match = pred_attr == target_attr
        if match:
          # If a match was found, we check if the distance is below the
          # specified threshold. Recall that we have rescaled the coordinates
          # in the dataset from [-3, 3] to [0, 1], both for `target_coords` and
          # `pred_coords`. To compare in the original scale, we thus need to
          # multiply the distance values by 6 before applying the norm.
          distance = np.linalg.norm((target_coords - pred_coords) * 3.)

          # If this is the best match we've found so far we remember it.
          if distance < best_distance:
            best_distance = distance
            best_id = target_object_id
    if best_distance < distance_threshold or distance_threshold == -1:
      # We have detected an object correctly within the distance confidence.
      # If this object was not detected before it's a true positive.
      if best_id is not None:
        if (original_image_idx, best_id) not in detection_set:
          true_positives[detection_id] = 1
          detection_set.add((original_image_idx, best_id))
        else:
          false_positives[detection_id] = 1
      else:
        false_positives[detection_id] = 1
    else:
      false_positives[detection_id] = 1
  accumulated_fp = np.cumsum(false_positives)
  accumulated_tp = np.cumsum(true_positives)
  recall_array = accumulated_tp / np.sum(attributes[:, :, -1])
  precision_array = np.divide(accumulated_tp, (accumulated_fp + accumulated_tp))

  return compute_average_precision(
        np.array(precision_array, dtype=np.float32),
        np.array(recall_array, dtype=np.float32))


def compute_average_precision(precision, recall):
    """Computation of the average precision from precision and recall arrays."""
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
    return average_precision


def process_preds(trace, id):
    
    """ returns a matrix of predictions from the proposed trace """

    max_obj = max(params['max_objects'], params['num_slots'])
    
    features_dim = 18
    preds = torch.zeros(max_obj, features_dim)
    for name, site in trace.nodes.items():
        if site['type'] == 'sample':
            if name == 'shape': preds[:, :3] = F.one_hot(site['value'][id], len(object_mapping))
            if name == 'color': preds[:, 3:11] = F.one_hot(site['value'][id], len(color_mapping))
            if name == 'size': preds[:, 11:13] = F.one_hot(site['value'][id], len(size_mapping))
            if name == 'mat': preds[:, 13:15] = F.one_hot(site['value'][id], len(material_mapping))
            #if name == 'pose': site['value']
            if name == 'x': preds[:, 15] = site['value'][id]
            if name == 'y': preds[:, 16] = site['value'][id]
            if name == 'mask': preds[:, 17] = site['value'][id]
    
    return preds

def process_preds_argmax(trace, id):
    
    """ returns a matrix of predictions from the proposed trace """

    max_obj = max(params['max_objects'], params['num_slots'])
    
    features_dim = 18
    preds = torch.zeros(max_obj, features_dim)
    for name, site in trace.nodes.items():
        if site['type'] == 'sample':
            if name == 'shape': preds[:, :3] = F.one_hot(torch.argmax(site['fn'].probs[id], dim=-1), len(object_mapping))
            if name == 'color': preds[:, 3:11] = F.one_hot(torch.argmax(site['fn'].probs[id], dim=-1), len(color_mapping))
            if name == 'size': preds[:, 11:13] = F.one_hot(torch.argmax(site['fn'].probs[id], dim=-1), len(size_mapping))
            if name == 'mat': preds[:, 13:15] = F.one_hot(torch.argmax(site['fn'].probs[id], dim=-1), len(material_mapping))
            #if name == 'pose': site['value']
            if name == 'x': preds[:, 15] = site['value'][id]
            if name == 'y': preds[:, 16] = site['value'][id]
            if name == 'mask': preds[:, 17] = site['value'][id]
    
    return preds

def process_targets(target_dict):   
    features_dim = 18
    max_obj = max(params['max_objects'], params['num_slots'])
    target = torch.zeros(max_obj, features_dim)

    for o, object in enumerate(target_dict['objects']):               
        target[o, :3] = F.one_hot(torch.tensor([idx for idx, tup in enumerate(object_mapping) if tup[1] == object['shape'][0]]), len(object_mapping))
        target[o, 3:11] = F.one_hot(torch.tensor([idx for idx, tup in enumerate(color_mapping) if tup[0] == object['color'][0]]), len(color_mapping))
        target[o, 11:13] = F.one_hot(torch.tensor([idx for idx, tup in enumerate(size_mapping) if tup[0] == object['size'][0]]), len(size_mapping))
        target[o, 13:15] = F.one_hot(torch.tensor([idx for idx, tup in enumerate(material_mapping) if tup[1] == object['material'][0]]), len(material_mapping))
        #target[o, 15] = torch.tensor(object['rotation']/360.)
        target[o, 15:17] = torch.tensor(object['3d_coords'][:2])/3.
        target[o, 17] = torch.tensor(1.)
    
    return target
    
def main(): 

    init_time = time.time()  

    assert params['batch_size'] == 1
    
    logfile_name = f"eval_split_{JOB_SPLIT['id']}.log"
    logger = logging.getLogger("eval")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile_name, mode='w')
    logger.addHandler(fh)

    # logger.info(object_mapping)
    # logger.info(size_mapping)
    # logger.info(color_mapping)
    # logger.info(material_mapping) 

    logger.info(device)

    seeds = [1]
    
    for seed in seeds: 
        pyro.set_rng_seed(seed)
        
        model = clevr_gen_model
        guide = InvSlotAttentionGuide(resolution = params['resolution'],
                              num_iterations = 3,
                              hid_dim = params["slot_dim"],
                              stage="eval"
                              ).to(device)
        
        GUIDE_PATH = os.path.join(main_dir, "inference", f"checkpoint-{params['jobID']}", f"guide_{params['guide_step']}.pth")
        if os.path.isfile(GUIDE_PATH): guide = load_trained_guide_clevr(guide, GUIDE_PATH, 
                                                                        dict(mat_map=material_mapping,
                                                                             shape_map=object_mapping,
                                                                             size_map=size_mapping,
                                                                             color_map=color_mapping))
        else: raise ValueError(f'{GUIDE_PATH} is not a valid path!')

        logger.info(f'seed {seed}')
        logger.info(GUIDE_PATH)
        logger.info(f"\nrunning inference with {params['num_inference_samples']} particles\n")

        optimiser = pyro.optim.Adam({'lr': 1e-4})
        csis = CSIS(model, guide, optimiser, training_batch_size=256, num_inference_samples=params["num_inference_samples"])

        plots_dir = os.path.abspath("set_prediction_plots")
        if not os.path.isdir(plots_dir): os.mkdir(plots_dir)
        else: 
            shutil.rmtree(plots_dir)
            os.mkdir(plots_dir)
        
        threshold = [-1., 1., 0.5, 0.25, 0.125, 0.0625]
        ap = {k: 0 for k in threshold}

        # define dataset
        images_path = os.path.join(dataset_path, 'images/val')
        scenes_path = os.path.join(dataset_path, 'scenes/CLEVR_val_scenes.json')
        test_dataset = CLEVRDataset(images_path, scenes_path, max_objs=params["max_objects"])
        b_s = 1 if params["num_inference_samples"] > 1 else 512
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=b_s, shuffle=False, num_workers=8, generator=torch.Generator(device='cuda'))

        # total_len = len(self.target)
        # split_len = int(total_len/JOB_SPLIT['total'])
        # if total_len % JOB_SPLIT['total'] != 0 and JOB_SPLIT['id'] == JOB_SPLIT['total']:
        #     final_idx = split_len*(JOB_SPLIT['id']) + total_len % JOB_SPLIT['total']
        # else:
        #     final_idx = split_len*(JOB_SPLIT['id'])
        # self.target = self.target[split_len*(JOB_SPLIT['id']-1) : final_idx]
        
        n_test_samples = 0
        num_iters = 0

        guide.eval()
        with torch.no_grad():
            for img, target in testloader:
                img = img.to(device)        

                #resampling = Empirical(torch.stack([torch.tensor(i) for i in range(len(log_wts))]), torch.stack(log_wts))

                if params["num_inference_samples"] == 1:

                    preds = guide(observations={"image": img})
                    
                    # logger.info(f"\npreds: {preds[0]}")
                    # logger.info(f"\ntarget: {target[0]}")
                    
                    for t in threshold: 
                        ap[t] += average_precision_clevr(preds.detach().cpu().numpy(), 
                                                         target.detach().cpu().numpy(), 
                                                         t)
                    num_iters += 1                 
                
                
                elif params["num_inference_samples"] > 1:
                    
                    posterior = csis.run(observations={"image": img})
                    prop_traces = posterior.prop_traces[0]
                    traces = posterior.exec_traces[0]
                    log_wts = posterior.log_weights[0]
                    
                    
                    #resampling_id = resampling().item()
                    log_wts = np.array([l.item() for l in log_wts])
                    logger.info(f"log weights: {log_wts}")
                    norm_log_wts = softmax(log_wts)
                    logger.info(f"norm log weights: {norm_log_wts}")

                    for i, w in enumerate(norm_log_wts):
                        preds = process_preds(prop_traces, i)
                        for t in threshold: 
                            ap[t] += w * compute_AP(preds, target, t)
                        
                    n_test_samples += 1

                    logger.info(f"current stats:")
                    aux_mAP = {k: v/n_test_samples for k, v in ap.items()}
                    logger.info(aux_mAP)



                        
                        # logger.info("\n")
                        # for name, site in traces.nodes.items():                    
                        #     # if site["type"] == "sample":
                        #     #     logger.info(f"{name} - {site['value'].shape}")# - {site['value'][resampling_id]}")
                            
                        #     if name == 'image':
                        #         for i in range(site["fn"].mean.shape[0]):
                        #             output_image = site["fn"].mean[i]
                        #             plt.imshow(visualize(output_image[:3].permute(1, 2, 0).cpu().numpy()))
                        #             plt.savefig(os.path.join(plots_dir, f"trace_{img_index}_{i}.png"))
                        #             plt.close()

                        
                        
                        
                        # else:
                        #     # only to analyze the impact of resampling traces based on log-likelihood weights
                        #     # when compared with the posterior trace that maximizes AP
                        #     # do not compute performance metrics using the below code!

                        #     max_ap_idx = 0
                        #     best_overall_ap = 0.   
                        #     for i in range(len(log_wts)):
                        #         aux_ap = {k: 0 for k in threshold}
                        #         preds = process_preds(prop_traces, i)
                        #         for t in threshold: 
                        #             aux_ap[t] = compute_AP(preds, target, t)
                        #         #logger.info(f"proposal trace {i} - AP values: {list(aux_ap.values())}")
                        #         overall_ap = np.mean(list(aux_ap.values()))
                        #         if overall_ap > best_overall_ap:
                        #             best_overall_ap = overall_ap
                        #             max_ap_idx = i
                            
                        #     logger.info(max_ap_idx)
                        #     preds = process_preds(prop_traces, max_ap_idx)


                #     else:
                #         preds = process_preds_argmax(prop_traces, 0)

                    
                #     for t in threshold: 
                #         ap[t] += compute_AP(preds, target, t)
                #     n_test_samples += 1

                    
                    
                #     logger.info(f"current stats:")
                #     aux_mAP = {k: v/n_test_samples for k, v in ap.items()}
                #     logger.info(aux_mAP)

                # else:
                #     resampling_id = resampling(torch.Size((sample_size,)))
                #     logger.info(f"log weights: {[l.item() for l in log_wts]} - resampled traces: {resampling_id}")

                #     max_ap_idx = 0
                #     best_overall_ap = 0.
                #     for p in resampling_id:
                #         p = int(p)
                #         preds = process_preds(prop_traces, p)
                #         aux_ap = {k: 0 for k in threshold}
                #         for t in threshold: 
                #             aux_ap[t] = compute_AP(preds, target, t)
                #         overall_ap = np.mean(list(aux_ap.values()))
                #         if overall_ap > best_overall_ap:
                #             best_overall_ap = overall_ap
                #             max_ap_idx = p
                #     preds = process_preds(prop_traces, max_ap_idx)
                    
                #     logger.info(f"\npreds: {preds}")
                #     logger.info(f"target: {target}\n")

                    
                #     for t in threshold: 
                #         ap[t] += compute_AP(preds, target, t)

                #     n_test_samples += 1

                #     logger.info(f"current stats:")
                #     aux_mAP = {k: v/n_test_samples for k, v in ap.items()}
                #     logger.info(aux_mAP)

        mAP = {k: v/num_iters for k, v in ap.items()}
        logger.info(f"distance thresholds: \n {threshold[0]} - {threshold[1]} - {threshold[2]} - {threshold[3]} - {threshold[4]} - {threshold[5]}")
        logger.info(f"mAP values: {mAP[threshold[0]]} - {mAP[threshold[1]]} - {mAP[threshold[2]]} - {mAP[threshold[3]]} - {mAP[threshold[4]]} - {mAP[threshold[5]]}\n")

    inference_time = time.time() - init_time 
    logger.info(f'\nInference complete in {inference_time} seconds.')

if __name__ == '__main__':
    main()
    