import pyro
import pyro.poutine as poutine
import torch
from torchvision import transforms
import numpy as np
import torch.nn.functional as F

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

def process_preds(trace, id):
    
    """ returns a matrix of predictions from the proposed trace """
    
    features_dim = 18
    preds = torch.zeros(params['max_objects'], features_dim)
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

def process_targets(target_dict):   
    features_dim = 18
    target = torch.zeros(params['max_objects'], features_dim)

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
    
    logfile_name = f"eval_occlusion.log"
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
        logger.info(GUIDE_PATH)

        optimiser = pyro.optim.Adam({'lr': 1e-4})
        csis = CSIS(model, guide, optimiser, training_batch_size=256, num_inference_samples=params["num_inference_samples"])

        # define dataset
        img_transform = transforms.Compose([transforms.ToTensor()])
        
        img = torch.from_numpy(np.asarray(Image.open('occlusion_imgs/unoccluded_2.png'))).permute(2, 0, 1).unsqueeze(0)
        img = preprocess_clevr(img)
        logger.info(img.shape)

        occlusion_plots_dir = os.path.abspath("occlusion_plots")
        if not os.path.isdir(occlusion_plots_dir): os.mkdir(occlusion_plots_dir)
        else: 
            shutil.rmtree(occlusion_plots_dir)
            os.mkdir(occlusion_plots_dir)

        # plt.imshow(visualize(img.squeeze(dim=0)[:3].permute(1, 2, 0).cpu().numpy()))
        # plt.savefig(os.path.join(occlusion_plots_dir, f"occluded_image.png"))
        # plt.close() 

        guide.eval()
        with torch.no_grad():
            img = img.to(device)

            posterior = csis.run(observations={"image": img})
            prop_traces = posterior.prop_traces[0]
            traces = posterior.exec_traces[0]
            log_wts = posterior.log_weights[0]

            resampling = Empirical(torch.stack([torch.tensor(i) for i in range(len(log_wts))]), torch.stack(log_wts))
            resampling_id = resampling().item()

            logger.info(f"trace {resampling_id} resampled with logwt: {log_wts[resampling_id]}")

            for name, site in traces.nodes.items():                    
                    # if site["type"] == "sample":
                    #     logger.info(f"{name} - {site['value'].shape}")# - {site['value'][resampling_id]}")
                    
                    if name == 'image':
                        for i in range(site["fn"].mean.shape[0]):
                            output_image = site["fn"].mean[i]
                            plt.imshow(visualize(output_image[:3].permute(1, 2, 0).cpu().numpy()))
                            plt.savefig(os.path.join(occlusion_plots_dir, f"trace_{i}.png"))
                            plt.close()

            shape_pred_dict = {s: 0 for s in range(3)}
            size_pred_dict = {s: 0 for s in range(2)}
            color_pred_dict = {s: 0 for s in range(8)}
            mat_pred_dict = {s: 0 for s in range(2)}

            
            analysed_traces = 0

            cylinder_log_wts = {}
            cube_log_wts = {}

            for i in range(len(log_wts)):
                preds = process_preds(prop_traces, i)
                
                shape = torch.argmax(preds[:, :3], dim=-1)
                color = torch.argmax(preds[:, 3:11], dim=-1)
                size = torch.argmax(preds[:, 11:13], dim=-1)
                mat = torch.argmax(preds[:, 13:15], dim=-1)
                real_obj = torch.round(preds[:, 17])

                if torch.sum(real_obj) == 2.:
                    
                    analysed_traces += 1
                    
                    occluder_shape = torch.tensor(0.)
                    occluder_size = torch.tensor(0.)
                    occluder_color = torch.tensor(1.)
                    occluder_mat = torch.tensor(1.)

                    for o in range(shape.shape[0]):

                        if real_obj[o]:
                            # exclude the occluder
                            if shape[o] == occluder_shape and color[o] == occluder_color and size[o] == occluder_size and mat[o] == occluder_mat:
                                #logger.info(f"found occluder in trace {i}")
                                continue
                            else:
                                logger.info(shape[o])
                                shape_pred_dict[shape[o].item()] += 1
                                size_pred_dict[size[o].item()] += 1
                                color_pred_dict[color[o].item()] += 1
                                mat_pred_dict[mat[o].item()] += 1

                                # check log weight of traces regarding their shape predictions
                                if shape[o].item() == 0: cube_log_wts[i] = log_wts[i].item()
                                elif shape[o].item() == 2: cylinder_log_wts[i] = log_wts[i].item()

            logger.info(f"{analysed_traces} used for uncertainty quantification...")
            logger.info(shape_pred_dict)
            shape_pred_dict = {k: v/analysed_traces for k, v in shape_pred_dict.items()}
            size_pred_dict = {k: v/analysed_traces for k, v in size_pred_dict.items()}
            color_pred_dict = {k: v/analysed_traces for k, v in color_pred_dict.items()}
            mat_pred_dict = {k: v/analysed_traces for k, v in mat_pred_dict.items()}
            
            logger.info(f"shape distributions: {shape_pred_dict}")
            logger.info(f"size distributions: {size_pred_dict}")
            logger.info(f"color distributions: {color_pred_dict}")
            logger.info(f"mat distributions: {mat_pred_dict}")

            preds = process_preds(prop_traces, resampling_id)
            logger.info(f"preds of resampled trace: {preds}")

            logger.info(f"cube traces log weigths: {cube_log_wts}")
            logger.info(f"cylinder traces log weigths: {cylinder_log_wts}")

            
            # for name, site in traces.nodes.items():                    
            #     # if site["type"] == "sample":
            #     #     logger.info(f"{name} - {site['value'].shape}")# - {site['value'][resampling_id]}")
                
            #     if name == 'image':
            #         for i in range(site["fn"].mean.shape[0]):
            #             output_image = site["fn"].mean[i]
            #             plt.imshow(visualize(output_image[:3].permute(1, 2, 0).cpu().numpy()))
            #             plt.savefig(os.path.join(occlusion_plots_dir, f"trace_{i}.png"))
            #             plt.close()



    inference_time = time.time() - init_time 
    logger.info(f'\nInference complete in {inference_time} seconds.')

if __name__ == '__main__':
    main()
    