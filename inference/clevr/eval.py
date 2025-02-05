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
        properties = json.load(open(os.path.join(dataset_path, 'scenes/CLEVR_val_scenes.json')))
        test_dataset = CLEVRDataset(images_path, properties, JOB_SPLIT)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, generator=torch.Generator(device='cuda'))

        logger.info(f"subset length: {len(test_dataset)}")
        
        n_test_samples = 0

        guide.eval()
        with torch.no_grad():
            for idx, (img, target_dict) in enumerate(testloader):
                img = img.to(device)
                target = process_targets(target_dict)
                img_index = target_dict['image_index'].item()

                logger.info(f"\ntarget image index: {img_index} - {n_test_samples}/{len(test_dataset)}")
                logger.info(f"# of objects: {len(target_dict['objects'])}")

                plt.imshow(visualize(img.squeeze(dim=0)[:3].permute(1, 2, 0).cpu().numpy()))
                plt.savefig(os.path.join(plots_dir, f"image_{img_index}.png"))
                plt.close()                

                # log_weights, model_trace, guide_trace = vectorized_importance_weights(model, guide, observations={"image": img},
                #                                                                       num_samples=params['num_inference_samples'],
                #                                                                       max_plate_nesting=0,
                #                                                                       normalized=False)                
                
                posterior = csis.run(observations={"image": img})
                prop_traces = posterior.prop_traces[0]
                traces = posterior.exec_traces[0]
                log_wts = posterior.log_weights[0]

                resampling = Empirical(torch.stack([torch.tensor(i) for i in range(len(log_wts))]), torch.stack(log_wts))
                resampling_id = resampling().item()

                logger.info(f"log weights: {[l.item() for l in log_wts]} - resampled trace: {resampling_id}")
                
                logger.info("\n")
                for name, site in traces.nodes.items():                    
                    # if site["type"] == "sample":
                    #     logger.info(f"{name} - {site['value'].shape}")# - {site['value'][resampling_id]}")
                    
                    if name == 'image':
                        for i in range(site["fn"].mean.shape[0]):
                            output_image = site["fn"].mean[i]
                            plt.imshow(visualize(output_image[:3].permute(1, 2, 0).cpu().numpy()))
                            plt.savefig(os.path.join(plots_dir, f"trace_{img_index}_{i}.png"))
                            plt.close()
                            

                preds = process_preds(prop_traces, resampling_id)
                for t in threshold: ap[t] += compute_AP(preds, target, t)
                n_test_samples += 1

                if n_test_samples == 10: break
                
        mAP = {k: v/n_test_samples for k, v in ap.items()}
        logger.info(f"distance thresholds: \n {threshold[0]} - {threshold[1]} - {threshold[2]} - {threshold[3]} - {threshold[4]} - {threshold[5]}")
        logger.info(f"mAP values: {mAP[threshold[0]]} - {mAP[threshold[1]]} - {mAP[threshold[2]]} - {mAP[threshold[3]]} - {mAP[threshold[4]]} - {mAP[threshold[5]]}\n")
        
        
        
        # n_test_samples = len(glob.glob(os.path.abspath(f'images/{COUNT}/*.png')))
        
        # # run the inference module
        # count_img_path = glob.glob(os.path.abspath(f'images/{COUNT}/*.png'))
        # count_img_path.sort()
        # logger.info(count_img_path)
        # for img_path in count_img_path:                             
            
        #     sample = img_to_tensor(Image.open(img_path))      
        #     sample = sample.to(device)
        #     sample_id = img_path.split('/')[-1].split('.')[0]
            
        #     plt.imshow(sample.squeeze(0).permute(1, 2, 0).detach().cpu().numpy())
        #     plt.savefig(f'{count_img_dir}/image_{sample_id}.png')
        #     plt.close()
            
        #     logger.info(sample_id)
            
        #     target_dict = json.load(open(os.path.abspath(f'metadata/{COUNT}/{sample_id}.json')))

        #     if PRINT_INFERENCE_TIME: since = time.time()
            
        #     # in ICSA set prediction, we assume that 'N' is known
        #     posterior = csis.run(observations={"image": sample}, N=COUNT)


        #         if params['inference_method'] == 'score_resample' and params['proposals'] == 'data_driven':

        #             # STEP 1: search for object-wise latent variables
        #             latent_vars = []
        #             hidden_vars = ['N']
        #             traces = posterior.prop_traces

        #             for tr in traces:
        #                 for name, site in tr.nodes.items():
        #                     if site['type'] == 'sample': 
        #                         if name not in hidden_vars and name not in latent_vars and int(name.split('_')[1]) < int(COUNT): 
        #                             latent_vars.append(name)
                    
        #             temp_v = {}
        #             for v in latent_vars:
        #                 object_id = v.split('_')[1]
        #                 if object_id not in temp_v: temp_v[object_id] = [v]
        #                 else: temp_v[object_id].append(v)
        #             latent_vars = list(temp_v.values())
                
        #             # STEP 2: for each object, iterate over all traces to score each one considering only one object at a time
        #             replace_params = {}
        #             include_ids = []
                    
        #             for vars in latent_vars: # 'vars' represent the group of latent variables associated with the same object
                        
        #                 #logger.info(f"loop over all traces to score vars {vars}\n")
        #                 vars_log_w = {}
        #                 vars_id = vars[0].split('_')[1]
        #                 include_ids.append(vars_id)
        #                 tracking_dict = {}

        #                 for t in range(len(traces)):
                
        #                     tracking_dict[t] = {}
                            
        #                     # mask -> False for all latent variables but 'vars'
        #                     for name, site in traces[t].nodes.items():
        #                         if site['type'] == 'sample':
        #                             if name not in vars: site['mask'] = False
        #                             else:  
        #                                 site['mask'] = True

        #                                 # track all hypotheses for the same object to get an idea of how close these are
        #                                 tracking_dict[t][name] = site['value']
        #                             del site['log_prob_sum']

        #                             if site['mask']: site['log_prob_sum'] = site['fn'].log_prob(site['value']).sum()
        #                             else: site['log_prob_sum'] = torch.tensor(0.)    
                            
        #                     model_trace = poutine.trace(poutine.replay(model, trace=traces[t])).get_trace(
        #                         observations={'image': sample},
        #                         show=vars,
        #                         N=COUNT
        #                         )
                            
        #                     for name, site in model_trace.nodes.items():
        #                         if site['type'] == 'sample': 
        #                             site['mask'] = True if name in vars or name == 'image' else False
        #                             try: del site['log_prob']
        #                             except: pass
                            
        #                     model_trace.compute_log_prob()
        #                     vars_log_w[t] = model_trace.log_prob_sum() - traces[t].log_prob_sum()
                        
        #                 # create an overlay img with all proposals for object 'vars_id'
        #                 img_transform = transforms.Compose([transforms.ToTensor()])
        #                 overlay_img = img_transform(render([tracking_dict[t][f"shape_{vars_id}"] for t in range(len(traces)) if t in tracking_dict],
        #                                                 [tracking_dict[t][f"size_{vars_id}"] for t in range(len(traces)) if t in tracking_dict],
        #                                                 [tracking_dict[t][f"color_{vars_id}"] for t in range(len(traces)) if t in tracking_dict],
        #                                                 [tracking_dict[t][f"locX_{vars_id}"] for t in range(len(traces)) if t in tracking_dict],
        #                                                 [tracking_dict[t][f"locY_{vars_id}"] for t in range(len(traces)) if t in tracking_dict],
        #                                                 background=None,
        #                                                 transparent_polygons=True
        #                                                 )
        #                                                 )
                        
        #                 # TO DO: save img overlay 
        #                 plt.imshow(overlay_img.permute(1, 2, 0).numpy())
        #                 plt.savefig(f'{count_img_dir}/traces_overlay_{sample_id}_vars_{vars_id}.png')
        #                 plt.close()

        #                 resampling = Empirical(torch.stack([torch.tensor(i) for i in range(len(traces)) if i in tracking_dict]), torch.stack([v for k, v in vars_log_w.items()]))
        #                 resampling_id = resampling().item()
                        
        #                 resampled_model_trace = poutine.trace(poutine.replay(model, trace=traces[resampling_id])).get_trace(
        #                     observations = {'image': sample},
        #                     show=vars,
        #                     N=COUNT
        #                     )
                        
        #                 for name, site in resampled_model_trace.nodes.items():
        #                     if site['type'] == 'sample': 
        #                         site['mask'] = True if name in vars or name == 'image' else False
        #                         try: del site['log_prob']
        #                         except: pass
                            
        #                     # save image at every object-wise SMC step
        #                     if name == 'image': # and vars == latent_vars[-1]:
        #                         plt.imshow(site["fn"].mean.squeeze().permute(1, 2, 0).cpu().numpy())
        #                         plt.savefig(f'{count_img_dir}/pred_{sample_id}_vars_{vars_id}.png')
        #                         plt.close()

                        
        #                 resampled_model_trace.compute_log_prob()                

        #                 # STEP 3: replace the params of 'vars' sample statements of all traces with the params of 'traces[resampling_id]'
        #                 for name, site in traces[resampling_id].nodes.items():
        #                     if name in vars: 
        #                         replace_params[name] = site
                            
                        
        #                 for t in range(len(traces)):
        #                     for name, site in traces[t].nodes.items():
        #                         if name in replace_params.keys(): 
        #                             msg = replace_params[name]
        #                             for k, v in msg.items(): site[k] = v
        #                         if site['type'] == 'sample': 
        #                             del site['log_prob_sum']

        #                             if site['mask']: site['log_prob_sum'] = site['fn'].log_prob(site['value']).sum()
        #                             else: site['log_prob_sum'] = torch.tensor(0.)
                    
                    
        #             # TO DO: make sure that all traces are equal now...
        #             try: trace = traces[resampling_id] 
        #             except:
        #                 logger.info(len(traces))

                
        #         elif params['inference_method'] == 'importance_sampling_only' and params['proposals'] == 'data_driven':

        #             resampling = Empirical(torch.stack([torch.tensor(i) for i in range(len(posterior.log_weights))]), torch.stack(posterior.log_weights))
        #             resampling_id = resampling().item()

        #             traces = posterior.prop_traces
        #             tracking_dict = {}
                    
        #             for t in range(len(traces)):
        #                 tracking_dict[t] = {}
        #                 for name, site in traces[t].nodes.items():
        #                     if site['type'] == 'sample': tracking_dict[t][name] = site['value']

        #             # create an overlay img with all proposals for object 'vars_id'
        #             img_transform = transforms.Compose([transforms.ToTensor()])
        #             overlay_img = img_transform(render([tracking_dict[t][f"shape_{v}"] for t in range(len(traces)) for v in range(COUNT)],
        #                                             [tracking_dict[t][f"size_{v}"] for t in range(len(traces)) for v in range(COUNT)],
        #                                             [tracking_dict[t][f"color_{v}"] for t in range(len(traces)) for v in range(COUNT)],
        #                                             [tracking_dict[t][f"locX_{v}"] for t in range(len(traces)) for v in range(COUNT)],
        #                                             [tracking_dict[t][f"locY_{v}"] for t in range(len(traces))  for v in range(COUNT)],
        #                                             background=None,
        #                                             transparent_polygons=True
        #                                             )
        #                                             )
                    
        #             plt.imshow(overlay_img.permute(1, 2, 0).numpy())
        #             plt.savefig(f'{count_img_dir}/traces_overlay_{sample_id}.png')
        #             plt.close()

        #             resampled_img = img_transform(render([tracking_dict[resampling_id][f"shape_{v}"] for v in range(COUNT)],
        #                                             [tracking_dict[resampling_id][f"size_{v}"] for v in range(COUNT)],
        #                                             [tracking_dict[resampling_id][f"color_{v}"] for v in range(COUNT)],
        #                                             [tracking_dict[resampling_id][f"locX_{v}"] for v in range(COUNT)],
        #                                             [tracking_dict[resampling_id][f"locY_{v}"] for v in range(COUNT)],
        #                                             background=None
        #                                             )
        #                                             )
                    
        #             plt.imshow(resampled_img.permute(1, 2, 0).numpy())
        #             plt.savefig(f'{count_img_dir}/pred_{sample_id}.png')
        #             plt.close()

                
        #         else: raise ValueError(f"{params['inference_method']} is not valid!")
                
        #         if PRINT_INFERENCE_TIME: 
        #             time_elapsed = time.time() - since
        #             logger.info(f'Inference complete in {time_elapsed*1000}ms')      
        #             break          
                
        #         preds = process_preds(traces[resampling_id], COUNT)
        #         targets = process_targets(target_dict)
                    
        #         for t in threshold: ap[t] += compute_AP(preds, targets, t)
            
        #     mAP = {k: v/n_test_samples for k, v in ap.items()}
        #     logger.info(f"COUNT {COUNT}: distance thresholds: \n {threshold[0]} - {threshold[1]} - {threshold[2]} - {threshold[3]} - {threshold[4]} - {threshold[5]}")
        #     logger.info(f"COUNT {COUNT}: mAP values: {mAP[threshold[0]]} - {mAP[threshold[1]]} - {mAP[threshold[2]]} - {mAP[threshold[3]]} - {mAP[threshold[4]]} - {mAP[threshold[5]]}\n")

    inference_time = time.time() - init_time 
    logger.info(f'\nInference complete in {inference_time} seconds.')

if __name__ == '__main__':
    main()
    