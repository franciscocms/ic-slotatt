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
sys.path.append(os.path.abspath(__file__+'/../../'))

main_dir = os.path.abspath(__file__+'/../../')

from main import models
from main.models import occlusion
from main import modifiedCSIS as mcsis
from main import modifiedImportance as mImportance
from utils.guide import load_trained_guide
from utils.generate import img_to_tensor, render
from inference.guide import InvSlotAttentionGuide
from utils.distributions import Empirical
from utils.baseline import compute_AP
from main.setup import params

import logging
logfile_name = f'icsa_set_prediction.log'
logger = logging.getLogger(params["running_type"])
logger.setLevel(logging.INFO)
fh = logging.FileHandler(logfile_name, mode='w')
logger.addHandler(fh)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_default_device(device)
main_dir = os.path.abspath(__file__+'/../../')

shape_vals = {'ball': 0, 'square': 1}
size_vals = {'small': 0 , 'medium': 1, 'large': 2}
color_vals = {'red': 0, 'green': 1, 'blue': 2}

PRINT_INFERENCE_TIME = False

def process_preds(trace, id):
    
    """ returns a matrix of predictions from the proposed trace """
    features_dim = 11
    preds = torch.zeros(params['max_objects'], features_dim) # 11 - the dimension of all latent variables (countinuous and discrete after OHE)
    for name, site in trace.nodes.items():
        if site['type'] == 'sample':
            if name == 'shape': preds[:, :2] = F.one_hot(site['value'][id], len(shape_vals))
            if name == 'size':  preds[:, 2:5] = F.one_hot(site['value'][id], len(size_vals))
            if name == 'color': preds[:, 5:8] = F.one_hot(site['value'][id], len(color_vals))
            if name == 'locX': preds[:, 8] = site['value'][id]
            if name == 'locY': preds[:, 9] = site['value'][id]
            if name == 'mask': preds[:, 10] = site['value'][id]
    return preds

def process_targets(target_dict):   
    
    n = int(target_dict['scene_attr']['N'])
    target = torch.zeros(params['max_objects'], 11) # 11 - the dimension of all latent variables (countinuous and discrete after OHE)
    for i in range(n):
        target[i, :2] = F.one_hot(torch.tensor(shape_vals[target_dict['scene_attr'][f'object_{i}']['shape']]), len(shape_vals))
        target[i, 2:5] = F.one_hot(torch.tensor(size_vals[target_dict['scene_attr'][f'object_{i}']['size']]), len(size_vals))
        target[i, 5:8] = F.one_hot(torch.tensor(color_vals[target_dict['scene_attr'][f'object_{i}']['color']]), len(color_vals))
        target[i, 8] = target_dict['scene_attr'][f'object_{i}']['initLocX']
        target[i, 9] = target_dict['scene_attr'][f'object_{i}']['initLocY']
        target[i, 10] = torch.tensor(1.)
    return target
    
def main():    

    assert params["running_type"] == "eval"

    logger.info(device)

    seeds = [1]

    OOD_EVAL = params["ood_eval"]
    
    for seed in seeds: 
        
        pyro.set_rng_seed(seed)
        
        threshold = [-1., 1., 0.5, 0.25, 0.125, 0.0625]
        
        model = models.model
        
        # set up trained guide and 'csis' object
        guide = InvSlotAttentionGuide(resolution = (128, 128),
                                    num_iterations = 3,
                                    hid_dim = 64,
                                    stage = "eval"
                                    ).to(device)
        
        GUIDE_PATH = f"{main_dir}/checkpoint-{params['jobID']}/guide_{params['guide_step']}.pth"
        
        if os.path.isfile(GUIDE_PATH): guide = load_trained_guide(guide, GUIDE_PATH)
        else: raise ValueError(f'{GUIDE_PATH} is not a valid path!')
        
        logger.info(f'seed {seed}')
        logger.info(GUIDE_PATH)

        optimiser = pyro.optim.Adam({'lr': 1e-4})
        csis = mcsis.CSIS(model, guide, optimiser, training_batch_size=256, num_inference_samples=params["num_inference_samples"])

        plots_dir = os.path.abspath("set_prediction_plots")
        if not os.path.isdir(plots_dir): os.mkdir(plots_dir)
        else: 
            shutil.rmtree(plots_dir)
            os.mkdir(plots_dir)
        
        all_mAP = {k: [] for k in threshold}
        
        for COUNT in range(1, 7):

            count_img_dir = os.path.join(plots_dir, str(COUNT))
            if not os.path.isdir(count_img_dir): os.mkdir(count_img_dir)

            logger.info(f'\nEVALUATION STARTED FOR SCENES WITH {COUNT} OBJECTS\n')

            
            ap = {k: 0 for k in threshold}

            if not OOD_EVAL: n_test_samples = len(glob.glob(os.path.abspath(f'images/{COUNT}/*.png')))
            else: n_test_samples = len(glob.glob(os.path.abspath(f'images_ood/{COUNT}/*.png')))
            
            # run the inference module
            if not OOD_EVAL: count_img_path = glob.glob(os.path.abspath(f'images/{COUNT}/*.png'))
            else: count_img_path = glob.glob(os.path.abspath(f'images_ood/{COUNT}/*.png'))

            count_img_path.sort()
            #logger.info(count_img_path)
            for img_path in count_img_path:                             
                
                sample = img_to_tensor(Image.open(img_path))      
                sample = sample.to(device)
                sample_id = img_path.split('/')[-1].split('.')[0]
                
                plt.imshow(sample.squeeze(0).permute(1, 2, 0).detach().cpu().numpy())
                plt.savefig(f'{count_img_dir}/image_{sample_id}.png')
                plt.close()
                
                #logger.info(sample_id)
                
                if not OOD_EVAL: target_dict = json.load(open(os.path.abspath(f'metadata/{COUNT}/{sample_id}.json')))
                else: target_dict = json.load(open(os.path.abspath(f'metadata_ood/{COUNT}/{sample_id}.json')))

                # logger.info(target_dict)

                if PRINT_INFERENCE_TIME: since = time.time()
                
                posterior = csis.run(observations={"image": sample})
                prop_traces = posterior.prop_traces[0]
                traces = posterior.exec_traces[0]
                log_wts = posterior.log_weights[0]


                if params['inference_method'] == 'score_resample' and params['proposals'] == 'data_driven':

                    # STEP 1: search for object-wise latent variables
                    latent_vars = []
                    hidden_vars = ['N']
                    traces = posterior.prop_traces

                    for tr in traces:
                        for name, site in tr.nodes.items():
                            if site['type'] == 'sample': 
                                if name not in hidden_vars and name not in latent_vars and int(name.split('_')[1]) < int(COUNT): 
                                    latent_vars.append(name)
                    
                    temp_v = {}
                    for v in latent_vars:
                        object_id = v.split('_')[1]
                        if object_id not in temp_v: temp_v[object_id] = [v]
                        else: temp_v[object_id].append(v)
                    latent_vars = list(temp_v.values())
                
                    # STEP 2: for each object, iterate over all traces to score each one considering only one object at a time
                    replace_params = {}
                    include_ids = []
                    
                    for vars in latent_vars: # 'vars' represent the group of latent variables associated with the same object
                        
                        #logger.info(f"loop over all traces to score vars {vars}\n")
                        vars_log_w = {}
                        vars_id = vars[0].split('_')[1]
                        include_ids.append(vars_id)
                        tracking_dict = {}

                        for t in range(len(traces)):
                
                            tracking_dict[t] = {}
                            
                            # mask -> False for all latent variables but 'vars'
                            for name, site in traces[t].nodes.items():
                                if site['type'] == 'sample':
                                    if name not in vars: site['mask'] = False
                                    else:  
                                        site['mask'] = True

                                        # track all hypotheses for the same object to get an idea of how close these are
                                        tracking_dict[t][name] = site['value']
                                    del site['log_prob_sum']

                                    if site['mask']: site['log_prob_sum'] = site['fn'].log_prob(site['value']).sum()
                                    else: site['log_prob_sum'] = torch.tensor(0.)    
                            
                            model_trace = poutine.trace(poutine.replay(model, trace=traces[t])).get_trace(
                                observations={'image': sample},
                                show=vars,
                                N=COUNT
                                )
                            
                            for name, site in model_trace.nodes.items():
                                if site['type'] == 'sample': 
                                    site['mask'] = True if name in vars or name == 'image' else False
                                    try: del site['log_prob']
                                    except: pass
                            
                            model_trace.compute_log_prob()
                            vars_log_w[t] = model_trace.log_prob_sum() - traces[t].log_prob_sum()
                        
                        # create an overlay img with all proposals for object 'vars_id'
                        img_transform = transforms.Compose([transforms.ToTensor()])
                        overlay_img = img_transform(render([tracking_dict[t][f"shape_{vars_id}"] for t in range(len(traces)) if t in tracking_dict],
                                                        [tracking_dict[t][f"size_{vars_id}"] for t in range(len(traces)) if t in tracking_dict],
                                                        [tracking_dict[t][f"color_{vars_id}"] for t in range(len(traces)) if t in tracking_dict],
                                                        [tracking_dict[t][f"locX_{vars_id}"] for t in range(len(traces)) if t in tracking_dict],
                                                        [tracking_dict[t][f"locY_{vars_id}"] for t in range(len(traces)) if t in tracking_dict],
                                                        background=None,
                                                        transparent_polygons=True
                                                        )
                                                        )
                        
                        # TO DO: save img overlay 
                        plt.imshow(overlay_img.permute(1, 2, 0).numpy())
                        plt.savefig(f'{count_img_dir}/traces_overlay_{sample_id}_vars_{vars_id}.png')
                        plt.close()

                        resampling = Empirical(torch.stack([torch.tensor(i) for i in range(len(traces)) if i in tracking_dict]), torch.stack([v for k, v in vars_log_w.items()]))
                        resampling_id = resampling().item()
                        
                        resampled_model_trace = poutine.trace(poutine.replay(model, trace=traces[resampling_id])).get_trace(
                            observations = {'image': sample},
                            show=vars,
                            N=COUNT
                            )
                        
                        for name, site in resampled_model_trace.nodes.items():
                            if site['type'] == 'sample': 
                                site['mask'] = True if name in vars or name == 'image' else False
                                try: del site['log_prob']
                                except: pass
                            
                            # save image at every object-wise SMC step
                            if name == 'image': # and vars == latent_vars[-1]:
                                plt.imshow(site["fn"].mean.squeeze().permute(1, 2, 0).cpu().numpy())
                                plt.savefig(f'{count_img_dir}/pred_{sample_id}_vars_{vars_id}.png')
                                plt.close()

                        
                        resampled_model_trace.compute_log_prob()                

                        # STEP 3: replace the params of 'vars' sample statements of all traces with the params of 'traces[resampling_id]'
                        for name, site in traces[resampling_id].nodes.items():
                            if name in vars: 
                                replace_params[name] = site
                            
                        
                        for t in range(len(traces)):
                            for name, site in traces[t].nodes.items():
                                if name in replace_params.keys(): 
                                    msg = replace_params[name]
                                    for k, v in msg.items(): site[k] = v
                                if site['type'] == 'sample': 
                                    del site['log_prob_sum']

                                    if site['mask']: site['log_prob_sum'] = site['fn'].log_prob(site['value']).sum()
                                    else: site['log_prob_sum'] = torch.tensor(0.)
                    
                    
                    # TO DO: make sure that all traces are equal now...
                    try: trace = traces[resampling_id] 
                    except:
                        logger.info(len(traces))

                
                elif params['inference_method'] == 'importance_sampling_only' and params['proposals'] == 'data_driven':

                    resampling = Empirical(torch.stack([torch.tensor(i) for i in range(len(log_wts))]), torch.stack(log_wts))
                    resampling_id = resampling().item()

                    logger.info(f"log weights: {[l.item() for l in log_wts]} - resampled trace: {resampling_id}")

                    for name, site in traces.nodes.items():                    
                        if name == 'image':
                            for i in range(site["fn"].mean.shape[0]):
                                output_image = site["fn"].mean[i]
                                plt.imshow(output_image.permute(1, 2, 0).cpu().numpy())
                                plt.savefig(f'{count_img_dir}/image_{sample_id}_trace_{i}.png')
                                plt.close()






                    # tracking_dict = {}
                    
                    # for t in range(len(traces)):
                    #     tracking_dict[t] = {}
                    #     for name, site in traces[t].nodes.items():
                    #         if site['type'] == 'sample': tracking_dict[t][name] = site['value']

                    # # create an overlay img with all proposals for object 'vars_id'
                    # img_transform = transforms.Compose([transforms.ToTensor()])
                    # overlay_img = img_transform(render([tracking_dict[t][f"shape_{v}"] for t in range(len(traces)) for v in range(COUNT)],
                    #                                 [tracking_dict[t][f"size_{v}"] for t in range(len(traces)) for v in range(COUNT)],
                    #                                 [tracking_dict[t][f"color_{v}"] for t in range(len(traces)) for v in range(COUNT)],
                    #                                 [tracking_dict[t][f"locX_{v}"] for t in range(len(traces)) for v in range(COUNT)],
                    #                                 [tracking_dict[t][f"locY_{v}"] for t in range(len(traces))  for v in range(COUNT)],
                    #                                 background=None,
                    #                                 transparent_polygons=True
                    #                                 )
                    #                                 )
                    
                    # plt.imshow(overlay_img.permute(1, 2, 0).numpy())
                    # plt.savefig(f'{count_img_dir}/traces_overlay_{sample_id}.png')
                    # plt.close()

                    # resampled_img = img_transform(render([tracking_dict[resampling_id][f"shape_{v}"] for v in range(COUNT)],
                    #                                 [tracking_dict[resampling_id][f"size_{v}"] for v in range(COUNT)],
                    #                                 [tracking_dict[resampling_id][f"color_{v}"] for v in range(COUNT)],
                    #                                 [tracking_dict[resampling_id][f"locX_{v}"] for v in range(COUNT)],
                    #                                 [tracking_dict[resampling_id][f"locY_{v}"] for v in range(COUNT)],
                    #                                 background=None
                    #                                 )
                    #                                 )
                    
                    # plt.imshow(resampled_img.permute(1, 2, 0).numpy())
                    # plt.savefig(f'{count_img_dir}/pred_{sample_id}.png')
                    # plt.close()

                
                else: raise ValueError(f"{params['inference_method']} is not valid!")
                
                if PRINT_INFERENCE_TIME: 
                    time_elapsed = time.time() - since
                    logger.info(f'Inference complete in {time_elapsed*1000}ms')      
                    #break          
                
                preds = process_preds(prop_traces, resampling_id)
                targets = process_targets(target_dict)

                # logger.info(preds)
                    
                for t in threshold: ap[t] += compute_AP(preds, targets, t)

                break
            
            mAP = {k: v/n_test_samples for k, v in ap.items()}
            logger.info(f"COUNT {COUNT}: distance thresholds: \n {threshold[0]} - {threshold[1]} - {threshold[2]} - {threshold[3]} - {threshold[4]} - {threshold[5]}")
            logger.info(f"COUNT {COUNT}: mAP values: {mAP[threshold[0]]} - {mAP[threshold[1]]} - {mAP[threshold[2]]} - {mAP[threshold[3]]} - {mAP[threshold[4]]} - {mAP[threshold[5]]}\n")
            
            for k in threshold:
                all_mAP[k].append(mAP[k])
            break

        logger.info(f"Average mAP: ")
        for k in threshold:
            logger.info(f"{k}: {np.mean(all_mAP[k])}")


if __name__ == '__main__':
    main()