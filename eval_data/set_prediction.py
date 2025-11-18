import pyro
import pyro.distributions as dist
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
from statistics import mean, stdev

# add project path to sys to import relative modules
import sys
sys.path.append(os.path.abspath(__file__+'/../../'))

main_dir = os.path.abspath(__file__+'/../../')

from main import models
from main import modifiedCSIS as mcsis
from main import modifiedImportance as mImportance
from utils.guide import load_trained_guide
from utils.generate import img_to_tensor, render
from inference.guide import InvSlotAttentionGuide
from utils.distributions import Empirical, MyNormal
from utils.baseline import compute_AP
from main.setup import params

import logging
logfile_name = f"eval-{params['jobID']}.log"
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

shape_lib = list(shape_vals.keys())
size_lib = list(size_vals.keys())
color_lib = list(color_vals.keys())

img_transform = transforms.Compose([transforms.ToTensor()])

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

    threshold = [-1., 1., 0.5, 0.25, 0.125, 0.0625]
    all_mAP = {k: [] for k in threshold}
    
    for seed in seeds: 
        
        pyro.set_rng_seed(seed)
    
        model = models.model
        
        # set up trained guide and 'csis' object
        guide = InvSlotAttentionGuide(resolution = (128, 128),
                                    num_iterations = 3,
                                    slot_dim = params["slot_dim"],
                                    stage = "eval"
                                    ).to(device)
        
        GUIDE_PATH = f"{main_dir}/checkpoint-{params['jobID']}/guide_{params['guide_step']}.pth"
        
        if os.path.isfile(GUIDE_PATH): guide = load_trained_guide(guide, GUIDE_PATH)
        else: raise ValueError(f'{GUIDE_PATH} is not a valid path!')
        
        logger.info(f'\n\nseed {seed}\n\n')
        logger.info(GUIDE_PATH)

        optimiser = pyro.optim.Adam({'lr': 1e-4})
        csis = mcsis.CSIS(model, guide, optimiser, training_batch_size=256, num_inference_samples=params["num_inference_samples"])

        plots_dir = os.path.abspath("set_prediction_plots")
        if not os.path.isdir(plots_dir): os.mkdir(plots_dir)
        else: 
            shutil.rmtree(plots_dir)
            os.mkdir(plots_dir)
        
        
        
        for COUNT in range(20, 21):

            count_img_dir = os.path.join(plots_dir, str(COUNT))
            if not os.path.isdir(count_img_dir): os.mkdir(count_img_dir)

            logger.info(f'\nEVALUATION STARTED FOR SCENES WITH {COUNT} OBJECTS\n')

            
            ap = {k: [] for k in threshold}

            n_test_samples = len(glob.glob(os.path.abspath(f'images_ood/{COUNT}/*.png')))
            count_img_path = glob.glob(os.path.abspath(f'images_ood/{COUNT}/*.png'))

            count_img_path.sort()
            
            resampled_logwts = {k: {} for k in range(len(count_img_path))}

            ap_den = 0
            for img_idx, img_path in enumerate(count_img_path): 
                
                if img_idx % 100 == 0:
                    logger.info(f"{img_idx}")
                
                sample = img_to_tensor(Image.open(img_path))      
                sample = sample.to(device)
                sample_id = img_path.split('/')[-1].split('.')[0]
                
                # plt.imshow(sample.squeeze(0).permute(1, 2, 0).detach().cpu().numpy())
                # plt.savefig(f'{count_img_dir}/image_{sample_id}.png')
                # plt.close()
                
                target_dict = json.load(open(os.path.abspath(f'metadata_ood/{COUNT}/{sample_id}.json')))

                # logger.info(target_dict)

                if PRINT_INFERENCE_TIME: since = time.time()
                
                posterior = csis.run(observations={"image": sample})
                prop_traces = posterior.prop_traces[0]  # guide
                traces = posterior.exec_traces[0]       # model
                log_wts = posterior.log_weights[0]      

                compute_ap_flag = True

                if params['inference_method'] == 'score_resample' and params['proposals'] == 'data_driven':

                    
                    # get the predictions of each trace
                    preds = torch.stack([process_preds(prop_traces, i) for i in range(len(log_wts))]) # [nif, M, feature_dim]

                    # only use the particles that got the # of objects right
                    correct_count_idx = [p for p, pred in enumerate(preds) if torch.sum(pred[:, -1]) == COUNT]
                    preds = preds[correct_count_idx]

                    #logger.info(preds.shape)
                    if len(preds.shape) == 2: preds = preds.unsqueeze(0)

                    if preds.shape[0] != 0: 

                        # remove padded objects from each particle
                        real_preds = []
                        for p, pred in enumerate(preds):
                            real_objects_idx = [int(i) for i in list(torch.nonzero(pred[:, -1]))]
                            real_preds.append(pred[real_objects_idx])
                        preds = torch.stack(real_preds)

                        # logger.info(f"after selecting the particles with correct counting: {preds.shape}")

                        # permute them according to the order defined by location (euclidean distance)
                        x = preds[:, :, 8]
                        y = preds[:, :, 9]

                        distance = torch.sqrt(x**2 + y**2) # [nif, M]
                        sorted, indices = torch.sort(distance, dim=-1)
                        sorted_preds = torch.gather(preds, 1, indices.unsqueeze(-1).expand(-1, -1, preds.shape[-1])) # [nif, M, feature_dim]
                        
                        for o in range(COUNT):
                            
                            #logger.info(f"starting score-resample procedure for object {o}...")
                            
                            scenes = []
                            for p, particle in enumerate(sorted_preds):
                                render_objects = particle[:o+1, :]

                                scene = []
                                for s in range(render_objects.shape[0]):
                                    shape = torch.argmax(render_objects[s, :2], dim=-1)
                                    size = torch.argmax(render_objects[s, 2:5], dim=-1)
                                    color = torch.argmax(render_objects[s, 5:8], dim=-1)
                                    locx, locy = render_objects[s, 8], render_objects[s, 9]
                                    
                                    scene.append({
                                        "shape": shape_lib[shape.item()],
                                        "color": color_lib[color.item()],
                                        "size": size_lib[size.item()],
                                        "position": (locx.item(), locy.item())
                                    })
                                scenes.append(scene)

                            # render only the properties of objects 1:o
                            rendered_particles = render(scenes)
                            particles = torch.stack([img_transform(s) for s in rendered_particles])

                            # evaluate the likelihood of each generated image against the observation (iteration log weights)
                            partial_likelihood_fn = MyNormal(particles, torch.tensor(0.05)).get_dist()
                            partial_likelihood = torch.sum(partial_likelihood_fn.log_prob(sample), dim=[1, 2, 3])/(sample.shape[-1]**2)

                            # logger.info(partial_likelihood)
                            
                            # choose the trace with best likelihood
                            resampling = Empirical(torch.stack([torch.tensor(i) for i in range(len(partial_likelihood))]), partial_likelihood)
                            resampling_id = resampling().item()

                            #logger.info(f"particle {resampling_id} chosen with likelihood {partial_likelihood[resampling_id]}")

                            resampled_logwts[img_idx][o] = torch.mean(partial_likelihood)

                            # save chosen image
                            # plt.imshow(particles[resampling_id].permute(1, 2, 0).cpu().numpy())
                            # plt.savefig(f'{count_img_dir}/image_{sample_id}_trace_{o}.png')
                            # plt.close()


                            # assign the chosen object features to all particles
                            # logger.info(f"sorted preds shape: {sorted_preds.shape}")
                            
                            for p in range(sorted_preds.shape[0]):
                                sorted_preds[p, o] = sorted_preds[resampling_id, o]
                            
                            # logger.info(f"score-resample procedure done for object {o}...")
                            #logger.info(f"sorted_preds after iteration {o}: {sorted_preds}\n")

                    else:
                        compute_ap_flag = False

                
                elif params['inference_method'] == 'importance_sampling_only' and params['proposals'] == 'data_driven':

                    resampling = Empirical(torch.stack([torch.tensor(i) for i in range(len(log_wts))]), torch.stack(log_wts))
                    resampling_id = resampling().item()

                    # logger.info(f"log weights: {[l.item() for l in log_wts]} - resampled trace: {resampling_id}")

                    # for name, site in traces.nodes.items():                    
                    #     if name == 'image':
                    #         for i in range(site["fn"].mean.shape[0]):
                    #             output_image = site["fn"].mean[i]
                    #             plt.imshow(output_image.permute(1, 2, 0).cpu().numpy())
                    #             plt.savefig(f'{count_img_dir}/image_{sample_id}_trace_{i}.png')
                    #             plt.close()

                    #     if site["type"] == "sample": 
                    #         logger.info(f"{name} - {site['fn']} - {site['value']} - {site['fn'].log_prob(site['value'])}")

                
                else: raise ValueError(f"{params['inference_method']} is not valid!")
                
                if PRINT_INFERENCE_TIME: 
                    time_elapsed = time.time() - since
                    logger.info(f'Inference complete in {time_elapsed*1000}ms')      
                    #break          
                
                if compute_ap_flag:
                    preds = process_preds(prop_traces, resampling_id) if params['inference_method'] == 'importance_sampling_only' else sorted_preds[0]
                    
                    #logger.info(preds)
                    
                    targets = process_targets(target_dict)

                    #logger.info(targets)

                    for t in threshold: ap[t].append(compute_AP(preds, targets, t))

                ap_den += 1
                
                #if img_idx == 10: break
            
            #logger.info(resampled_logwts)
            
            if params['inference_method'] == 'score_resample':
                avg_log_wts = {k: [] for k in range(COUNT)}
                for s, count_dict in resampled_logwts.items():
                    if len(count_dict) > 0:
                        for k in range(COUNT):
                            avg_log_wts[k].append(count_dict[k].item())
                
            # logger.info("\naveraged log_wts across all inference iterations:")
            # logger.info(avg_log_wts)

            mAP_mean = {k: mean(v) for k, v in ap.items()}
            mAP_std = {k: stdev(v) for k, v in ap.items()}
            for t_idx, t in enumerate(threshold):
                logger.info(f"{t}: {round(mAP_mean[threshold[t_idx]], 3)} +- {round(mAP_std[threshold[t_idx]], 3)}")

        




if __name__ == '__main__':
    main()