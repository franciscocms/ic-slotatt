import pyro
import pyro.poutine as poutine
import torch
from torchvision import transforms
import numpy as np

from sklearn.metrics import accuracy_score

import os
import glob
from PIL import Image
import matplotlib.pyplot as plt

# add project path to sys to import relative modules
import sys
sys.path.append(os.path.abspath(__file__+'/../../'))

import logging
logfile_name = f'dme.log'
logger = logging.getLogger("eval_count")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(logfile_name, mode='w')
logger.addHandler(fh)

from main import models
from main import modifiedCSIS as mcsis
from main import modifiedImportance as mImportance
from utils.guide import load_trained_guide
from utils.generate import img_to_tensor, render
from inference.guide import InvSlotAttentionGuide
from utils.distributions import Empirical
from main import setup

# imports for testing with neural baseline
from synthetic_data.network import Model as DME
from synthetic_data.network import Unet
from synthetic_data.dataset import MyDataset
from synthetic_data.test import Tester

params = setup.params
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_default_device(device)

main_dir = os.path.abspath(__file__+'/../../')

results_dir = f"{main_dir}/results"
if not os.path.isdir(results_dir): os.mkdir(results_dir)
counting_dir = f"{results_dir}/counting"
if not os.path.isdir(counting_dir): os.mkdir(counting_dir)

def main():    

    if params['inference_method'] == 'score_resample' and params['proposals'] == 'data_driven':

        logger.info(device)

        seeds = [1, 2, 3, 4, 5]
        for seed in seeds:
            
            logger.info(f'seed {seed}')
            pyro.set_rng_seed(seed)
        
            model = models.model
            
            # set up trained guide and 'csis' object
            guide = InvSlotAttentionGuide(resolution = (128, 128),
                                        num_iterations = 3,
                                        hid_dim = 64,
                                        stage = "eval",
                                        mixture_components=params["mixture_components"])
            guide.to(device)
            GUIDE_PATH = f"{main_dir}/checkpoint-{params['jobID']}/guide_{params['guide_step']}.pth"
            if os.path.isfile(GUIDE_PATH): guide = load_trained_guide(guide, GUIDE_PATH)
            else: raise ValueError(f'{GUIDE_PATH} is not a valid path!')

            logger.info(GUIDE_PATH)

            optimiser = pyro.optim.Adam({'lr': 1e-4})
            csis = mcsis.CSIS(model, guide, optimiser, training_batch_size=256, num_inference_samples=params["num_inference_samples"])
            
            for COUNT in range(1, 11):
            
                logger.info(f'\nEVALUATION STARTED FOR SCENES WITH {COUNT} OBJECTS\n')
                
                count_vals = []
                # check if correspondent folders of images and metadata are not empty
                #assert len(os.listdir(os.path.abspath('images'))) != 0, f'directory of evaluation images with {COUNT} objects is empty!'
                #assert len(os.listdir(os.path.abspath('metadata'))) != 0, f'directory of evaluation metadata with {COUNT} objects is empty!'
                
                # run the inference module
                imgs_path = glob.glob(os.path.abspath(f'images/{COUNT}/*.png'))          
                imgs_path.sort()
                for img_path in imgs_path:
                    
                    # create a folder to aggregate all results for each evaluation image
                    if not os.path.isdir(f"{counting_dir}/{COUNT}"): os.mkdir(f"{counting_dir}/{COUNT}")             
                    
                    sample = img_to_tensor(Image.open(img_path))
                    sample = sample.to(device)
                    sample_id = img_path.split('/')[-1].split('.')[0]

                    logger.info(sample_id)

                    criterion = 0.
                    accepted_logl = 50000.
                    logl_reached = False
                    N_MIN = 1
                    N_MAX = 11

                    best_traces = {}
                    best_logl = {}

                    for iter_n in range(N_MIN, N_MAX):
                        posterior = csis.run(observations={"image": sample}, N=iter_n)

                        # STEP 1: search for object-wise latent variables
                        latent_vars = []
                        hidden_vars = ['N']
                        traces = posterior.prop_traces

                        #logger.info(f'iter {iter_n}: # of traces is {len(traces)}')

                        # only go through counting hypotheses that allow to generate at least 1/5 of the original # of particles
                        if len(traces) >= 10: 

                            for tr in traces:
                                for name, site in tr.nodes.items():
                                    if site['type'] == 'sample': 
                                        if name not in hidden_vars and name not in latent_vars and int(name.split('_')[1]) < int(iter_n): 
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
                                    
                                    # mask all latent variables but 'vars'
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
                                        N=iter_n
                                        )
                                    
                                    for name, site in model_trace.nodes.items():
                                        if site['type'] == 'sample': 
                                            site['mask'] = True if name in vars or name == 'image' else False
                                            try: del site['log_prob']
                                            except: pass
                                    
                                    model_trace.compute_log_prob()
                                    vars_log_w[t] = model_trace.log_prob_sum() - traces[t].log_prob_sum()


                                resampling = Empirical(torch.stack([torch.tensor(i) for i in range(len(traces)) if i in tracking_dict]), torch.stack([v for k, v in vars_log_w.items()]))
                                resampling_id = resampling().item()
                                resampled_model_trace = poutine.trace(poutine.replay(model, trace=traces[resampling_id])).get_trace(
                                    observations = {'image': sample},
                                    show=vars,
                                    N=iter_n
                                    )
                                
                                for name, site in resampled_model_trace.nodes.items():
                                    if site['type'] == 'sample': 
                                        site['mask'] = True if name in vars or name == 'image' else False
                                        try: del site['log_prob']
                                        except: pass
                                
                                resampled_model_trace.compute_log_prob()

                                for name, site in resampled_model_trace.nodes.items():
                                    if name == 'image':
                                        if vars == latent_vars[-1]: 
                                            criterion = site['fn'].log_prob(site['value'])   
                                            best_traces[iter_n] = [s for _, s in resampled_model_trace.nodes.items()]
                                            best_logl[iter_n] = criterion
                                

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
                    
                            if criterion >= accepted_logl: 
                                logl_reached = True
                                break

                    if logl_reached: 
                        logger.info(f'Inference stopped with {iter_n} objects explaining the data...')
                        count_vals.append(iter_n)
                    else:
                        logger.info(f'Likelihood threshold was not reached. Searching for best hypothesis: ')    
                        max_logl = max([logl for iter, logl in best_logl.items()])
                        max_n = [k for k, v in best_logl.items() if v == max_logl]
                        logger.info(f'Best hypothesis corresponds to explaining the data with {max_n} objects (log_likelihood = {max_logl})\n') 
                        count_vals.append(max_n[0])

                acc = accuracy_score(count_vals, [COUNT for _ in range(len(count_vals))])

                logger.info(f"counting results for scenes with {COUNT} objects: {count_vals} with acc: {acc}")
    
        
    elif params['inference_method'] == 'importance_sampling_only' and params['proposals'] == 'data_driven':
        
        seeds = [1, 2, 3, 4, 5]
        for seed in seeds:
        
            pyro.set_rng_seed(seed)

            model = models.model
            
            # set up trained guide and 'csis' object
            guide = InvSlotAttentionGuide(resolution = (128, 128),
                                        num_iterations = 3,
                                        hid_dim = 64,
                                        stage = "eval",
                                        mixture_components=params["mixture_components"])
            guide.to(device)
            #GUIDE_PATH = f"{main_dir}/checkpoint-{params['jobID']}/guide_{params['guide_step']}.pth"
            GUIDE_PATH = f"/Users/franciscosilva/Downloads/guide_{params['guide_step']}.pth"
            
            logger.info(f'\nseed {seed}')
            logger.info(f'{GUIDE_PATH}\n')
            
            if os.path.isfile(GUIDE_PATH): guide = load_trained_guide(guide, GUIDE_PATH)
            else: raise ValueError(f'{GUIDE_PATH} is not a valid path!')

            optimiser = pyro.optim.Adam({'lr': 1e-4})
            csis = mcsis.CSIS(model, guide, optimiser, training_batch_size=256, num_inference_samples=params["num_inference_samples"])
            
            
            for COUNT in range(1, 11):
            
                logger.info(f'\nEVALUATION STARTED FOR SCENES WITH {COUNT} OBJECTS\n')
                
                # check if correspondent folders of images and metadata are not empty
                #assert len(os.listdir(os.path.abspath('images'))) != 0, f'directory of evaluation images with {COUNT} objects is empty!'
                #assert len(os.listdir(os.path.abspath('metadata'))) != 0, f'directory of evaluation metadata with {COUNT} objects is empty!'
                
                # run the inference module
                preds = []
                imgs_path = glob.glob(f'/Users/franciscosilva/Downloads/eval_data/images/{COUNT}/*.png')
                imgs_path.sort()
                #for img_path in glob.glob(os.path.abspath(f'images/{COUNT}/*.png')):
                for img_path in imgs_path:

                    sample = img_to_tensor(Image.open(img_path))      
                    sample = sample.to(device)
                    sample_id = img_path.split('/')[-1].split('.')[0]
                    logger.info(sample_id)
                    
                    criterion = 0.
                    accepted_logl = 50000.
                    logl_reached = False

                    all_traces = []
                    log_w = []
                    N_MIN = 1
                    N_MAX = 11

                    resampled_log_w = {}

                    for iter_n in range(N_MIN, N_MAX):
                        found = False
                        posterior = csis.run(observations={"image": sample}, N=iter_n) 
                        
                        # only evaluate if the set of proposed traces is valid (i.e. does not extract overlapped objects)
                        if len(posterior.log_weights) > 0: 
                            resampling = Empirical(torch.stack([torch.tensor(i) for i in range(len(posterior.log_weights))]), torch.stack(posterior.log_weights))
                            resampled_id = resampling().item()

                            for name, site in posterior.exec_traces[resampled_id].nodes.items():
                                if name == 'image':
                                    criterion = site['fn'].log_prob(site['value'])
                                    if criterion >= accepted_logl: 
                                        preds.append(iter_n)
                                        logger.info(preds)
                                        found = True
                            if found: break
                            else: resampled_log_w[iter_n] = posterior.log_weights[resampled_id]
                    
                    if not found:
                        overall_resampling = Empirical(torch.stack([torch.tensor(k) for k, _ in resampled_log_w.items()]), torch.stack([v for _, v in resampled_log_w.items()]))
                        preds.append(overall_resampling().item())
                        logger.info(preds)

                acc = accuracy_score(preds, [COUNT for _ in range(len(preds))])
                logger.info(f"counting results for scenes with {COUNT} objects: {preds} with acc: {acc}")
        

    elif params['inference_method'] == 'neural_baseline':

        for seed in [1]:
            logger.info(f'seed {seed}')

            torch.manual_seed(seed)

            new_params = {
                "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                "checkpoint_path": os.path.join(main_dir, "synthetic_data", "model_checkpoints"),
                "test_image_save_path": os.path.abspath("neural_baseline"),
                "epoch_to_load": 100
                } 
            
            logger.info(f"loading DME from epoch {new_params['epoch_to_load']}...\n")
            
            #model = DME()
            model = Unet()
            if new_params['device'] == torch.device("cuda:0"): model.load_state_dict(torch.load(f"{new_params['checkpoint_path']}/model_epoch_{new_params['epoch_to_load']}.pth"))
            else: model.load_state_dict(torch.load(f"{new_params['checkpoint_path']}/model_epoch_{new_params['epoch_to_load']}.pth", map_location='cpu'))
            model = model.to(new_params['device'])   
            
            # iterate over testing images for each counting possibility and instantiate a new dataset
            for COUNT in range(1, 11):
                
                # create a folder to aggregate all results for each evaluation image
                #if not os.path.isdir(f"{new_params['test_image_save_path']}"): os.mkdir(f"{new_params['test_image_save_path']}")
                #if not os.path.isdir(f"{new_params['test_image_save_path']}/{COUNT}"): os.mkdir(f"{new_params['test_image_save_path']}/{COUNT}")

                imgs_path = glob.glob(os.path.abspath(f'images/{COUNT}/*.png'))            
                test_dataset = MyDataset(img = imgs_path,
                                        gt = [np.zeros((128, 128)) for _ in range(len(imgs_path))],
                                        params = new_params)
                testloader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = False)
                tester = Tester(model, testloader, new_params)
                counts = tester.test()

                acc = accuracy_score(counts, [COUNT for _ in range(len(counts))])
                logger.info(f'counts for scenes with {COUNT} objects: {counts} with acc: {acc}')

if __name__=="__main__":
    main()