import torch
import logging
import os

main_dir = os.path.abspath(__file__+'/../../')

params = {
    "dataset": "2Dobjects", # 'clevr' or '2Dobjects'
    "mixture_components": 10,
    "guide_step" : 10000,
    "print_distributions": False,
    "print_importance_sampling": False,
    "bernoulli_inf_reduction": 'none',
    "num_inference_samples": 50,
    "N_proposal" : "normal", # mixture
    "loc_proposal" : "wo_net",
    "loc_proposal_std": 0.05, 
    "loc_proposal_k" : 5,
    "prior_stddevs" : 0.05,
    "N_prior_std" : 0.1,
    "pos_from_attn" : "attn-masks", # "attn-masks" if computing locations from slot attention masks (alternative: "dme" from the estimated density maps)
    "training_from_scratch" : True,
    "lr" : 1e-3, 
    "batch_size" : 1, # 256
    "training_iters": 10000, # 10k
    "running_type": "train", # train, debug, eval, inspect
    "slot_dim" : 64,
    "infer_background": False,
    "slotatt_recurrence": True,
    "softmax_temperature": 1.0,
    "strided_convs": True,
    "check_attn": True,
    "jobID": 70, # 69 holds the results for ICSA trained on '2Dobjects'
    "mesh_iters": 4,
    "mesh_lr": 3,
    "logprob_coeff": 1.,
    "slot_pos_learned_init": False,
    "perm_inv_loss": True,
    "inference_method": "score_resample", # score_resample, rejuvenation_ft, importance_sampling_only, neural_baseline
    "proposals": "data_driven", # "data_driven", "prior"
    "plots_dir": f'{main_dir}/plots'
}

if not os.path.isdir(params['plots_dir']): os.mkdir(params['plots_dir'])


params["no_slots"] = "w_background" if params["infer_background"] else "wo_background"

if params['running_type'] == 'train': params["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else: params['device'] = torch.device('cpu')

if params["check_attn"]: params["check_attn_folder"] = f"{main_dir}/check_imgs_{params['jobID']}"
#if params["running_type"] == "debug": params["logfile_name"] = "debug.log"
#elif params["running_type"] == "inspect": 
#    params["logfile_name"] = "inspect.log"
#    params["inspect_img_path"] = f"inspect/inspect-imgs-{params['jobID']}-{params['guide_step']}"
#elif params["running_type"] == "train": params["logfile_name"] = f"log-{params['jobID']}.log"
#elif params["running_type"] == "eval": params["logfile_name"] = "eval.log"

#else: raise ValueError(f"Unknown running type {params['running_type']}")