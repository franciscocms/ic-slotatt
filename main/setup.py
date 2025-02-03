import torch
import logging
import os

main_dir = os.path.abspath(__file__+'/../../')

# jobID 70 holds results for ICSA scenes with upgraded code
# jobID 71 holds results for ICSA scenes with dependencies shape -> color in the generative program


# jobID 87 holds results for 4 objects CLEVR scenes
# jobID 88 holds results for 6< objects CLEVR scenes 
# jobID 90 is training for 10< objects CLEVR scenes
# jobID 91 will train for 6< objects CLEVR scenes with new loss

params = {
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "dataset": "2Dobjects", # 'clevr' or '2Dobjects'
    "resolution": (128, 128),
    "num_slots": 6,
    "max_objects": 6,
    "guide_step" : 9950,
    "print_distributions": False,
    "print_importance_sampling": False,
    "bernoulli_inf_reduction": 'none',
    "num_inference_samples": 100,
    "N_proposal" : "normal", # mixture
    "loc_proposal" : "wo_net",
    "loc_proposal_std": 0.05, 
    "loc_proposal_k" : 5,
    "prior_stddevs" : 0.05,
    "N_prior_std" : 0.1,
    "pos_from_attn" : "attn-masks", # "attn-masks" if computing locations from slot attention masks (alternative: "dme" from the estimated density maps)
    "training_from_scratch" : True,
    "batch_size" : 64, # 64
    "training_iters": 10000, # 10k
    "step_size": 50,
    "running_type": "eval", # train, debug, eval, inspect
    "slot_dim" : 64,
    "infer_background": False,
    "slotatt_recurrence": True,
    "softmax_temperature": 1.0,
    "strided_convs": True,
    "check_attn": True,
    "jobID": 70, # 69 holds the results for ICSA trained on '2Dobjects' and 87 for only 4 CLEVR objects
    "mesh_iters": 4,
    "mesh_lr": 5, # 76: 3, 77: 5
    "logprob_coeff": 1.,
    "slot_pos_learned_init": False,
    "perm_inv_loss": True,
    "inference_method": "importance_sampling_only", # score_resample, rejuvenation_ft, importance_sampling_only, neural_baseline
    "proposals": "data_driven", # "data_driven", "prior"
    "plots_dir": f'{main_dir}/plots'
}

# set job split settings for inference
JOB_SPLIT = {
            'id': 1,
            'total': 4
            }

if params["dataset"] == "clevr": params["lr"] = 4-4
elif params["dataset"] == "2Dobjects": params["lr"] = 1e-3
else: raise ValueError(f"Dataset error: dataset named {params['dataset']} not found!")

if not os.path.isdir(params['plots_dir']): os.mkdir(params['plots_dir'])

params["no_slots"] = "w_background" if params["infer_background"] else "wo_background"

if params["check_attn"]: params["check_attn_folder"] = f"{main_dir}/check_imgs_{params['jobID']}"
