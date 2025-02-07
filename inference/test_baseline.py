import os
import torch
import glob
import json
import time

# add project path to sys to import relative modules
import sys
sys.path.append(os.path.abspath(__file__+'/../../'))

from utils.baseline import Tester, MyDataset, compute_AP
from baseline import Baseline

import logging

def main():

    main_dir = os.path.abspath(__file__+'/../../')
    logfile_name = f'{main_dir}/inference/test_baseline_inspect.log'
    
    logger = logging.getLogger("baseline")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile_name, mode='w')
    logger.addHandler(fh)

    baseline_path = f"{main_dir}/inference/baseline"
    if not os.path.isdir(baseline_path): os.mkdir(baseline_path)
    checkpoint_path = baseline_path + '/checkpoints'
    loss_path = baseline_path + '/loss'
    test_path = baseline_path + '/test_plots'

    params = {
        "num_epochs": 1000,
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "lr": 1e-3,
        "batch_size": 256,
        "max_objects": 17,
        "checkpoint_path": checkpoint_path,
        "loss_path": loss_path,
        "epoch_log_rate": 1,
        "mode": "train",
        "epoch_to_load": 100,
        "slot_dim": 64,
        "test_image_save_path": test_path
    }

    DEVICE = params["device"]
    logger.info(f"device is {DEVICE}")

    seeds = [1]
    for seed in seeds:

        logger.info(f'seed {seed}')
        GUIDE_PATH = f"{params['checkpoint_path']}/model_epoch_{params['epoch_to_load']}.pth"
        #GUIDE_PATH = f"/Users/franciscosilva/Downloads/model_epoch_{params['epoch_to_load']}.pth"
        logger.info(GUIDE_PATH)

        model = Baseline(resolution = (128, 128), num_iterations = 3, hid_dim = params["slot_dim"], stage="train", num_slots=params['max_objects'])
        if DEVICE == 'cuda:0': model.load_state_dict(torch.load(GUIDE_PATH))
        else: model.load_state_dict(torch.load(GUIDE_PATH, map_location='cpu'))
        model.to(DEVICE)

        overall_mAP = {}
        for COUNT in range(5, 6):

            overall_mAP[COUNT] = 0.

            #logger.info(f'\nEVALUATION STARTED FOR SCENES WITH {COUNT} OBJECTS\n')

            img_path = glob.glob(f"/nas-ctm01/homes/fcsilva/ic-slotatt/eval_data/images/{COUNT}/*.png")
            #img_path = glob.glob(f'/Users/franciscosilva/Downloads/eval_data/images/{COUNT}/*.png')
            img_path.sort()
            target_path = glob.glob(f"/nas-ctm01/homes/fcsilva/ic-slotatt/eval_data/metadata/{COUNT}/*.json")
            #target_path = glob.glob(f'/Users/franciscosilva/Downloads/eval_data/metadata/{COUNT}/*.json')
            target_path.sort()

            #logger.info(img_path)

            test_dataset = MyDataset(img = img_path[:5],
                                    target = target_path[:5],
                                    params = params)
            
            #logger.info(test_dataset.__len__())

            testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
            tester = Tester(model, testloader, params)
            
            init_time = time.time()
            preds, targets = tester.test()
            inf_time = time.time() - init_time
            logger.info(inf_time*1000)


            logger.info(preds.shape) # (50, 17, 11)
            logger.info(targets.shape) # (50, 17, 11)

            threshold = [-1., 1., 0.5, 0.25, 0.125, 0.0625]
            ap = {k: 0 for k in threshold}

            for i in range(preds.shape[0]):

                logger.info(f'\n{i}')

                for t in threshold: ap[t] += compute_AP(preds[i], targets[i], t)
            
            mAP = {k: v/(preds.shape[0]) for k, v in ap.items()}
            logger.info(f"COUNT {COUNT}: distance thresholds: \n {threshold[0]} - {threshold[1]} - {threshold[2]} - {threshold[3]} - {threshold[4]} - {threshold[5]}")
            logger.info(f"COUNT {COUNT}: mAP values: {mAP[threshold[0]]} - {mAP[threshold[1]]} - {mAP[threshold[2]]} - {mAP[threshold[3]]} - {mAP[threshold[4]]} - {mAP[threshold[5]]}\n")

            """
            overall_mAP stores mAP values across all COUNTs
            """
            
            for t in threshold: 
                if t in overall_mAP: overall_mAP[t] += mAP[t]
                else: overall_mAP[t] = mAP[t]

        for k, v in overall_mAP.items(): overall_mAP[k] = v/10
        logger.info(f"overall mAP: {overall_mAP[threshold[0]]} - {overall_mAP[threshold[1]]} - {overall_mAP[threshold[2]]} - {overall_mAP[threshold[3]]} - {overall_mAP[threshold[4]]} - {overall_mAP[threshold[5]]}\n")

if __name__ == "__main__":
    main()
