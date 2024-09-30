import os
import torch
import glob
import json

# add project path to sys to import relative modules
import sys
sys.path.append(os.path.abspath(__file__+'/../../'))

from utils.baseline import Tester, MyDataset, process_preds
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
        "batch_size": 1,
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
        #GUIDE_PATH = f"{params['checkpoint_path']}/model_epoch_{params['epoch_to_load']}.pth"
        GUIDE_PATH = f"/Users/franciscosilva/Downloads/model_epoch_{params['epoch_to_load']}.pth"
        logger.info(GUIDE_PATH)

        model = Baseline(resolution = (128, 128), num_iterations = 3, hid_dim = params["slot_dim"], stage="eval", num_slots=params['max_objects'])
        if DEVICE == 'cuda:0': model.load_state_dict(torch.load(GUIDE_PATH))
        else: model.load_state_dict(torch.load(GUIDE_PATH, map_location='cpu'))
        model.to(DEVICE)

        img_path = ["synthetic_data/ood_samples/00000.png"]
        target_path = []

        logger.info(img_path)

        test_dataset = MyDataset(img = img_path,
                                target = target_path,
                                params = params)
        
        logger.info(test_dataset.__len__())

        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
        tester = Tester(model, testloader, params)
        preds = tester.test_ood()

        shape, size, color, locx, locy, pred_real_obj = process_preds(preds[0])
        print(shape)
        print(size)
        print(color)
        print(locx)
        print(locy)
        print(pred_real_obj)
            
            
if __name__ == "__main__":
    main()
