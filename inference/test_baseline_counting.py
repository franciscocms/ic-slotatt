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
        "batch_size": 256,
        "max_objects": 17,
        "checkpoint_path": checkpoint_path,
        "loss_path": loss_path,
        "epoch_log_rate": 1,
        "mode": "train",
        "epoch_to_load": 1,
        "slot_dim": 64,
        "test_image_save_path": test_path
    }

    DEVICE = params["device"]
    logger.info(f"device is {DEVICE}")

    seeds = [1, 2, 3, 4, 5]
    for seed in seeds:

        torch.manual_seed(seed)

        print(f'seed {seed}')
        #GUIDE_PATH = f"{params['checkpoint_path']}/model_epoch_{params['epoch_to_load']}.pth"
        GUIDE_PATH = f"/Users/franciscosilva/Downloads/model_epoch_{params['epoch_to_load']}.pth"

        model = Baseline(resolution = (128, 128), num_iterations = 3, hid_dim = params["slot_dim"], stage="eval", num_slots=params['max_objects'])
        if DEVICE == 'cuda:0': model.load_state_dict(torch.load(GUIDE_PATH))
        else: model.load_state_dict(torch.load(GUIDE_PATH, map_location='cpu'))
        model.to(DEVICE)

        for COUNT in range(1, 11):

            #img_path = glob.glob(f"/nas-ctm01/homes/fcsilva/ic-slotatt/eval_data/images/{COUNT}/*.png")
            img_path = glob.glob(f'/Users/franciscosilva/Downloads/eval_data/images/{COUNT}/*.png')
            img_path.sort()
            #target_path = glob.glob(f"/nas-ctm01/homes/fcsilva/ic-slotatt/eval_data/metadata/{COUNT}/*.json")
            target_path = glob.glob(f'/Users/franciscosilva/Downloads/eval_data/metadata/{COUNT}/*.json')
            target_path.sort()

            test_dataset = MyDataset(img = img_path,
                                    target = target_path,
                                    params = params)

            testloader = torch.utils.data.DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
            tester = Tester(model, testloader, params)
            preds, targets = tester.test()

            #print(preds.shape) # (50, 17, 11)
            #print(targets.shape) # (50, 17, 11)

            acc = 0
            
            for i in range(preds.shape[0]):
                _, _, _, _, _, pred_real_obj = process_preds(preds[i])
                pred_count = torch.sum(torch.round(pred_real_obj))
                if pred_count.item() == COUNT: acc += 1

            acc /= preds.shape[0]  
            print(f"accuracy for count = {COUNT} -> {acc}")
            
if __name__ == "__main__":
    main()
