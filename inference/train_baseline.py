import os
import torch
import glob

# add project path to sys to import relative modules
import sys
sys.path.append(os.path.abspath(__file__+'/../../'))

from utils.baseline import Trainer, MyDataset
from baseline import Baseline

import logging

def main():

    main_dir = os.path.abspath(__file__+'/../../')
    logfile_name = f'{main_dir}/inference/baseline.log'
    
    logger = logging.getLogger("baseline")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile_name, mode='w')
    logger.addHandler(fh)

    baseline_path = f"{main_dir}/inference/baseline"
    for p in [baseline_path]: 
        if not os.path.isdir(p): os.mkdir(p)
    checkpoint_path = baseline_path + '/checkpoints'
    loss_path = baseline_path + '/loss'
    check_attn_masks = baseline_path + '/train_attn_masks'
    for p in [checkpoint_path, loss_path, check_attn_masks]: 
        if not os.path.isdir(p): os.mkdir(p)

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
        "epoch_to_load": 400,
        "slot_dim": 64,
    }

    DEVICE = params["device"]
    logger.info(f"device is {DEVICE}")

    model = Baseline(resolution = (128, 128), num_iterations = 3, hid_dim = params["slot_dim"], stage="train", num_slots=params['max_objects'])
    model.to(DEVICE)    
 
    img_path = glob.glob('') # path to synthetic dataset (.png images)
    img_path.sort()
    target_path = glob.glob('') # path to synthetic dataset (.json metadata)
    target_path.sort()

    train_dataset = MyDataset(img = img_path[:27500],
                            target = target_path[:27500],
                            params = params)
    validation_dataset = MyDataset(img = img_path[27500:],
                                target = target_path[27500:],
                                params = params)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    validloader = torch.utils.data.DataLoader(validation_dataset, batch_size=params['batch_size'], shuffle=True)
    trainer = Trainer(model = model, 
                    dataloaders = {"train": trainloader, "validation": validloader},
                    params = params)

    trainer.train()

    logger.info("\ntraining ended...")

if __name__ == "__main__":
    main()
