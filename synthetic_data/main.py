import glob
import os

import torch
import numpy as np
from sklearn.metrics import accuracy_score

# add project path to sys to import relative modules
import sys
sys.path.append(os.path.abspath(__file__+'/../../'))

from network import Model, Unet
from dataset import MyDataset
from train import Trainer
from test import Tester

import logging

params = {
  "num_epochs": 1000,
  "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
  "lr": 1e-3,
  "batch_size": 256,
  "checkpoint_path": os.path.abspath("model_checkpoints"),
  "image_save_path": os.path.abspath("train_plots"),
  "test_image_save_path": os.path.abspath("test_plots"),
  "epoch_log_rate": 1,
  "mode": "test",
  "epoch_to_load": 400
}

if params["mode"] == "train": logfile_name = "training.log"
elif params["mode"] == "test": logfile_name = "testing.log"

logging.basicConfig(filename=logfile_name, level=logging.INFO)

if __name__ == "__main__":
    
    model = Unet()
    model = model.to(params['device'])
    logging.info([p.requires_grad for p in list(model.parameters())])

    logging.info(params)


    """ train and validation data must be generated using the 'synthetic-data/generate.py' script"""
    img_path = '' # path for dataset images
    density_maps_path = '' #path for dataset density maps

    all_img = glob.glob(f"{img_path}/*.png")
    all_img.sort()
    all_dm = glob.glob(f"{density_maps_path}/*.npy")
    all_dm.sort()
    
    train_dataset = MyDataset(img = all_img[:25000],
                              gt = all_dm[:25000],
                              params = params)
    validation_dataset = MyDataset(img = all_img[25000:],
                                   gt = all_dm[25000:],
                                   params = params)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = params['batch_size'], shuffle = True)
    validloader = torch.utils.data.DataLoader(validation_dataset, batch_size = params['batch_size'], shuffle = True)
    trainer = Trainer(model = model, 
                      dataloaders = {"train": trainloader, "validation": validloader},
                      params = params)
    
    if params["mode"] == "train":
      trainer.train()
    elif params["mode"] == "test":
      model = Unet()
      model.load_state_dict(torch.load(f"{params['checkpoint_path']}/model_epoch_{params['epoch_to_load']}.pth"))
      model = model.to(params['device'])    
      test_dataset = MyDataset(img = all_img[29990:],
                               gt = all_dm[29990:],
                               params = params)
      testloader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = False)
      tester = Tester(model, testloader, params)
      preds = tester.test()
      gt = []
      for dm in test_dataset.gt: gt.append(torch.round(torch.sum(torch.from_numpy(np.load(dm)))).item())
      
      logging.info(f"i.i.d accuracy score: {accuracy_score(preds, gt)}")

    else: raise ValueError(f"Unknown mode {params['mode']}")
    