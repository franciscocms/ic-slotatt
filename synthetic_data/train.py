import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy
import os

# add project path to sys to import relative modules
import sys
sys.path.append(os.path.abspath(__file__+'/../../'))

import logging
logfile_name = "training.log"
logging.basicConfig(filename=logfile_name, level=logging.INFO)

class Trainer:
    def __init__(self, model, dataloaders, params): 
        self.trainloader = dataloaders["train"]
        self.validloader = dataloaders["validation"]
        self.params = params
        self.model = model
        self.num_iters = 0
        self.epoch = 0
        self.num_epochs = self.params['num_epochs']
        self.device = self.params['device']
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.optimizer = torch.optim.Adam([p for p in list(self.model.parameters()) if p.requires_grad], lr = self.params['lr'])
        self.criterion_ = nn.MSELoss(reduction = "sum")

        # create dir to save output images
        if not os.path.isdir(self.params['image_save_path']):
            try: os.mkdir(self.params['image_save_path'])
            except: logging.info('unable to create directory to save training plots!')
    
    def _save_checkpoint(self, epoch):
        if not os.path.isdir(self.params['checkpoint_path']):
            try: os.mkdir(self.params['checkpoint_path'])
            except: logging.info('unable to create directory to save training checkpoints!')
        else:
            path = self.params['checkpoint_path']
            torch.save(self.model.state_dict(), path + '/model_epoch_' + str(epoch) + '.pth')
    
    def _save_loss(self, *args):
        step = 1 if self.num_epochs < 10 else int(self.num_epochs/10)
        loss_list = args[0]
        plt.figure()
        for i, loss in enumerate(loss_list):
            plt.plot(range(len(loss_list[loss])), loss_list[loss], label = loss)
        plt.xlabel('epochs')
        plt.xticks(np.arange(0, len(loss_list[loss]), step))
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(self.params['image_save_path'] + '/loss_plot.png', bbox_inches = 'tight')
        plt.close()

    def _train_epoch(self):
        loss = 0.
        num_iters = 0
        self.model.train() 
        for img, dm in self.trainloader:
            img, dm = img.to(self.device), dm.to(self.device)
            dm_hat, count_hat = self.model(img)
            batch_loss = self.criterion_(dm_hat, dm)

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step() 
            
            loss += batch_loss.item()
            num_iters += 1
        return loss/num_iters
    
    def _valid_epoch(self):
        loss = 0.
        num_iters = 0
        self.model.eval()
        with torch.no_grad(): 
            for img, dm in self.validloader:
                img, dm = img.to(self.device), dm.to(self.device)
                dm_hat, count_hat = self.model(img)
                batch_loss = self.criterion_(dm_hat, dm)
                
                loss += batch_loss.item()
                num_iters += 1
        return loss/num_iters

    def train(self):
        since = time.time()  
        train_loss, valid_loss = [], [] 

        for epoch in range(self.num_epochs):                  
            self.epoch = epoch
            if epoch % self.params["epoch_log_rate"] == 0:
                logging.info("Epoch {}/{}".format(epoch, self.num_epochs - 1))

            epoch_train_loss = self._train_epoch()
            train_loss.append(epoch_train_loss) 
            epoch_valid_loss = self._valid_epoch()
            valid_loss.append(epoch_valid_loss)      

            loss_dic = {"train_loss": train_loss,
                        "valid_loss": valid_loss}
            
            if epoch % self.params["epoch_log_rate"] == 0 or epoch == self.num_epochs-1:
                logging.info("... train_loss: {:.3f}" .format(train_loss[-1]))
                logging.info("... valid_loss: {:.3f}" .format(valid_loss[-1]))
                self._save_checkpoint(epoch)
                self._save_loss(loss_dic)
        
        time_elapsed = time.time() - since
        logging.info('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))