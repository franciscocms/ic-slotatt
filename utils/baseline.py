import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from PIL import Image
import copy
import os
import json

# add project path to sys to import relative modules
import sys
sys.path.append(os.path.abspath(__file__+'/../../'))

import logging
logger = logging.getLogger("baseline")

def hungarian_loss(pred, target, loss_fn=F.smooth_l1_loss):
    
    """
    adapted from 'https://github.com/davzha/MESH/blob/main/losses.py'
    """
    
    pdist = loss_fn(
        pred.unsqueeze(1).expand(-1, target.size(1), -1, -1), 
        target.unsqueeze(2).expand(-1, -1, pred.size(1), -1),
        reduction='none').mean(3)

    pdist_ = pdist.detach().cpu().numpy()

    indices = np.array([linear_sum_assignment(p) for p in pdist_])

    indices_ = indices.shape[2] * indices[:, 0] + indices[:, 1]
    losses = torch.gather(pdist.flatten(1,2), 1, torch.from_numpy(indices_).to(device=pdist.device))
    total_loss = torch.mean(losses.sum(1))

    return total_loss, dict(indices=indices)


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

        # create dir to save output images
        if not os.path.isdir(self.params['loss_path']):
            try: os.mkdir(self.params['loss_path'])
            except: logger.info('unable to create directory to save training plots!')
    
    def _save_checkpoint(self, epoch):
        if not os.path.isdir(self.params['checkpoint_path']):
            try: os.mkdir(self.params['checkpoint_path'])
            except: logger.info('unable to create directory to save training checkpoints!')
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
        plt.savefig(self.params['loss_path'] + '/loss_plot.png', bbox_inches = 'tight')
        plt.close()

    def _train_epoch(self):
        loss = 0.
        num_iters = 0
        self.model.train() 
        for img, target in self.trainloader:
            img, target = img.to(self.device), target.to(self.device)
            preds = self.model(img)
            batch_loss, _ = hungarian_loss(preds, target)

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
            for img, target in self.validloader:
                img, target = img.to(self.device), target.to(self.device)
                preds = self.model(img)
                batch_loss, _ = hungarian_loss(preds, target)
                
                loss += batch_loss.item()
                num_iters += 1
        return loss/num_iters

    def train(self):
        since = time.time()  
        train_loss, valid_loss = [], [] 

        for epoch in range(self.num_epochs):                  
            self.epoch = epoch
            if epoch % self.params["epoch_log_rate"] == 0:
                logger.info("Epoch {}/{}".format(epoch, self.num_epochs - 1))

            epoch_train_loss = self._train_epoch()
            train_loss.append(epoch_train_loss) 
            epoch_valid_loss = self._valid_epoch()
            valid_loss.append(epoch_valid_loss)      

            loss_dic = {"train_loss": train_loss,
                        "valid_loss": valid_loss}
            
            if epoch % self.params["epoch_log_rate"] == 0 or epoch == self.num_epochs-1:
                logger.info("... train_loss: {:.3f}" .format(train_loss[-1]))
                logger.info("... valid_loss: {:.3f}" .format(valid_loss[-1]))
                self._save_checkpoint(epoch)
                self._save_loss(loss_dic)
        
        time_elapsed = time.time() - since
        logger.info('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        

class Tester:
    def __init__(self, model, testloader, params): 
        self.testloader = testloader
        self.params = params
        self.model = model
        self.device = self.params['device']

        # create dir to save output images
        if not os.path.isdir(self.params['test_image_save_path']):
            try: os.mkdir(self.params['test_image_save_path'])
            except: raise ValueError(f"unable to create directory {self.params['test_image_save_path']} to save testing plots!")

    
    def _test_batch(self, img):
        img = img.to(self.device)
        return self.model(img)
    
    def test(self):
        self.model.eval()
        with torch.no_grad(): 
            for idx, (img, target) in enumerate(self.testloader):                                
                # plt.imshow(img.squeeze().permute(1, 2, 0).cpu().numpy())
                # plt.savefig(self.params['test_image_save_path'] + '/' + str(idx) + '.png', bbox_inches = 'tight')
                # plt.close()
                preds = self._test_batch(img)
                return preds, target


class MyDataset():
  def __init__(self, img, target, params):
    self.img = img
    self.target = target
    self.params = params
    self.img_transform = transforms.Compose([transforms.ToTensor()])

    self.shape_vals = {'ball': 0, 'square': 1}
    self.size_vals = {'small': 0 , 'medium': 1, 'large': 2}
    self.color_vals = {'red': 0, 'green': 1, 'blue': 2}

    self.target_dim = len(self.shape_vals) + len(self.size_vals) + len(self.color_vals) + 3 # 3 = 2 (x- y-coords) + 1 (real/padded object)
  
  def __getitem__(self, index):
    img = self.img_transform(Image.open(self.img[index]))
    target_dict = json.load(open(self.target[index]))
    
    target = torch.zeros(self.params['max_objects'], self.target_dim)

    for n in range(int(target_dict['scene_attr']['N'])):
        target[n, :2] = F.one_hot(torch.tensor(self.shape_vals[target_dict['scene_attr'][f'object_{n}']['shape']]), len(self.shape_vals))
        target[n, 2:5] = F.one_hot(torch.tensor(self.size_vals[target_dict['scene_attr'][f'object_{n}']['size']]), len(self.size_vals))
        target[n, 5:8] = F.one_hot(torch.tensor(self.color_vals[target_dict['scene_attr'][f'object_{n}']['color']]), len(self.color_vals))
        target[n, 8] = target_dict['scene_attr'][f'object_{n}']['initLocX']
        target[n, 9] = target_dict['scene_attr'][f'object_{n}']['initLocY']
        target[n, 10] = torch.tensor(1.)
    
    n = int(target_dict['scene_attr']['N'])
    target[n:self.params['max_objects'], :] = torch.zeros(self.params['max_objects']-n, self.target_dim)

    return img, target # target is shape (b_s, 17, 11)
  
  def __len__(self):
    return len(self.img)

def process_preds(preds):
    # preds have shape (max_objects, n_features)

    shape = torch.argmax(preds[:, :2], dim=-1)
    size = torch.argmax(preds[:, 2:5], dim=-1)
    color = torch.argmax(preds[:, 5:8], dim=-1)
    locx, locy = preds[:, 8], preds[:, 9]
    real_obj = preds[:, 10]
    return shape, size, color, locx, locy, real_obj

def distance(loc1, loc2):
    return torch.sqrt(torch.square(torch.abs(loc1[0]-loc2[0])) + torch.square(torch.abs(loc1[1]-loc2[1])))

def compute_AP(preds, targets, threshold_dist):

    """
    adapted from 'https://github.com/google-research/google-research/blob/master/slot_attention/utils.py'
    """

    # preds have shape (max_objects, n_features)
    # targets have shape (max_objects, n_features)

    shape, size, color, locx, locy, pred_real_obj = process_preds(preds)
    # logger.info(shape)
    # logger.info(size)
    # logger.info(color)
    # logger.info(locx)
    # logger.info(locy)
    # logger.info(pred_real_obj)
    target_shape, target_size, target_color, target_locx, target_locy, target_real_obj = process_preds(targets)

    # shape, size, ...  has shape (17)

    max_objects = shape.shape[0]
    
    tp = np.zeros(1)
    fp = np.zeros(1)
    
    found_objects = []
    for o in range(max_objects):
        if torch.round(pred_real_obj[o]):

            #logger.info(f'{o} - {pred_real_obj[o]}')
            
            #logger.info(shape[o])
            #logger.info(target_shape[0])

            #logger.info(f'{locx[o]} - {locy[o]}')
            #logger.info(f'{target_locx[o]} - {target_locy[o]}')

            # tries to find a match between predicted object 'o' and any target object
            # returns the best distance match (if more than one occurs)

            found = False
            found_idx = -1 
            best_distance = 1000
            
            for j in range(max_objects):
                if target_real_obj[j]:
                    if [shape[o], size[o], color[o]] == [target_shape[j], target_size[j], target_color[j]]: 
                        dist = distance((locx[o], locy[o]), (target_locx[j], target_locy[j]))
                        if dist < best_distance and j not in found_objects:
                            #logger.info(f'found at best distance {dist}')
                            found = True
                            best_distance = dist
                            found_idx = j # stores the best match between an object and all possible targets
            
            if found:
                if distance((locx[o], locy[o]), (target_locx[found_idx], target_locy[found_idx])) <= threshold_dist or threshold_dist == -1:
                    found_objects.append(found_idx)
                    #logger.info('found match below distance threshold!')
                    tp += 1
            else: fp += 1

            #logger.info(found_objects)
    
    precision = tp / (tp+fp)
    recall = tp / np.sum(np.asarray(target_real_obj.cpu()))

    #logger.info(f'precision: {precision}')
    #logger.info(f'recall: {recall}')

    recall = recall.tolist()
    precision = precision.tolist()
    recall = [0] + recall + [1]
    precision = [0] + precision + [0]

    for i in range(len(precision) - 1, -0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    indices_recall = [
        i for i in range(len(recall) - 1) if recall[1:][i] != recall[:-1][i]
    ]

    average_precision = 0.
    for i in indices_recall:
        average_precision += precision[i + 1] * (recall[i + 1] - recall[i])

    #logger.info(f'ap: {average_precision}')
    return average_precision
            



    
        




