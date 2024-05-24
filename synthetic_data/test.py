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
    
    def _save_outputs(self, img, gt_dm, count_hat, predicted_dm, name):       
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(15, 15), sharex=True, sharey=True)                                          
        ax1.set_title('scene')
        ax1.axis('off')
        ax1.imshow(img)
        ax2.set_title('GT density map')
        ax2.axis('off')
        ax2.imshow(gt_dm)
        ax3.set_title(f"predicted density map ({count_hat})")
        ax3.axis('off')
        ax3.imshow(predicted_dm)
        ax4.set_title("overlay")
        ax4.axis('off')
        ax4.imshow(img)
        ax4.imshow(predicted_dm, alpha=0.6)
        plt.savefig(self.params['test_image_save_path'] + '/' + name + '.png', bbox_inches = 'tight')
        plt.close()

    
    def _test_batch(self, img, dm):
        img, dm = img.to(self.device), dm.to(self.device)
        dm_hat, count_hat = self.model(img)
        return dm_hat, count_hat

    def test(self):
        self.model.eval()
        counts = []

        with torch.no_grad(): 
            for idx, (img, dm) in enumerate(self.testloader):
                
                if dm is None: dm = torch.zeros(img.shape[-2:])
                
                # plt.imshow(img.squeeze().permute(1, 2, 0).cpu().numpy())
                # plt.savefig(self.params['test_image_save_path'] + '/' + str(idx) + '.png', bbox_inches = 'tight')
                # plt.close()
                
                dm_hat, count_hat = self._test_batch(img, dm)
                counts.append(torch.round(count_hat).item())
                # self._save_outputs(img.squeeze().permute(1, 2, 0).cpu().numpy(),
                #                    dm.squeeze().cpu().numpy(),
                #                    count_hat,
                #                    dm_hat.squeeze().cpu().numpy(), 
                #                    str(idx))
        return counts