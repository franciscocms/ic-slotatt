import torch
from torch import nn

import os

# add project path to sys to import relative modules
import sys
sys.path.append(os.path.abspath(__file__+'/../../'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class unet_encoder(nn.Module):
    def __init__(self):
      super(unet_encoder, self).__init__()
      
      self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, padding = 1)
      self.conv2 = nn.Conv2d(16, 16, kernel_size = 3, padding = 1)
      self.conv3 = nn.Conv2d(16, 32, kernel_size = 3, padding = 1)
      self.conv4 = nn.Conv2d(32, 32, kernel_size = 3, padding = 1)
      self.conv5 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
      self.conv6 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)
      self.conv7 = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
      self.conv8 = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
      self.conv9 = nn.Conv2d(128, 256, kernel_size = 3, padding = 1)
      self.conv10 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
      
      self.relu = nn.ReLU(inplace=True)
      self.mp = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices = False)
                  
    def forward(self, x):
      x = self.relu(self.conv1(x))             
      c2 = self.relu(self.conv2(x))

      x = self.mp(c2)      
      x = self.relu(self.conv3(x))      
      c4 = self.relu(self.conv4(x))
  
      x = self.mp(c4)  
      x = self.relu(self.conv5(x))
      c6 = self.relu(self.conv6(x))

      x = self.mp(c6)  
      x = self.relu(self.conv7(x))
      c8 = self.relu(self.conv8(x))

      x = self.mp(c8)  
      x = self.relu(self.conv9(x))
      x = self.relu(self.conv10(x))
                    
      return x, c2, c4, c6, c8

class unet_decoder(nn.Module):
  def __init__(self):
    super(unet_decoder, self).__init__()

    self.tconv1 = nn.ConvTranspose2d(256, 128, kernel_size = 2, stride = 2)
    self.tconv2 = nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2)
    self.tconv3 = nn.ConvTranspose2d(64, 32, kernel_size = 2, stride = 2)
    self.tconv4 = nn.ConvTranspose2d(32, 16, kernel_size = 2, stride = 2)
    
    self.conv11 = nn.Conv2d(256, 128, kernel_size = 3, padding = 1)
    self.conv12 = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
    self.conv13 = nn.Conv2d(128, 64, kernel_size = 3, padding = 1)
    self.conv14 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)
    self.conv15 = nn.Conv2d(64, 32, kernel_size = 3, padding = 1)
    self.conv16 = nn.Conv2d(32, 32, kernel_size = 3, padding = 1)
    self.conv17 = nn.Conv2d(32, 16, kernel_size = 3, padding = 1)
    self.conv18 = nn.Conv2d(16, 16, kernel_size = 3, padding = 1)
    self.conv19 = nn.Conv2d(16, 1, kernel_size = 1, padding = 0)
            
    self.relu = nn.ReLU(inplace=True)


    
  def forward(self, x, c2, c4, c6, c8):        
    x = self.relu(self.tconv1(x))
    x = torch.cat((c8, x), dim = 1)
    x = self.relu(self.conv11(x))         
    x = self.relu(self.conv12(x))
    
    x = self.relu(self.tconv2(x))
    x = torch.cat((c6, x), dim = 1)
    x = self.relu(self.conv13(x))         
    x = self.relu(self.conv14(x))
    
    x = self.relu(self.tconv3(x))
    x = torch.cat((c4, x), dim = 1)
    x = self.relu(self.conv15(x))         
    x = self.relu(self.conv16(x))
    
    x = self.relu(self.tconv4(x))
    x = torch.cat((c2, x), dim = 1)
    x = self.relu(self.conv17(x))         
    x = self.relu(self.conv18(x))
    
    x = self.conv19(x)
    x = torch.abs(x)
    count = torch.sum(x)
    
    return x, count

class Unet(nn.Module):
  def __init__(self):
    super(Unet, self).__init__()
    self.encoder = unet_encoder()
    self.decoder = unet_decoder()
  
  def forward(self, x):
    x, c2, c4, c6, c8 = self.encoder(x)
    x = self.decoder(x, c2, c4, c6, c8)
    return x