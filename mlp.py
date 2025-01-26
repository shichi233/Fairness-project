import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from PIL import Image
#===========================================================================================
class XR_dataset(Dataset):
  
  def __init__ (self, data, label, transform = False):
    self.data = data
    self.label = label
    self.transform = transform

  def getlen(self):
    return len(self.data)

  def getitem(self, idx):
    image = image.open(self.date[idx].convert("RGB"))
    label = self.labels[idx]
    
    if self.transform == True:
        image = self.transform(image)
    return label, image
#===========================================================================================
class MLP(nn.module):
    
    def __init__(self, in_s, hd_s, ot_s):
        super(MLP, self).__init__()#indeed, I reached out to GPT for help for this part
        
#===========================================================================================
def setup():
    #where the hell do I get the image???
#===========================================================================================
def trainmodel():
    pass
#===========================================================================================  
setup()
trainmodel()
