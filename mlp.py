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

class XR_dataset(dataset):
  
  def __init__ (self, data, label, transform = False):
    self.data = data
    self.label = label
    self.transform = transform

  def getlen(self):
    return len(self.data)

class mlp():
    pass

def trainmodel():
    pass

#饿了先吃个饭
#等会再写
