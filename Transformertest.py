import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
import torchvision.models as models
import torch.optim as optim

class CovidRayDataset(Dataset):
    def __init__ (self, annotation_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = []
        with open(annotation_file, "r", encoding = "utf-8") as f:
            for line in f:
                self.data.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, n):
        item = self.data[n]
        img_filename = item["images"][0]
        img_path = os.path.join(self.img_dir,img_filename)
        image = Image.open(img_path).convert("RGB")
        label = float(item["score"])
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
annotation_file = "/Users/weidai/Desktop/dataforsciencefair/brixia/annotation_train.jsonl"
img_dir = "/Users/weidai/Desktop/dataforsciencefair/brixia"


if __name__ == '__main__':
    dataset = COVIDRayDataset(annotation_file, img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

class CovidVit(nn.Module):
    def __init__(self):
        self.vit = models.vit_b_16(pretrained=True)
        self.vit.heads = nn.Linear(self.vit.hidden_im, 1)
        def forward(self, x):
            return self.vit(x).squeeze()

def train_model(model)






