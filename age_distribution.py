import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
class COVIDRayDataset(Dataset):
    def __init__ (self, annotation_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = []
        with open(annotation_file, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line.strip()))
    def __len__ (self):
        return len(self.data)
    def __getitem__(self, n):
        item = self.data[n]
        img_filename = item["images"][0]
        img_path = os.path.join(self.img_dir, img_filename)
        image = Image.open(img_path).convert("RGB")
        label = float(item["score"])
        age = item["age_group"]
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float), age

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


class CovidCNN(nn.Module):
    def __init__(self, task = "regression"):
        super(CovidCNN, self).__init__()

        self.cnn = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()
        self.fc1 = nn.Linear(num_features, 1)
        self.fc2 = nn.Linear(num_features, 3)
        self.sigmoid = nn.Sigmoid()
        self.task = task

    def forward(self,x):
        features = self.cnn(x)
        if self.task == "regression":
            out = self.fc1(features)
            out = self.sigmoid(out).view(-1) * 18
        elif self.task == "classification":
            out = self.fc2(features)
        return out




if __name__ == '__main__':
    annotation_file = "/Users/weidai/Desktop/dataforsciencefair/brixia/annotation_test.jsonl"
    img_dir = "/Users/weidai/Desktop/dataforsciencefair/brixia"


    dataset = COVIDRayDataset(annotation_file, img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
    sum1 = 0
    sum2 = 0
    sum3 = 0
    sum4 = 0
    sum5 = 0
    for image, scores, age in dataloader:
        for i, a in enumerate(age):
            if 1 <= a < 4:
                sum1 += 1
            elif 4 <= a < 8:
                sum2 += 1
            elif 8 <= a < 12:
                sum3 += 1
            elif 12 <= a < 16:
                sum4 += 1
            elif 16 <= a <= 18:
                sum5 += 1
    print()


    model = CovidCNN()
    model.load_state_dict(torch.load("/Users/weidai/Desktop/model/age2.pth", weights_only=True))