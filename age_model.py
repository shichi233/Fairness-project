import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
from cnn_age_test import test

class BrixiaDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = []

        with open(annotation_file, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                if 16 < record.get("age_group", 0) <= 18:
                    self.data.append(record)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, n):
        item = self.data[n]
        img_filename = item["images"][0]
        img_path = os.path.join(self.img_dir, img_filename)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = float(item["score"])
        return image, torch.tensor(label, dtype=torch.float)


class PneumoniaDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = []

        with open(annotation_file, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line.strip()))

        self.label_map = {"Normal": 0, "BacterialPneumonia": 1, "ViralPneumonia": 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, n):
        item = self.data[n]
        img_filename = item["images"][0]
        img_path = os.path.join(self.img_dir, os.path.basename(img_filename))

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        diagnosis = item["conversations"][1]["value"]
        label = self.label_map[diagnosis]
        return image, torch.tensor(label, dtype=torch.long)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
annotation_file1 = "/Users/weidai/Desktop/dataforsciencefair/brixia/annotation_train.jsonl"
img_dir1 = "/Users/weidai/Desktop/dataforsciencefair/brixia"
annotation_file2 = "/Users/weidai/Desktop/dataset2/annotation_train.jsonl"
img_dir2 = "/Users/weidai/Desktop/dataset2/Coronahack-Chest-XRay-Dataset/train"


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
    dataset1 = BrixiaDataset(annotation_file1, img_dir1, transform=transform)
    dataset2 = PneumoniaDataset(annotation_file2, img_dir2, transform=transform)

    dataloader1 = DataLoader(dataset1, batch_size=128, shuffle=True, num_workers=4)
    dataloader2 = DataLoader(dataset2, batch_size=128, shuffle=True, num_workers=4)
    device = torch.device("mps")

    age3 = CovidCNN().to(device)
    age3.load_state_dict(torch.load("/Users/weidai/Desktop/model/shichi.pth", weights_only=True, map_location=device))
    optimizer = torch.optim.Adam(age3.parameters(), 0.0001)
    criterion_regression = nn.MSELoss()
    criterion_classification = nn.CrossEntropyLoss()

    for epoch in range (2):
        age3.train()
        age3.task = "regression"
        for image, scores in dataloader1:
            image, scores = image.to(device), scores.to(device)
            optimizer.zero_grad()
            output = age3(image)
            loss1 = criterion_regression(output, scores)
            loss1.backward()
            optimizer.step()
            print("loss:" + str(loss1.item()))
        print("\n")
        test(age3)


    save_path = "/Users/weidai/Desktop/model/age5.pth"
    torch.save(age3.state_dict(), save_path)


