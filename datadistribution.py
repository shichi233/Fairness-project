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
annotation_file = "/Users/weidai/Desktop/dataforsciencefair/brixia/annotation_test.jsonl"
img_dir = "/Users/weidai/Desktop/dataforsciencefair/brixia"


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

model = CovidCNN()
model.load_state_dict(torch.load("/Users/weidai/Desktop/model/shichi.pth", weights_only=True))
model.eval()


if __name__ == '__main__':
    dataset = COVIDRayDataset(annotation_file, img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
    s1 = 0
    s2 = 0
    s3 = 0
    s4 = 0
    s5 = 0
    s6 = 0
    s7 = 0
    s8 = 0
    s9 = 0
    s10 = 0
    s11 = 0
    s12 = 0
    s13 = 0
    s14 = 0
    s15 = 0
    s16 = 0
    s17 = 0
    s18 = 0
    s0 = 0

    ss1 = 0
    ss2 = 0
    ss3 = 0
    ss4 = 0
    ss5 = 0
    ss6 = 0
    ss7 = 0
    ss8 = 0
    ss9 = 0
    ss10 = 0
    ss11 = 0
    ss12 = 0
    ss13 = 0
    ss14 = 0
    ss15 = 0
    ss16 = 0
    ss17 = 0
    ss18 = 0
    ss0 = 0

    sss1 = 0
    sss2 = 0
    sss3 = 0
    sss4 = 0
    sss5 = 0
    sss6 = 0
    sss7 = 0
    sss8 = 0
    sss9 = 0
    sss10 = 0
    sss11 = 0
    sss12 = 0
    sss13 = 0
    sss14 = 0
    sss15 = 0
    sss16 = 0
    sss17 = 0
    sss18 = 0
    sss0 = 0

    ssss1 = 0
    ssss2 = 0
    ssss3 = 0
    ssss4 = 0
    ssss5 = 0
    ssss6 = 0
    ssss7 = 0
    ssss8 = 0
    ssss9 = 0
    ssss10 = 0
    ssss11 = 0
    ssss12 = 0
    ssss13 = 0
    ssss14 = 0
    ssss15 = 0
    ssss16 = 0
    ssss17 = 0
    ssss18 = 0
    ssss0 = 0

    sssss1 = 0
    sssss2 = 0
    sssss3 = 0
    sssss4 = 0
    sssss5 = 0
    sssss6 = 0
    sssss7 = 0
    sssss8 = 0
    sssss9 = 0
    sssss10 = 0
    sssss11 = 0
    sssss12 = 0
    sssss13 = 0
    sssss14 = 0
    sssss15 = 0
    sssss16 = 0
    sssss17 = 0
    sssss18 = 0
    sssss0 = 0
    for _, scores, ages in dataloader:
        for a, b in zip(scores, ages):
            if 1 <= b < 4:
                if a == 1:
                    s1 += 1
                elif a == 2:
                    s2 += 1
                elif a == 3:
                    s3 += 1
                elif a == 4:
                    s4 += 1
                elif a == 5:
                    s5 += 1
                elif a == 6:
                    s6 += 1
                elif a == 7:
                    s7 += 1
                elif a == 8:
                    s8 += 1
                elif a == 9:
                    s9 += 1
                elif a == 10:
                    s10 += 1
                elif a == 11:
                    s11 += 1
                elif a == 12:
                    s12 += 1
                elif a == 13:
                    s13 += 1
                elif a == 14:
                    s14 += 1
                elif a == 15:
                    s15 += 1
                elif a == 16:
                    s16 += 1
                elif a == 17:
                    s17 += 1
                elif a == 18:
                    s18 += 1
                elif a == 0:
                    s0 += 1
            elif 4 <= b < 8:
                if a == 1:
                    ss1 += 1
                elif a == 2:
                    ss2 += 1
                elif a == 3:
                    ss3 += 1
                elif a == 4:
                    ss4 += 1
                elif a == 5:
                    ss5 += 1
                elif a == 6:
                    ss6 += 1
                elif a == 7:
                    ss7 += 1
                elif a == 8:
                    ss8 += 1
                elif a == 9:
                    ss9 += 1
                elif a == 10:
                    ss10 += 1
                elif a == 11:
                    ss11 += 1
                elif a == 12:
                    ss12 += 1
                elif a == 13:
                    ss13 += 1
                elif a == 14:
                    ss14 += 1
                elif a == 15:
                    ss15 += 1
                elif a == 16:
                    ss16 += 1
                elif a == 17:
                    ss17 += 1
                elif a == 18:
                    ss18 += 1
                elif a == 0:
                    ss0 += 1
            elif 8 <= b < 12:
                if a == 1:
                    sss1 += 1
                elif a == 2:
                    sss2 += 1
                elif a == 3:
                    sss3 += 1
                elif a == 4:
                    sss4 += 1
                elif a == 5:
                    sss5 += 1
                elif a == 6:
                    sss6 += 1
                elif a == 7:
                    sss7 += 1
                elif a == 8:
                    sss8 += 1
                elif a == 9:
                    sss9 += 1
                elif a == 10:
                    sss10 += 1
                elif a == 11:
                    sss11 += 1
                elif a == 12:
                    sss12 += 1
                elif a == 13:
                    sss13 += 1
                elif a == 14:
                    sss14 += 1
                elif a == 15:
                    sss15 += 1
                elif a == 16:
                    sss16 += 1
                elif a == 17:
                    sss17 += 1
                elif a == 18:
                    sss18 += 1
                elif a == 0:
                    sss0 += 1
            elif 12 <= b < 16:
                if a == 1:
                    ssss1 += 1
                elif a == 2:
                    ssss2 += 1
                elif a == 3:
                    ssss3 += 1
                elif a == 4:
                    ssss4 += 1
                elif a == 5:
                    ssss5 += 1
                elif a == 6:
                    ssss6 += 1
                elif a == 7:
                    ssss7 += 1
                elif a == 8:
                    ssss8 += 1
                elif a == 9:
                    ssss9 += 1
                elif a == 10:
                    ssss10 += 1
                elif a == 11:
                    ssss11 += 1
                elif a == 12:
                    ssss12 += 1
                elif a == 13:
                    ssss13 += 1
                elif a == 14:
                    ssss14 += 1
                elif a == 15:
                    ssss15 += 1
                elif a == 16:
                    ssss16 += 1
                elif a == 17:
                    ssss17 += 1
                elif a == 18:
                    ssss18 += 1
                elif a == 0:
                    ssss0 += 1
            elif 16 <= b <= 18:
                if a == 1:
                    sssss1 += 1
                elif a == 2:
                    sssss2 += 1
                elif a == 3:
                    sssss3 += 1
                elif a == 4:
                    sssss4 += 1
                elif a == 5:
                    sssss5 += 1
                elif a == 6:
                    sssss6 += 1
                elif a == 7:
                    sssss7 += 1
                elif a == 8:
                    sssss8 += 1
                elif a == 9:
                    sssss9 += 1
                elif a == 10:
                    sssss10 += 1
                elif a == 11:
                    sssss11 += 1
                elif a == 12:
                    sssss12 += 1
                elif a == 13:
                    sssss13 += 1
                elif a == 14:
                    sssss14 += 1
                elif a == 15:
                    sssss15 += 1
                elif a == 16:
                    sssss16 += 1
                elif a == 17:
                    sssss17 += 1
                elif a == 18:
                    sssss18 += 1
                elif a == 0:
                    sssss0 += 1
    print(s0)
    print(s1)
    print(s2)
    print(s3)
    print(s4)
    print(s5)
    print(s6)
    print(s7)
    print(s8)
    print(s9)
    print(s10)
    print(s11)
    print(s12)
    print(s13)
    print(s14)
    print(s15)
    print(s16)
    print(s17)
    print(s18)

    print(ss0)
    print(ss1)
    print(ss2)
    print(ss3)
    print(ss4)
    print(ss5)
    print(ss6)
    print(ss7)
    print(ss8)
    print(ss9)
    print(ss10)
    print(ss11)
    print(ss12)
    print(ss13)
    print(ss14)
    print(ss15)
    print(ss16)
    print(ss17)
    print(ss18)

    print(sss0)
    print(sss1)
    print(sss2)
    print(sss3)
    print(sss4)
    print(sss5)
    print(sss6)
    print(sss7)
    print(sss8)
    print(sss9)
    print(sss10)
    print(sss11)
    print(sss12)
    print(sss13)
    print(sss14)
    print(sss15)
    print(sss16)
    print(sss17)
    print(sss18)

    print(ssss0)
    print(ssss1)
    print(ssss2)
    print(ssss3)
    print(ssss4)
    print(ssss5)
    print(ssss6)
    print(ssss7)
    print(ssss8)
    print(ssss9)
    print(ssss10)
    print(ssss11)
    print(ssss12)
    print(ssss13)
    print(ssss14)
    print(ssss15)
    print(ssss16)
    print(ssss17)
    print(ssss18)

    print(sssss0)
    print(sssss1)
    print(sssss2)
    print(sssss3)
    print(sssss4)
    print(sssss5)
    print(sssss6)
    print(sssss7)
    print(sssss8)
    print(sssss9)
    print(sssss10)
    print(sssss11)
    print(sssss12)
    print(sssss13)
    print(sssss14)
    print(sssss15)
    print(sssss16)
    print(sssss17)
    print(sssss18)





