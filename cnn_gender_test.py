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
        sex = item["sex"]
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float), sex

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
annotation_file = "/Users/weidai/Desktop/dataforsciencefair/brixia/annotation_test.jsonl"
img_dir = "/Users/weidai/Desktop/dataforsciencefair/brixia"


class CovidCNN(nn.Module):
    def __init__(self):
        super(CovidCNN, self).__init__()

        self.cnn = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.cnn(x)
        x = self.sigmoid(x)
        x = x.view(-1)
        x = x * 18
        return x

model = CovidCNN()
model.load_state_dict(torch.load("/Users/weidai/Desktop/model/cnn.pth", weights_only=True))
model.eval()


if __name__ == '__main__':
    dataset = COVIDRayDataset(annotation_file, img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
    male = 0
    female = 0
    le = 0
    for image, scores, sex in dataloader:
        output = model(image).squeeze()
        print(output)
        abs_error = torch.abs(output - scores)
        for i, s in enumerate(sex):
            if s == "M":
                male += abs_error[i].item()
            else:
                female += abs_error[i].item()
            le += 1

    malemae = male / le
    femalemae = female / le
    print(malemae)
    print(femalemae)

