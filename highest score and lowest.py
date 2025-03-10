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
                record = json.loads(line.strip())
                if 12 < record.get("age_group", 0) <= 16:
                    self.data.append(record)
    def __len__ (self):
        return len(self.data)
    def __getitem__(self, n):
        item = self.data[n]
        if "images" not in item:
            raise KeyError(f"Missing 'images' key in dataset entry: {item}")
        img_filename = item["images"][0]

        if not os.path.isabs(img_filename):
            img_path = os.path.join(self.img_dir, img_filename)
        else:
            img_path = img_filename

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

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
model.load_state_dict(torch.load("/Users/weidai/Desktop/model/age4.pth", weights_only=True))
model.eval()


if __name__ == '__main__':
    dataset = COVIDRayDataset(annotation_file, img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=1)
    sum_abs_error = 0
    le = 0
    b = 0
    s = 18
    for i, (image, scores) in enumerate(dataloader):
        output = model(image).squeeze()
        abs_error = torch.abs(output - scores)
        max_idx = abs_error.argmax().item()
        min_idx = abs_error.argmin().item()

        if abs_error[max_idx].item() >= b:
            b = abs_error[max_idx].item()
            bp = dataset.data[i * 128 + max_idx]["images"][0]
            bc = output[max_idx].item()
        if abs_error[min_idx].item() <= s:
            s = abs_error[min_idx].item()
            sp = dataset.data[i * 128 + min_idx]["images"][0]
            sc = output[min_idx].item()

    print("biggest mae is:" + str(b))
    print("biggest image path is: " + bp)
    print("smallest mae is:" + str(s))
    print("smallest image path is:" + sp)
    print("biggest score is " + str(bc))
    print("smallest score is " + str(sc))
