import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
from collections import defaultdict


class BrixiaDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = [json.loads(line.strip()) for line in open(annotation_file, "r", encoding="utf-8")]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, n):
        item = self.data[n]
        img_path = os.path.join(self.img_dir, item["images"][0])

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(float(item["score"]), dtype=torch.float), item["age_group"]


class PneumoniaDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = [json.loads(line.strip()) for line in open(annotation_file, "r", encoding="utf-8")]
        self.label_map = {"Normal": 0, "BacterialPneumonia": 1, "ViralPneumonia": 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, n):
        item = self.data[n]
        img_path = os.path.join(self.img_dir, os.path.basename(item["images"][0]))

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(self.label_map[item["conversations"][1]["value"]], dtype=torch.long)


class CovidCNN(nn.Module):
    def __init__(self, task="regression"):
        super(CovidCNN, self).__init__()
        self.cnn = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()
        self.fc1 = nn.Linear(num_features, 1)
        self.fc2 = nn.Linear(num_features, 3)
        self.sigmoid = nn.Sigmoid()
        self.task = task

    def forward(self, x):
        features = self.cnn(x)
        if self.task == "regression":
            return self.sigmoid(self.fc1(features)).view(-1) * 18
        return self.fc2(features)


def load_models(model_dir):
    age_groups = ["4-8", "8-12", "12-16", "16-18"]
    models = {age: CovidCNN() for age in age_groups}

    for i, age_group in enumerate(age_groups, start=2):
        model_path = os.path.join(model_dir, f"age{i}.pth")
        if os.path.exists(model_path):
            models[age_group].load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            models[age_group].eval()
        else:
            print(f"Warning: Model file {model_path} not found!")

    return models


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    annotation_file = "/Users/weidai/Desktop/dataforsciencefair/brixia/annotation_test.jsonl"
    img_dir = "/Users/weidai/Desktop/dataforsciencefair/brixia"
    model_dir = "/Users/weidai/Desktop/model"

    models = load_models(model_dir)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = BrixiaDataset(annotation_file, img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    sum_abs_error, total_samples = 0, 0
    age_mae = defaultdict(lambda: {"sum_error": 0, "count": 0})

    for images, scores, age_groups in dataloader:
        for i in range(scores.size(0)):
            age_group = age_groups[i]
            if age_group in models:
                output = models[age_group](images[i].unsqueeze(0)).squeeze().item()
                abs_error = abs(output - scores[i].item())
                sum_abs_error += abs_error
                total_samples += 1
                age_mae[age_group]["sum_error"] += abs_error
                age_mae[age_group]["count"] += 1

    if total_samples > 0:
        print(f"Overall MAE: {sum_abs_error / total_samples:.4f}")
    else:
        print("No valid samples processed.")

    for age_group, values in age_mae.items():
        if values["count"] > 0:
            print(f"Age Group {age_group}: MAE = {values['sum_error'] / values['count']:.4f}")
