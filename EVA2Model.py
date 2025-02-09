import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
import timm

class COVIDRayDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = []
        with open(annotation_file, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, n):
        item = self.data[n]
        if "images" not in item:
            raise KeyError(f"Missing 'images' key in dataset entry: {item}")
        img_filename = item["images"][0]

        img_path = os.path.join(self.img_dir, img_filename) if not os.path.isabs(img_filename) else img_filename

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = float(item["score"])
        return image, torch.tensor(label, dtype=torch.float)


class EVA2Model(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.model = timm.create_model("eva2_base_patch16_224", pretrained=True, num_classes=num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        x = x * 18
        return x


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

annotation_file = "/Users/weidai/Desktop/dataforsciencefair/brixia/annotation_test.jsonl"
img_dir = "/Users/weidai/Desktop/dataforsciencefair/brixia"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EVA2Model().to(device)

# Load pretrained checkpoint
checkpoint = torch.load("/Users/weidai/Desktop/model/eva2.pth", map_location=device)
model.load_state_dict(checkpoint)

if __name__ == '__main__':
    dataset = COVIDRayDataset(annotation_file, img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    sum_abs_error = 0
    total_samples = 0

    with torch.no_grad():
        for images, scores in dataloader:
            images, scores = images.to(device), scores.to(device)
            outputs = model(images).squeeze()
            print(outputs)
            abs_error = torch.abs(outputs - scores)
            sum_abs_error += abs_error.sum().item()
            total_samples += scores.size(0)

    mean_abs_error = sum_abs_error / total_samples
    print(f"Mean Absolute Error: {mean_abs_error:.4f}")
