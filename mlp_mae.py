import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os


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
        
        
        image = image.view(-1)

        label = float(item["score"])
        return image, torch.tensor(label, dtype=torch.float)



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

annotation_file = "/Users/weidai/Desktop/dataforsciencefair/brixia/annotation_test.jsonl"
img_dir = "/Users/weidai/Desktop/dataforsciencefair/brixia"

class CovidMLP(nn.Module):
    def __init__(self):
        super(CovidMLP, self).__init__()
        input_size = 224 * 224 * 3 
        self.fc1 = nn.Linear(input_size, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)  

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)  
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CovidMLP()
model.load_state_dict(torch.load("/Users/weidai/Desktop/model/mlp.pth", weights_only=True))
model.eval()

if __name__ == '__main__':
    dataset = COVIDRayDataset(annotation_file, img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    sum_abs_error = 0
    le = 0

    with torch.no_grad():
        for images, scores in dataloader:
            images, scores = images.to(device), scores.to(device)
            outputs = model(images).squeeze()
            print(outputs)
            abs_error = torch.abs(outputs - scores)
            sum_abs_error += abs_error.sum().item()
            le += scores.size(0)

    mean_abs_error = sum_abs_error / le
    print(mean_abs_error)


