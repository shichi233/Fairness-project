import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
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

    def __getitem__(self, idx):
        item = self.data[idx]
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

annotation_file = "C:/Users/frank/Documents/science fair 2025/brixia/annotation_test.jsonl"
img_dir = "C:/Users/frank/Documents/science fair 2025/brixia"

class CovidEfficientNet(nn.Module):
    def __init__(self):
        super(CovidEfficientNet, self).__init__()

        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.efficientnet.classifier = nn.Linear(self.efficientnet.classifier[1].in_features, 1)

    def forward(self, x):
        x = self.efficientnet(x)
        x = x.view(-1)  
        x = x * 18  
        return x


if __name__ == '__main__':
    dataset = COVIDRayDataset(annotation_file, img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CovidEfficientNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.L1Loss()  

    for epoch in range(10):
        model.train()
        total_loss = 0.0
        for images, scores in dataloader:
            images, scores = images.to(device), scores.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, scores)  
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")


    save_path = "C:/Users/frank/Documents/science fair 2025/model/efficientnet.pth"
    torch.save(model.state_dict(), save_path)
