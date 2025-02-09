import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
import timm  # Required for EVA-2
import torch.optim as optim


class CovidRayDataset(Dataset):
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
        img_filename = item["images"][0]
        img_path = os.path.join(self.img_dir, img_filename)
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

# Data path
annotation_file = "/Users/weidai/Desktop/dataforsciencefair/brixia/annotation_train.jsonl"
img_dir = "/Users/weidai/Desktop/dataforsciencefair/brixia"

# Dataloader
dataset = CovidRayDataset(annotation_file, img_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)


class CovidEVA2(nn.Module):
    def __init__(self):
        super(CovidEVA2, self).__init__()
        self.eva2 = timm.create_model("eva02_base_patch14_224", pretrained=True, num_classes=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.eva2(x)
        x = self.sigmoid(x) * 18
        return x.squeeze()


# Training function
def train_model(model, dataloader, epochs=10, lr=1e-4, device="cuda"):
    model.to(device)
    criterion = nn.MSELoss()  # Regression loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            print(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(loss.item())
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)

    return model


if __name__ == '__main__':
    device = torch.device("mps")
    model = CovidEVA2()
    trained_model = train_model(model, dataloader, epochs=10, lr=1e-4, device=device)

    torch.save(trained_model.state_dict(), "/Users/weidai/Desktop/model/eva2.pth")
