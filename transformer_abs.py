import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
import torchvision.models as models
import torch.optim as optim

class CovidRayDataset(Dataset):
    def __init__ (self, annotation_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = []
        with open(annotation_file, "r", encoding = "utf-8") as f:
            for line in f:
                self.data.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, n):
        item = self.data[n]
        img_filename = item["images"][0]
        img_path = os.path.join(self.img_dir,img_filename)
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
annotation_file = "/Users/weidai/Desktop/dataforsciencefair/brixia/annotation_train.jsonl"
img_dir = "/Users/weidai/Desktop/dataforsciencefair/brixia"


dataset = CovidRayDataset(annotation_file, img_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

class CovidVit(nn.Module):
    def __init__(self):
        super(CovidVit, self).__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads = nn.Linear(self.vit.heads.head.in_features, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.vit(x)
        x = self.sigmoid(x)
        x = x * 18
        return x.squeeze()
def train_model(model, dataloader, epochs=10, lr=1e-4, device="mps"):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    for epoch in range(epochs):
        model.train()
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(loss.item())
    return model

def evaluate_mae(model, dataloader, device="mps"):
    model.to(device)
    model.eval()
    
    total_abs_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images).squeeze()
            abs_error = torch.sum(torch.abs(outputs - labels)).item()

            total_abs_error += abs_error
            total_samples += labels.size(0)

    avg_mae = total_abs_error / total_samples
    print(f"Mean Absolute Error (MAE): {avg_mae:.4f}")

if __name__ == '__main__':
    model = CovidVit()
    trained_model = train_model(model, dataloader, epochs=10, lr=1e-4, device="mps")

    evaluate_mae(trained_model, dataloader, device="mps")

    torch.save(trained_model.state_dict(), "/Users/weidai/Desktop/model/vit.pth")
