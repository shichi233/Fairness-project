import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
import torch.optim as optim


# 1/26/2025
# ===============================================================================================
class CovidRayDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = []
        file = open(annotation_file, "r", encoding="utf-8")
        for line in file:
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
        return image.view(-1), torch.tensor(label, dtype=torch.float)


# ===============================================================================================

# Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

annotation_file = "/Users/weidai/Desktop/dataforsciencefair/brixia/annotation_train.jsonl"
img_dir = "/Users/weidai/Desktop/dataforsciencefair/brixia"

dataset = CovidRayDataset(annotation_file, img_dir, transform=transform)

dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)


# ===============================================================================================

# model
class CovidMLP(nn.Module):
    def __init__(self, ips, hds, ops):  # input_size, hidden_size, outputsize
        super(CovidMLP, self).__init__()
        self.fc1 = nn.Linear(ips, hds)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hds, hds)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hds, ops)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


# ===============================================================================================
# trainging

def train_model(model, dtld, crt, opt, epochs):  # varchange: dataloader, criterion, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for images, labels in dtld:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = crt(outputs.squeeze(), labels)

            sq_error = torch.sum((outputs.squeeze() - labels) ** 2)
            ab_error = torch.sum(torch.abs(outputs.squeeze() - labels))
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(loss.item())
            
            total_sq_error += sq_error.item()
            total_ab_error += ab_error.item()
            total_samples += labels.size(0)

        avg_mse = total_sq_error / total_samples
        avg_mae = total_ab_error / total_samples

        print(f"Epoch [{epoch+1}/{epochs}], MSE: {avg_mse:.4f}, MAE: {avg_mae:.4f}")
# ===============================================================================================


ips = 224 * 224 * 3
hds = 512
ops = 1

model = CovidMLP(ips, hds, ops)
crt = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr=0.001)
# ===============================================================================================

# train model
if __name__ == '__main__':
    train_model(model, dataloader, crt, opt, epochs=10)
    save_path = "/Users/weidai/Desktop/model/mlp.pth"
    torch.save(model.state_dict(), save_path)
