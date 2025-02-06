import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
import torch.optim as optim


#======================================================================================
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

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: {img_path} not found. Skipping...")
            return None, None

        label = float(item["score"])
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float)


# Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

annotation_file = "C:/Users/frank/Documents/science fair 2025/brixia/annotation_train.jsonl"
img_dir = "C:/Users/frank/Documents/science fair 2025/brixia"

dataset = CovidRayDataset(annotation_file, img_dir, transform=transform)

dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)  # Set workers to 0 if on Mac

#======================================================================================
# MLP Model
class CovidMLP(nn.Module):
    def __init__(self, ips, hds, ops):  # input_size, hidden_size, output_size
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


# Training Function
def train_model(model, dtld, crt, opt, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for images, labels in dtld:
            if images is None:  # Skip missing images
                continue

            images, labels = images.to(device), labels.to(device)
            images = images.view(images.size(0), -1)  # Flatten before passing to MLP

            outputs = model(images)
            loss = crt(outputs.squeeze(), labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_samples += labels.size(0)

        avg_loss = total_loss / total_samples
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

#======================================================================================

def mae(model, dtld):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    total_abs_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dtld:
            if images is None:  
                continue

            images, labels = images.to(device), labels.to(device)
            images = images.view(images.size(0), -1) 

            outputs = model(images).squeeze()
            abs_error = torch.abs(outputs - labels) 

            total_abs_error += abs_error.sum().item()  
            total_samples += labels.size(0)

    avg_mae = total_abs_error / total_samples  
    print(f"Final MAE after training: {avg_mae:.4f}")

#======================================================================================
# Main Script
if __name__ == '__main__':
    input_size = 224 * 224 * 3  
    hidden_size = 512  
    output_size = 1 
    model = CovidMLP(input_size, hidden_size, output_size)

    crt = nn.MSELoss()  # Mean Squared Error 
    opt = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, dataloader, crt, opt, epochs=10)
    mae(model, dataloader)

    save_path = "/Users/weidai/Desktop/model/mlp.pth"
    torch.save(model.state_dict(), save_path)
