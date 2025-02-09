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

        label = float(item["score"])
        return image, torch.tensor(label, dtype=torch.float)


class VisionTransformerEncoder(nn.Module):
    def __init__(self, hidden_size=768, num_layers=12, num_heads=12, mlp_size=3072, dropout=0.1):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, 197, hidden_size))
        self.ln = nn.LayerNorm(hidden_size)

        self.layers = nn.ModuleDict({
            f'encoder_layer_{i}': nn.ModuleDict({
                'ln_1': nn.LayerNorm(hidden_size),
                'self_attention': nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True),
                'ln_2': nn.LayerNorm(hidden_size),
                'mlp': nn.Sequential(
                    nn.Linear(hidden_size, mlp_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_size, hidden_size),
                )
            }) for i in range(num_layers)
        })

    def forward(self, x):
        x = x + self.pos_embedding

        for i in range(len(self.layers)):
            layer = self.layers[f'encoder_layer_{i}']

            residual = x
            x_ln = layer['ln_1'](x)
            x_attn, _ = layer['self_attention'](x_ln, x_ln, x_ln)
            x = residual + x_attn

            residual = x
            x_ln = layer['ln_2'](x)
            x_mlp = layer['mlp'](x_ln)
            x = residual + x_mlp

        x = self.ln(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3,
                 hidden_size=768, mlp_size=3072, num_layers=12, num_heads=12,
                 num_classes=1, dropout=0.1):
        super().__init__()

        self.vit = nn.Module()

        self.vit.conv_proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

        self.vit.class_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.vit.encoder = VisionTransformerEncoder(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_size=mlp_size,
            dropout=dropout
        )

        self.vit.heads = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.vit.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)

        batch_size = x.shape[0]
        class_token = self.vit.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_token, x], dim=1)

        x = self.vit.encoder(x)
        x = x[:, 0]

        x = self.vit.heads(x)
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
model = VisionTransformer().to(device)

checkpoint = torch.load("/Users/weidai/Desktop/model/vit.pth", map_location=device, weights_only=True)
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