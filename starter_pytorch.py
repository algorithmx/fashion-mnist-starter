import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

# Step 1: Load the Fashion-MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Use shared loader which also handles downloading the raw IDX files.
from dataset_loader import get_torch_loaders

# `get_torch_loaders` will download raw IDX files (via torchvision) if needed,
# then return DataLoaders. This keeps download logic centralized in the loader.
train_loader, test_loader = get_torch_loaders(
    batch_size=64,
    root='data',
    transform=transform,
    download=True,
    num_workers=0,
    pin_memory=False,
)

# Step 2: Define neural network model
class MinimalFashionCNN(nn.Module):
    def __init__(self):
        super(MinimalFashionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 13 * 13, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

model = MinimalFashionCNN()

# Step 3: Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: Select device.
# If CUDA is available, check the GPU compute capability
# and fall back to CPU for GPUs with compute capability < 7.0 (sm_70).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    try:
        props = torch.cuda.get_device_properties(0)
        major, minor = props.major, props.minor
    # Allow GPUs with compute capability >= 6 (Pascal and newer)
        if major < 6:
            print(f"GPU compute capability {major}.{minor} is likely too old. Falling back to CPU.")
            device = torch.device("cpu")
    except Exception:
        # If anything goes wrong querying properties, fall back to CPU
        print("Could not query CUDA device properties; using CPU instead.")
        device = torch.device("cpu")

model = model.to(device)


# Step 5: Train the model
def train_model(num_epochs=10):
    train_acc_history, val_acc_history = [], []

    for epoch in range(num_epochs):
        # Training
        model.train()
        t_loss, t_corr, n = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            t_corr += (outputs.argmax(1) == y).sum().item()
            n += len(y)
        
        # Validation
        model.eval()
        v_corr, m = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                v_corr += (model(x).argmax(1) == y).sum().item()
                m += len(y)
        
        t_acc, v_acc = t_corr/n, v_corr/m
        train_acc_history.append(t_acc)
        val_acc_history.append(v_acc)
        print(f'Epoch {epoch+1}: Loss: {t_loss/len(train_loader):.4f}, Train Acc: {t_acc:.4f}, Val Acc: {v_acc:.4f}')

    print(f'Test accuracy: {v_acc:.4f}')

if __name__ == '__main__':
    train_model()