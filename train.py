import torch
from Data.dataset import SpectrogramDataset
from Model.unet import UNetModel
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torch.nn import functional as F

# hyper parameters
BATCH_SIZE = 8
EPOCHS = 50
LR = 5e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model/data setup stuff
SAVE_MODEL_DIR = "resulting_model"
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)

train_dataset = SpectrogramDataset()
num_classes = len(train_dataset.genres)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

model = UNetModel(in_channels=3, out_channels=num_classes)
model = model.to(DEVICE)

# needs to be good at mutually exclusive classes
criterion = nn.CrossEntropyLoss()
"""
More info about this
https://medium.com/data-science/musical-genre-classification-with-convolutional-neural-networks-ff04f9601a74
"""
optimizer = optim.Adam(model.parameters(), lr=LR)


def train_epoch(model, dataloader):
    model.train()
    running_loss = 0.0
    for _, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(dataloader)


train_losses = []

for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader)

    train_losses.append(train_loss)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Train Loss: {train_loss:.5f}"
    )

    if ((epoch+1) % 5) == 0:
        torch.save(
            model.state_dict(),
            f"{SAVE_MODEL_DIR}/unet_epoch_{epoch+1}.pth"
        )

torch.save(model.state_dict(), f"{SAVE_MODEL_DIR}/unet_final.pth")
