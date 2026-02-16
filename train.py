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

BATCH_SIZE = 8
EPOCHS = 5
LR = 5e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_MODEL_DIR = "resulting_model"
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)

train_dataset = SpectrogramDataset()
num_classes = len(train_dataset.genres)

model = UNetModel(in_channels=3, out_channels=num_classes)
model = model.to(DEVICE)
