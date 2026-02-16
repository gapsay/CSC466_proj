import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class SpectrogramDataset(Dataset):
    def __init__(self, split="train"):
        if split == "test":
            self.root_dir = r".\Data\processed_data\test"
        else:
            self.root_dir = r".\Data\processed_data\train"

        # get the genres and map them to indices
        self.split_dir = os.path.join(self.root_dir, split)
        self.genres = sorted([d for d in os.listdir(self.split_dir)
                             if os.path.isdir(os.path.join(self.split_dir, d))])
        self.class_to_idx = {genre_name: i for i,
                             genre_name in enumerate(self.genres)}
        self.samples = []

        # splits the images up into a list of (filepath, label) tuples
        for genre in self.genres:
            genre_folder = os.path.join(self.split_dir, genre)
            for filename in os.listdir(genre_folder):
                filepath = os.path.join(genre_folder, filename)
                self.samples.append((filepath, self.class_to_idx[genre]))
        self.transform = T.Compose([
            T.Resize((128, 512)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label)
