import torch
from Data.dataset import SpectrogramDataset
from Model.unet import UNetModel
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_dataset = SpectrogramDataset(split="test")
num_classes = len(test_dataset.genres)
model = UNetModel(in_channels=3, out_channels=num_classes)
model = model.to(DEVICE)

model.load_state_dict(torch.load(
    "resulting_model/unet_final.pth",
    map_location=DEVICE,
    weights_only=True
))
model.eval()

target_filename = "country.00090.png"

for path, label in test_dataset.samples:
    if target_filename in path:
        img = Image.open(path).convert("RGB")
        input_tensor = test_dataset.transform(
            img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)
            predicted_label = torch.argmax(output, dim=1).item()
            predicted_genre = test_dataset.genres[predicted_label]

        print(f"Predicted genre for {target_filename}: {predicted_genre}")
        break
