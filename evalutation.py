import torch
from Data.dataset import SpectrogramDataset
from Model.unet import UNetModel
from PIL import Image

from sklearn.metrics import classification_report, confusion_matrix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_dataset = SpectrogramDataset(split="test")
num_classes = len(test_dataset.genres)

model = UNetModel(in_channels=3, out_channels=num_classes).to(DEVICE)

model.load_state_dict(torch.load(
    "resulting_model/unet_final.pth",
    map_location=DEVICE,
    weights_only=True
))

model.eval()
genres = test_dataset.genres

y_true = []
y_pred = []

for path, label in test_dataset.samples:
    img = Image.open(path).convert("RGB")
    input_tensor = test_dataset.transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_label = torch.argmax(output, dim=1).item()
    y_true.append(label)
    y_pred.append(predicted_label)

print("\n===== Classification Report =====\n")

print(classification_report(
    y_true,
    y_pred,
    target_names=genres,
    digits=4
))

print("\n===== Confusion Matrix =====\n")
cm = confusion_matrix(y_true, y_pred)
print(cm)
