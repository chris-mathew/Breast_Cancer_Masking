import pydicom
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn.functional as F
from segmentationandriskmodel import BreastCancer_CSAE
from torchvision import transforms
from segmentationandriskmodel import BreastCancer_CSAE  # Adjust the import based on your project structure
import torch
import io


def dicom_to_image(dicom_image):
    dicom_data = pydicom.dcmread(dicom_image)

    pixel_data = dicom_data.pixel_array
    pixel_data = pixel_data - np.min(pixel_data)
    pixel_data = pixel_data / np.max(pixel_data)

    image_rgb = plt.cm.gray(pixel_data)
    image_rgb = (image_rgb[:, :, :3] * 255).astype(np.uint8)

    return Image.fromarray(image_rgb)


path = 'dicom_sample.dcm'

with open(path, 'rb') as file:
    image = dicom_to_image(file)

model = BreastCancer_CSAE()

# Load the model's state dictionary into a BytesIO buffer
with open('density_model.pth', 'rb') as file:
    buffer = io.BytesIO(file.read())

# Load the state dictionary into the model
state_dict = torch.load(buffer, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
])

input_data = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_data)
    predicted_class = torch.argmax(output, dim=1).item()

print("Predicted Class:", predicted_class)