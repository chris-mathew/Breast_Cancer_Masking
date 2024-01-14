import pydicom
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def dicom_to_image(dicom_image):

    dicom_data = pydicom.dcmread(dicom_image)

    pixel_data = dicom_data.pixel_array
    pixel_data = pixel_data - np.min(pixel_data)
    pixel_data = pixel_data / np.max(pixel_data)

    image_rgb = plt.cm.gray(pixel_data)
    image_rgb = (image_rgb[:, :, :3] * 255).astype(np.uint8)
    
    return Image.fromarray(image_rgb)