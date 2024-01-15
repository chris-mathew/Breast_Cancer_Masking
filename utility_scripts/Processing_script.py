import pydicom
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


def dicom_to_png(dicom_path, png_path):

    dicom_data = pydicom.dcmread(dicom_path)

    pixel_data = dicom_data.pixel_array
    pixel_data = (pixel_data - np.min(pixel_data))/ np.max(pixel_data)

    rgb_image = plt.cm.gray(pixel_data)
    #Adding RGB channels to the image
    rgb_image = (rgb_image[:, :, :3] * 255).astype(np.uint8)
    #Saving 
    Image.fromarray(rgb_image).save(png_path)


# Image resize function to ensure images are consistent
def resize_image(path_in, size):
    original_image = Image.open(path_in)

    resized_image = original_image.resize(size)

    resized_image.save(path_in)


path = "C:/Users/chris/OneDrive - Imperial College London/CBIS Dataset/manifest-Egq0PU078220738724328010106/CBIS-DDSM" #PATH TO THE DICOM FILES
folder_names = os.listdir(path)
size_new = (320,320)

#Scans throught the local dataset folder
for folder in folder_names:
    input_path = path + '/' + folder
    output_path = path + '/' + 'Reformatted' + '/' + folder[:-4] + '.png'  #PNG IMAGES ARE CREATED IN A FOLDER CALLED "Reformatted"
    dicom_to_png(input_path, output_path) #DICOM is convereted to a png format
    resize_image(output_path, size_new)