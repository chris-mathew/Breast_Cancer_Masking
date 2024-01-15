import pydicom
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pickle

#Extracts image data from a dicom file
def dicom_to_image(dicom_image):

    dicom_data = pydicom.dcmread(dicom_image)

    pixel_data = dicom_data.pixel_array
    pixel_data = pixel_data - np.min(pixel_data)
    pixel_data = pixel_data / np.max(pixel_data)

    image_rgb = plt.cm.gray(pixel_data)
    image_rgb = (image_rgb[:, :, :3] * 255).astype(np.uint8)
    
    return Image.fromarray(image_rgb)

#Returns the extension from a file name
def get_extension(name):
    format_name = name.split('.')[-1]
    return format_name

#Get classification values from the machine learning model
def get_classification(image_data):
    density_classification = {0:'A',1:'B',2:'C',3:'D'}
    cancer_classification = {0:'Low',1:'High'}
    
    with open('density_model.pkl', 'rb') as density_file:
        density_model = pickle.load(density_file)
    with open('cancer_model.pkl', 'rb') as cancer_file:
        cancer_model = pickle.load(cancer_file)
    
    density_predict = density_model.predict(image_data)
    cancer_predict = cancer_model.predict(image_data)
    
    return density_classification[density_predict], cancer_classification[cancer_predict]
    
    