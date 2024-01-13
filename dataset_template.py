from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import torch



class ImagesData(Dataset):
    def __init__(self, data_path, tranform=None):
        #data path refers to the path where the folders containing the grouped images are located (ie.C:\Users\chris\OneDrive - Imperial College London\CBIS Dataset\manifest-Egq0PU078220738724328010106\CBIS-DDSM\Grouped)
        
        self.path = data_path
        self.transform = tranform
        self.image_sets = [folder_name for folder_name in os.listdir(self.path)]      #Takes the folder name of each set in the folder
        
    
    def __len__(self):
        return len(self.image_sets)
    
    def __getitem__(self, index):
        image_group = self.image_sets[index]
        image_data = []  #List of the image data for all of the views
        image_annotation = [] #List of which view the image corresponds to
        
        for image_name in os.listdir(os.path.join(self.path,image_group)):   #Image file names in the chosen set
            image_path = os.path.join(self.path,image_group,image_name)
            image_data.append(Image.open(image_path))
            
            image_name_split = image_name.split(".")[0].split("_")   #The file name is split to obtain the direction and view of the image
            label = f'{image_name_split[2]}_{image_name_split[3]}'
            annotation_int = self.label_to_int(label)   #Converting the string label into an integer label as there are only 4 possible label options
            image_annotation.append(annotation_int)
        
            
        input_images = torch.stack(image_data)
        annotation_tensor = torch.tensor(image_annotation)
        
        if self.transform:
            input_images = self.transform(input_images)
        
        return input_images, annotation_tensor
        
    def label_to_int(label):
        label_dict = {
            'LEFT_CC':0,
            'RIGHT_CC':1,
            'LEFT_MLO':2,
            'RIGHT_MLO':3
        }
        return label_dict[label]