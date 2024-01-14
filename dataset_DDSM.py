from PIL import Image
import os
from torch.utils.data import Dataset
import torch
import os
import torch
from torchvision import transforms
from sql_database.database_connect import DDSMDataset



class Cancer_Classification_Data(Dataset):
    def __init__(self, tranform=None):
        self.transform = tranform
        self.database = DDSMDataset()
        
    def __len__(self):
        return self.database.get_length()

    def __getitem__(self, index):
        table_data = self.database.get_grouped_data(index)
        image_data = list()
        image_annotation = None
        annotation_format = (('MLO','LEFT'),('MLO','RIGHT'),('CC','LEFT'),('CC','RIGHT'))
        
        for number in range(len(annotation_format)):
            for item in table_data:
                if item['direction'] == annotation_format[number][1] and item['image_view'] == annotation_format[number][0]:
                    image_data.append(item['pixel_data'])
                    
                    if item['cancer'] == "BENIGN":
                        image_annotation = 0
                    elif item['cancer'] == "MALIGNANT":
                        image_annotation = 1
                    

        input_images = torch.stack([transforms.ToTensor()(img) for img in image_data])
        image_labels = torch.tensor(image_annotation)

        if self.transform:
            input_images = self.transform(input_images)

        return input_images, image_labels
    
    
class Density_Classification_Data(Dataset):
    def __init__(self, tranform=None):
        self.transform = tranform
        self.database = DDSMDataset()
        
    def __len__(self):
        return self.database.get_length()

    def __getitem__(self, index):
        table_data = self.database.get_grouped_data(index)
        image_data = list()
        image_annotation = None
        annotation_format = (('MLO','LEFT'),('MLO','RIGHT'),('CC','LEFT'),('CC','RIGHT'))
        
        for number in range(len(annotation_format)):
            for item in table_data:
                if item['direction'] == annotation_format[number][1] and item['image_view'] == annotation_format[number][0]:
                    image_data.append(item['pixel_data'])
                    image_annotation = item['density']-1


        input_images = torch.stack([transforms.ToTensor()(img) for img in image_data])
        image_labels = torch.tensor(image_annotation)

        if self.transform:
            input_images = self.transform(input_images)

        return input_images, image_labels