from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import torch



class ImagesData(Dataset):
    def __init__(self, data_path, tranform=None):
        self.path = data_path
        self.transform = tranform
        self.image_sets = [file_name for file_name in os.listdir(self.path)]
        
    
    def __len__(self):
        return len(self.image_sets)
    
    def __getitem__(self, index):
        image_group = self.image_sets[index]
        image_data = []
        image_annotation = []
        
        for image_name in os.listdir(os.path.join(self.path,image_group)):
            image_path = os.path.join(self.path,image_group,image_name)
            image_data.append(Image.open(image_path))
            
            #Needs to be completed and annotations added
            
        
        
            
        input_images = torch.stack(image_data)
        