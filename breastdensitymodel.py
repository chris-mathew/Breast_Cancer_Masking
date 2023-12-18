# Initial breast density model
# Adapted from: https://www.frontiersin.org/articles/10.3389/fpubh.2022.885212/full#B25
# Last Update: 6th Dec 2023


# Imports
import os
import pickle
import numpy as np
import torch
import csv
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import datasets, transforms
import torchvision.models as models
import tensorflow
import pandas as pd
from PIL import Image
import os
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt


data_path = 'ADD YOUR DATA PATH HERE'


# Data Loader
class BreastImageDataset(Dataset):
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
            label = image_name_split[4]
            annotation_int = int(label)   #Converting the string label into an integer label
            image_annotation.append(annotation_int)


        # input_images = torch.stack(image_data)
        input_images = torch.stack([transforms.ToTensor()(img) for img in image_data])
        image_labels = torch.tensor(image_annotation)

        if self.transform:
            input_images = self.transform(input_images)

        return input_images, image_labels


## The proposed model performance is validated by spitting the image dataset in a ratio of 80% as training and 20% as testing

# Load  train and test datasets using the data loader and split randomly
dataset = BreastImageDataset(data_path)

# Split 80:20
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# Split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders for train and test datasets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)




# Define Model: 


# This section consists of a batch normalization layer and a one × one convolution layer followed
# by a two × two average pooling layer. This layer combines two nearby dense block layers to reduce 
# the feature map size. A combination of 4 dense blocks and transition layers converts the image size 
# into 7 × 7 × 3, further provided to the output layer. Each layer connects to the previous stage as an 
# input described by Equation (4): Xl=Hl([x0,x1,……,xl−1]) 

# A non-linear transformation function Hl(.) is responsible for combining series output of batch normalization,
# ReLU, pooling, and convolution operation. 


class TransitionLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class BottleneckLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(BottleneckLayer, self).__init__()
        inter_channels = 4 * growth_rate
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        return torch.cat([x, out], 1)
 

# In four dense blocks, the individual layer is responsible for forming a k-characteristic map after convolution, 
# which also maintains feature maps of each layer are in the same size. K convolution kernels extract all the features from the layers. 
# Parameter k is known as a hyperparameter in DenseNet, which is the growth rate of the network. Each dense layer receives the different 
# inputs from previous layers to reduce computation and enhance the efficiency of the dense block. The dense block internally uses the 
# bottleneck layer (1 × 1 convolution layer between batch normalization, ReLU, and 3 × 3 convolution layer).


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(BottleneckLayer(in_channels + i * growth_rate, growth_rate))
        self.layers = nn.Sequential(*layers)



class CustomDenseNet(nn.Module):
    def __init__(self):
        super(CustomDenseNet, self).__init__()
        
        # Define parameters
        input_channels = 64
        output_classes = 4 # BIRADS
        growth_rate = 12
        block_config = [6, 12, 24, 16]  # Number of layers in each dense block
        
        # Loade the pretrained densenet 121 model
        self.densenet = models.densenet121(pretrained=True)
        self.densenet = nn.Sequential(*list(self.densenet.children())[:-1])
        
    
        # Build Dense Blocks & Transition Layers (!! CHECK THIS !!)
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(input_channels, growth_rate, num_layers)
            setattr(self, f'denseblock{i + 1}', block)
            input_channels += num_layers * growth_rate
            
            if i != len(block_config) - 1:
                trans = TransitionLayer(input_channels, input_channels // 2)
                setattr(self, f'transition{i + 1}', trans)
                input_channels //= 2
        

        # Output Layer (avg pool for each of the 4 images)
        self.avg_pool_L_CC = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_L_MLO = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_R_CC = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_R_MLO = nn.AdaptiveAvgPool2d(1)
        
        self.flatten = nn.Flatten()

        # Individual Dense Layers
        self.dense_L_CC = nn.Linear(input_channels, output_classes)
        self.dense_L_MLO = nn.Linear(input_channels, output_classes)
        self.dense_R_CC = nn.Linear(input_channels, output_classes)
        self.dense_R_MLO = nn.Linear(input_channels, output_classes)

        # Concatenation Blocks
        self.concat_block1 = nn.Linear(4 * output_classes, input_channels)  
        self.concat_block2 = nn.Linear(2 * input_channels, input_channels)  
        self.concat_block3 = nn.Linear(3 * input_channels, input_channels)  # All features together
    

        # Final Classification Layer
        self.classifier = nn.Linear(input_channels, output_classes)


    def forward(self, x):
        features = self.densenet(x)
        
        # Apply average pooling for each channel
        avg_L_CC = self.avg_pool_L_CC(features)
        avg_L_MLO = self.avg_pool_L_MLO(features)
        avg_R_CC = self.avg_pool_R_CC(features)
        avg_R_MLO = self.avg_pool_R_MLO(features)
        
        # Flatten the pooled features
        flat_L_CC = self.flatten(avg_L_CC)
        flat_L_MLO = self.flatten(avg_L_MLO)
        flat_R_CC = self.flatten(avg_R_CC)
        flat_R_MLO = self.flatten(avg_R_MLO)
        
        # Individual Dense Layers for each channel
        out_L_CC = self.dense_L_CC(flat_L_CC)
        out_L_MLO = self.dense_L_MLO(flat_L_MLO)
        out_R_CC = self.dense_R_CC(flat_R_CC)
        out_R_MLO = self.dense_R_MLO(flat_R_MLO)
        
        # Concatenate MBD features
        concat1 = torch.cat((out_L_CC, out_L_MLO, out_R_CC, out_R_MLO), dim=1)
        concat1 = self.concat_block1(concat1)
        
        # Concatenate MBD features and Concatenate blocks
        concat2 = torch.cat((concat1, flat_L_CC, flat_L_MLO, flat_R_CC, flat_R_MLO), dim=1)
        concat2 = self.concat_block2(concat2)
        
        # Concatenate all features together
        concat3 = torch.cat((concat2, flat_L_CC, flat_L_MLO, flat_R_CC, flat_R_MLO), dim=1)
        concat3 = self.concat_block3(concat3)
        
        # Final classification layer
        logits = self.classifier(concat3)
        return logits



model = CustomDenseNet() 

# Define Optimiser 
params = model.parameters() 
optimiser = optim.SGD(params, lr=0.1) 

# Define Loss (objective function)
## Model uses categorical cross entropy loss
loss = nn.CrossEntropyLoss() 



# Training Loop 
## The entire model is trained with stochastic gradient descent (SGD) algorithm using batch sizes 4 and 30 epoch on the whole dataset
epochs = 30 # Number of loops through data set 

for epoch in range(epochs):
  losses = list() 
  accuracies = list()
  model.train() # Needed since using dropout
  for batch in train_loader: 
    x, y =  batch #(x is input (features) and y is label)    


# SUPERVISED LEARNING: 
    # Step 1: Forward 
    l = model(x) #logits 

    # Step 2: Compute Objective Function 
    J = loss(l, y) 

    # Step 3: Clean the Gradient --> necessary for computing gradients 
    model.zero_grad() 

    # Step 4: Accumulate Partial Derivatives of J (with respect to parameters)
    J.backward()

    # Step 5: Step (in opposite direction of gradient)
    optimiser.step() 

    losses.append(J.item())
    accuracies.append(y.eq(l.detach().argmax().cpu()).float().mean())
    
  
  print(f'Epoch {epoch + 1}: Training Loss = {torch.tensor(losses).mean():.2f}, Accuracy = {torch.tensor(accuracies).mean():.2f}')


# Validation Loops
losses = list()
accuracies = list() 
model.eval() # Set model evaluation mode since dropout is sensitive to mode
for batch in test_loader: 
    x, y =  batch 

    # Step 1: Forward 
    with torch.no_grad(): 
      l = model(x)  # Just compute final outcome (no recording gradients etc) 

    # Step 2: Compute Objective Function 
    J = loss(l, y) 

    losses.append(J.item())
    accuracies.append(y.eq(l.detach().argmax().cpu()).float().mean()) 

print(f'Final Values: Validation Loss = {torch.tensor(losses).mean():.2f}, Accuracy = {torch.tensor(accuracies).mean():.2f}')
     



## SOME CODE FOR PLOTTING TO CHECK MODEL PERFORMANCE (NEED TO UPDATE ACCORDINGLY)

# Plot the validation accuracy
plt.plot(val_acc_history, label='Validation Accuracy', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Print out average accuracy and standard deviation of accuracy
print('Average accuracy: {:.2f} %'.format(np.mean(val_acc_history)*100))
print('Standard deviation of accuracy: {:.2f}'.format(np.std(val_acc_history)))


# Plot learning curves
plt.figure()
plt.title("Learning Curves")
plt.plot(train_loss_history, label='Training Loss', color='blue')
plt.plot(val_loss_history, label='Validation Loss', color='purple')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()


# Compute ROC curve and AUC score
y_true = []
y_score = []
with torch.no_grad():
    for batch_idx, (x, y) in enumerate(test_data):
        y_pred = model(x)
        y_true.append(y.numpy())
        y_score.append(y_pred.numpy())
y_true = np.array(y_true)
y_score = np.array(y_score)


from sklearn.metrics import roc_curve, auc
fpr = dict()
tpr = dict()
auc_score = dict()
for i in range(8): 
  fpr[i], tpr[i], _ = roc_curve(y_true == i, y_score[:,i])
  auc_score[i] = auc(fpr[i], tpr[i])

# Plot ROC curve
plt.figure()
for i in range(8):
    plt.plot(fpr[i], tpr[i], label='Class %d (AUC = %0.2f)' % (i, auc_score[i]))
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curves')
plt.legend(loc="lower right")
plt.show()

