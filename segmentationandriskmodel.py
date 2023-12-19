
## Model based on "Unsupervised deep learning applied to breast density segmentation and mammographic risk scoring"  Kallenberg et. Al 

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Initial Skeleton Structure

# Convolutional Architecture
class breastmodel(nn.Module):
    def __init__(self):
        super(breastmodel, self).__init__()
        # Define your convolutional layers, pooling layers, etc.
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.pool = nn.MaxPool2d(kernel_size)
       
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        return x

# Sparse Autoencoder
class SparseAutoencoder(nn.Module):
    def __init__(self):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Instantiate model and autoencoder
model = breastmodel()
autoencoder = SparseAutoencoder()

# Define  
params = model.parameters() 
optimiser = optim.Adam(model.parameters(), lr=0.001)
loss = nn.CrossEntropyLoss() 



# Training Loop 
epochs = 30 # Number of loops through data set 

for epoch in range(epochs):
  losses = list() 
  accuracies = list()
  model.train() # Needed since using dropout
  for i, batch in enumerate(train_loader): 
    x, y =  batch #(x is input (features) and y is label)    
    l = model(x) #logits 
    J = loss(l, y) 
    model.zero_grad() 
    J.backward()
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


