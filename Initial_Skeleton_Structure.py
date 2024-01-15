# Initial Skeleton Structure

# Convolutional Architecture
class breastmodel(nn.Module):
    def __init__(self):
        super(breastmodel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 50, 7)
        self.conv2 = nn.Conv2d(50, 50, 2)  
        self.conv3 = nn.Conv2d(50, 50, 5)
        self.conv4 = nn.Conv2d(50, 100, 5)
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        # Sparse autoencoder layers
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Apply convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        # Flatten the tensor before feeding it into the autoencoder
        x = torch.flatten(x, 1)  # Assuming the batch size is dimension 0
        
        # Apply sparse autoencoder layers
        encoded = F.relu(self.encoder(x))
        decoded = torch.sigmoid(self.decoder(encoded))  # Using sigmoid for output
        
        return encoded, decoded
    

# Sparse Autoencoder
# class SparseAutoencoder(nn.Module):
#     def __init__(self):
#         super(SparseAutoencoder, self).__init__()
#         self.N = 48000
#         self.p = 0.01
#         self.lamba = 1
#         self.encoder = nn.Linear(input_size, hidden_size)
#         self.decoder = nn.Linear(hidden_size, output_size)
        
#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded

# Instantiate model and autoencoder
model = breastmodel()
# autoencoder = SparseAutoencoder()
