import torch
from torch_geometric.nn import GraphConv
import torch.nn as nn
import torch.optim as optim

# Load tensors
features = torch.load("models/data/features.pt")
embeddings = torch.load("models/data/embeddings.pt")

# Define MLP network
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Training the model
model = MLP(input_dim, hidden_dim, output_dim).to(device)
input_dim = features.shape[1]  

# Assume that each embedding has size 8
output_dim = 8  
hidden_dim = 16  

model = MLP(input_dim, hidden_dim, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 10000

# Move data to the device
features = features.to(device)
embeddings = embeddings.to(device)

# Use Mean Squared Error Loss as our regression loss function
criterion = nn.MSELoss()

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(features)
    
    # Compute loss
    loss = criterion(outputs, embeddings)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Save trained model
torch.save(model.state_dict(), "models/feed_forward_models/feed_forward_model.pt")
