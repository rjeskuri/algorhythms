import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
import torch.optim as optim


data = torch.load("models/data/data.pt")

# Define graph network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GraphConv(29, 64)
        self.conv2 = GraphConv(64, 8)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight))

        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Training the model...
model = Net().to(device)  
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# Define Binary Cross Entropy with Logits Loss
bce_with_logits_loss = torch.nn.BCEWithLogitsLoss()

for epoch in range(10):
    print(epoch)
    optimizer.zero_grad()
    out = model(data)  # These are our node embeddings
    pos_out = (out[data.edge_index[0]] * out[data.edge_index[1]]).sum(-1)  # Get embeddings for nodes that are connected (edges)

    # We now need to compute the loss for the nonexistent edges
    # We'll sample random edges for this.
    neg_edge_index = negative_sampling(edge_index, num_nodes=data.num_nodes, num_neg_samples=data.edge_index.size(1), method="sparse")
    neg_out = (out[neg_edge_index[0]] * out[neg_edge_index[1]]).sum(-1)  # Get embeddings for nodes that aren't connected
    
    pos_labels = torch.ones(pos_out.size(0)).to(device)
    neg_labels = torch.zeros(neg_out.size(0)).to(device)
    
    pos_loss = bce_with_logits_loss(pos_out, pos_labels)  # This calculates the loss for the positive (real) edges
    neg_loss = bce_with_logits_loss(neg_out, neg_labels)  # This calculates the loss for the negative edges

    loss = pos_loss + neg_loss  # The total loss is the sum of the positive and negative loss
    loss.backward()
    optimizer.step()
# Save trained model
torch.save(model.state_dict(), "models/graph_models/graph_model.pt")

# Save embeddings
embeddings = model(data)
torch.save(embeddings, "models/data/embeddings.pt")
