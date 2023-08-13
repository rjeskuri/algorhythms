"""
This script requires 2 prompts:
Prompt 1 (required) : Folder name of the data version to gather the pytorch-geometric data object.
Prompt 2 (required) : Path to the folder and file within the 'embeddings' folder
Prompt 3 (optional) : Number of epochs for training. Default is 1000.
"""

import os
import pickle
import configparser
import glob
import sys
import torch
from torch_geometric.nn import GraphConv
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim


if __name__ == "__main__":
    # Load tensors
    #features = torch.load("models/data/features.pt")
    #embeddings = torch.load("models/data/embeddings.pt")

    config_options = configparser.ConfigParser()
    script_directory = os.path.dirname(os.path.abspath(__file__))
    conf_dir = os.path.join(script_directory, 'conf')
    config_options.read(os.path.join(conf_dir, 'models.conf'))
    dataBaseDirectory = dict(config_options.items("MODELS_CONFIGS")).get('datawarehousedir')

    print("Base directory for data is located at : {} \n".format(dataBaseDirectory))

    if len(sys.argv) < 3:
        print("""
        Error: A total of 2 arguments have to be provided.
        Argument 1 : Folder name of the data version being used to gather the node features
        E.g. 'version_20230726_123356', which has to be valid folder with contents in 'saved_folder/data_representations'.
        Argument 2 : Path to the folder and file within the 'embeddings' folder
        E.g. 'model_gcnconv_20230806_070751_using_data_v20230804_030210/embedding_10.0_percentile_20230809_041545.pkl'
        """)
        sys.exit(1)  # Exit with a non-zero status code to indicate an error

    dataVersion = sys.argv[1]
    versionDirectoryName = '{}/{}'.format(dataBaseDirectory + "/saved_files/data_representations", dataVersion)

    embeddingChoice = sys.argv[2]
    embeddingFilePath = '{}/{}'.format(dataBaseDirectory + "/saved_files/embeddings", embeddingChoice)

    num_epochs = 10000 # Default
    if len(sys.argv) >= 4:
        try:
            num_epochs = int(sys.argv[3])  # Use the 2nd argument as num_epochs
        except:
            print("Please provide an integer value for 3rd argument , i.e. number of epochs")
            sys.exit(1)
    print("Number of epochs of training : {}".format(num_epochs))

    # Attempt to load 'features'
    search_path = os.path.join(versionDirectoryName, 'data_obj_*.pkl')
    try:
        matching_file = glob.glob(search_path)[0]
    except:
        print("""
        Error: Check if folder you provided as argument exists inside 'saved_folder/data_representations'.
        Path to review the contents is : {}.
        """.format(dataBaseDirectory))
        sys.exit(1)  # Exit with a non-zero status code to indicate an error

    print("Data file chosen for training : {}".format(matching_file))
    with open(matching_file, 'rb') as file:
        data = pickle.load(file)
    features = data.x # Features of each of the songs is part of 'x' attribute
    print("The features has the shape : {}".format(features.shape))

    # Attempt to load 'embeddings'
    try:
        with open(embeddingFilePath, 'rb') as file:
            embeddings = pickle.load(file)
    except:
        print("""
        The embedding file provided is not valid. Please choose a existing folder/file with below example formatting.
        E.g. 'model_gcnconv_20230806_070751_using_data_v20230804_030210/embedding_10.0_percentile_20230809_041545.pkl'
        """)

    # Define MLP network
    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = nnf.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training the model
    input_dim = features.shape[1]
    output_dim = 10
    hidden_dim = 64
    model = MLP(input_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

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

    print("Loss after training: {}".format(loss.item()))

    # Save trained model
    #torch.save(model.state_dict(), "models/feed_forward_models/feed_forward_model.pt")

    # Save model as file
    modelDirectoryPath = dataBaseDirectory + "/saved_files/model_artefacts"
    # Build naming convention for file based on embedding and data version it was based on
    modelList = embeddingChoice.split("/")[0].split("_")
    embeddingList = embeddingChoice.split("/")[1].split("_")
    file_path = "{}/{}".format(modelDirectoryPath, modelList[0]+"_mlp_using_data_"+modelList[-2]+"_"+modelList[-1]+"_"+embeddingList[0]+"_"+embeddingList[1]+"_percentile.pkl")
    print("\nSaving model as pickle file : {}".format(file_path))
    torch.save(model.state_dict(), file_path)
