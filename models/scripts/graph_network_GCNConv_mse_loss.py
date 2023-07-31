"""
This script accepts 3 prompts:
Prompt 1 (required) : Folder name of the data version to gather the pytorch-geometric data object.
Prompt 2 (optional) : Number of epochs for training. Default is 10.
Prompt 3 (optional) : Number of iterations inside each epoch. Default is 10.

Examples:
    a) When running directly as Python script : python graph_network_GCNConv_mse_loss.py version_20230728_060743 12 12
    b) When running in Slurm using SBATCH     : sbatch batch_gcnconv.sh version_20230728_060743 12 12
"""

import glob
import configparser
import os
import sys
import datetime
import pickle
import torch
import torch.nn.functional as torch_func
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

print("Is torch cuda available? : {}".format(torch.cuda.is_available()))

if __name__ == "__main__":

    '''
    # Uncomment once the '../../data_engineering/code/conf' directory is available to access 
    config_options = configparser.ConfigParser()
    conf_dir = os.environ.get('SPARK_CONF_DIR') or '../../data_engineering/code/conf'  # Options to support Spark CLuster and local modes
    config_options.read('{}/spark.conf'.format(conf_dir))  # Load entries defined in 'spark-start' shell script
    dataBaseDirectory = dict(config_options.items("SPARK_APP_CONFIGS")).get('spark.sql.warehouse.dir')
    '''
    # REMOVE later : The below line of code is temporary while data_engineering code has not been merged
    dataBaseDirectory = '/scratch/siads699s23_class_root/siads699s23_class/shared_data/team_16_algorhythms/data/spark_table_warehouse'

    print("Base directory for data is located at : {} \n".format(dataBaseDirectory))

    if len(sys.argv) < 2:
        print("""
        Error: Data version information not provided as argument. 
        E.g. 'version_20230726_123356', which has to be valid folder with contents in 'saved_folder/data_representations'. 
        Path to review the contents is : {}.
        """.format(dataBaseDirectory))
        sys.exit(1)  # Exit with a non-zero status code to indicate an error

    # Define default number of epochs and number of samples trained per epoch.
    # Override if provided as arguments to script
    num_epochs = 10
    num_train_inside_epoch = 10

    if len(sys.argv) >= 3:
        try:
            num_epochs = int(sys.argv[2])  # Use the 2nd argument as num_epochs
        except:
            print("Please provide an integer value for 2nd argument , i.e. number of epochs")
            sys.exit(1)
    if len(sys.argv) >= 4:
        try:
            num_train_inside_epoch = int(sys.argv[3])  # Use the 3rd argument as num_train_inside_epoch
        except:
            print("Please provide an integer value for 3rd argument , i.e. number of iterations inside an epoch")
            sys.exit(1)

    # Load 'data' object directly from saved pickle file that was created earlier
    dataVersion = sys.argv[1]
    versionDirectoryName = '{}/{}'.format(dataBaseDirectory + "/saved_files/data_representations", dataVersion)

    # Pick out the 1st from the list (assume there is only 1 in each version directory as per design)
    search_path = os.path.join(versionDirectoryName, 'data_obj_*.pkl')

    try:
        matching_file = glob.glob(search_path)[0]
    except:
        print("""
        Error: Check if folder you provided as argument exists inside 'saved_folder/data_representations'. 
        Path to review the contents is : {}.
        """.format(dataBaseDirectory))
        sys.exit(1)  # Exit with a non-zero status code to indicate an error

    # Log out the choices for traceability
    print("Data file chosen for training : {}".format(matching_file))
    print("Number of epochs chosen for training : {}".format(num_epochs))
    print("Data version chosen for training : {}".format(num_train_inside_epoch))

    with open(matching_file, 'rb') as file:
        data = pickle.load(file)
    print("The entire 'data' object has the below representation : \n{}".format(data))

    # Create mutually exclusive masks with 'zero' tensors
    num_nodes = data.num_nodes
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # 80-10-10 split
    num_train_nodes = int(num_nodes * 0.8)
    num_val_nodes = int(num_nodes * 0.1)
    num_test_nodes = num_nodes - num_train_nodes - num_val_nodes

    # Generate random indices and apply those random indices as '1' for the train, val, and test sets in respective indexes
    random_indices = torch.randperm(num_nodes)
    train_mask[random_indices[:num_train_nodes]] = 1
    val_mask[random_indices[num_train_nodes:num_train_nodes + num_val_nodes]] = 1
    test_mask[random_indices[num_train_nodes + num_val_nodes:]] = 1

    # Helper function to create a validation or test set using the data object and mask generated.
    def create_data_using_mask(data, mask):

        # Get the indices of nodes that are present in the source and destination of edge_index
        edge_src, edge_tgt = data.edge_index

        # Edge mask to keep only those edges 'True' that have the nodes that qualify mask
        # (i.e. 'AND' operation to ensure that edge source and target is from a node that qualifies mask on nodes)
        new_edge_mask = mask[edge_src] & mask[edge_tgt]
        # Now get the new_node_indices that will used by edge_index_new later
        new_node_indices = mask.nonzero(as_tuple=False).squeeze()
        # Get a preliminary edge_index that will be fixed later on with actual node indices
        edge_index_prelim = data.edge_index[:, new_edge_mask]

        # In edge_index_prelim, refer the indexes in new_node_indices and find corresponding index location
        # to create edge_index_new
        """
        Idea behind below code: We essentially need to 're-index' values in edge_index_prelim
        E.g. if it used to refer to an edge as source as 117123 node and target as 856432 node in the original 'data'
        , we need to find their relative position with the nodes we have filtered down into our new dataset nodes. 
        Since the node indices is essentially an 'arange' (i.e. from 0 till length of tensor), it makes it easy to find the nodes position
        from 'new_node_indices' and use that to create the new source and target edge indices.
        Stack the source and edge indices to get back a new edge_index that is same shape as old one
        but with the right references to node indices
        """
        edge_src_new, edge_tgt_new = edge_index_prelim
        edge_index_src_new = torch.searchsorted(new_node_indices, edge_src_new)
        edge_index_tgt_new = torch.searchsorted(new_node_indices, edge_tgt_new)

        edge_index_new = torch.stack([edge_index_src_new, edge_index_tgt_new], dim=0)
        new_node_features = data.x[new_node_indices]
        new_edge_weight = data.edge_weight[new_edge_mask]

        # Create the new new_data using the new_node_features, edge_index_new, new_edge_weight
        new_data = Data(x=new_node_features, edge_index=edge_index_new
                        , edge_weight=new_edge_weight
                        , num_nodes=new_node_features.shape[0])

        return new_data

    val_data = create_data_using_mask(data, val_mask)
    print("Validation data object summary: \n {}".format(val_data))
    test_data = create_data_using_mask(data, test_mask)
    print("Test data object summary: \n {}".format(test_data))

    """
    For train data, although we will apply the train_mask, the training will happen in batches due to the size of the data.
    'train_mask' will ensure that training nodes from the sampler do not overlap with those we have in validation and test
    """

    """
    NeighborLoader : Sample from a neighborhood of nodes that allows for training on messages passed between closeby nodes
    Note: pytorch-geometric 2.3.1 does not support 'subgraph_type' argument to make the graph undirected.
    Appears that newer versions do, but unavailable to pip install at this time. Hence, handled as separate code later.
    """
    dataLoader = NeighborLoader(
        data,
        num_neighbors=[100, 30],  # 100 from 1st adjacent neighbors and 30 from each of neighbors of neighbors
        batch_size=10000, # Number of nodes per batch size. Note: Once sampler picks up neighbors, this will be exponentially larger
        replace=False,  # Sampling will not replace and will essentially go through all the nodes
        input_nodes=train_mask,  # Apply train_mask created earlier in code
    )

    print("NeighborLoader initialization has completed. Moving onto training...")

    # Define the graph convolutional network model based on the 'MessagePassing' class
    class GCN(pyg_nn.MessagePassing):
        def __init__(self, num_features, hidden_size, embedding_size):
            super(GCN, self).__init__(
                aggr='add')  # Message Passing aggregation is 'aggr' (can be sum/mean/min/max) from neighboring nodes
            self.conv1 = pyg_nn.GCNConv(num_features, hidden_size, cached=False)
            self.conv2 = pyg_nn.GCNConv(hidden_size, embedding_size, cached=False)
            self.conv1.reset_parameters()
            self.conv2.reset_parameters()

        def forward(self, x, edge_index):
            x = self.conv1(x=x, edge_index=edge_index)
            x = torch.relu(x)
            x = self.conv2(x=x, edge_index=edge_index)
            return x

    # Custom loss using edge_weights
    def custom_loss(embeddings, edge_index, edge_weight):
        src_embeddings = embeddings[edge_index[0]]
        tgt_embeddings = embeddings[edge_index[1]]
        dot_product = torch.sum(src_embeddings * tgt_embeddings, dim=1)
        loss = torch_func.mse_loss(dot_product, edge_weight)
        return loss

    """
    Note: dataLoader can generate samples infinitely, hence will use a counter 'idx' against 'num_train_inside_epoch' to break.
    """
    # Initialize the model
    model = GCN(num_features=data.x.size(1), hidden_size=16, embedding_size=10)

    # Move over to GPU if environment supports it
    if torch.cuda.is_available():
        model = model.to("cuda")
        model = torch.nn.DataParallel(model)  # Wrap the model with DataParallel

    # Define the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()  # Mean Squared Error loss for regression
    clip_value = 1.0  # Set the value you want to clip gradients at
    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)

    # Training loop for multiple epochs
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()  # Set to training mode
        # Iterate through the dataLoader and train the model on each sample
        for idx, new_sample in enumerate(dataLoader):
            print(idx + 1, end="...")
            # If the sample produced has undirected graphs. Transform the sample graph using .to_undirected()
            if not new_sample.is_undirected():
                new_edge_index, new_edge_weight = pyg_utils.to_undirected(new_sample.edge_index, new_sample.edge_weight)
                new_sample.edge_index = new_edge_index
                new_sample.edge_weight = new_edge_weight

            optimizer.zero_grad()
            output_forwardpass = model(x=new_sample.x
                                       , edge_index=new_sample.edge_index
                                       )
            loss = custom_loss(output_forwardpass, new_sample.edge_index, new_sample.edge_weight)
            loss.backward()
            optimizer.step()
            if idx + 1 == num_train_inside_epoch:
                break

        print("\nTraining Loss after {} epoch/s = {}".format(epoch + 1, loss))
        # Check against validation dataset here
        model.eval()
        output_validation = model(x=val_data.x
                                  , edge_index=val_data.edge_index
                                  )
        val_loss = custom_loss(output_validation, val_data.edge_index, val_data.edge_weight)
        print("Validation Loss after {} epoch/s = {}".format(epoch + 1, val_loss))

    # Assess against test data
    model.eval()
    output_validation = model(x=test_data.x
                              , edge_index=test_data.edge_index)
    test_loss = custom_loss(output_validation, test_data.edge_index, test_data.edge_weight)
    print("\nMSE Loss for all test edge_weights on held out test data = {}".format(test_loss))

    # To view all the learned weights and params
    # print(model.state_dict())

    # Save model as file
    modelDirectoryPath = dataBaseDirectory + "/saved_files/model_artefacts"
    timeStamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Use subsequently
    versioning = "v" + dataVersion.split('_', 1)[-1]  # Pull up version of data that model used to help trace the data

    file_path = '{}/model_gcnconv_{}_using_data_{}.pkl'.format(modelDirectoryPath, timeStamp, versioning)
    print("\nSaving model as pickle file : {}".format(file_path))
    torch.save(model.state_dict(), file_path)
