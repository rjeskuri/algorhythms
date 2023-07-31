"""
This script is intended to generate embeddings on the entire data object using a model artefact

"""

import os
import sys
import random
import pickle
import glob
import configparser
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel

def main_gpu(rank, world_size):
    # Initialize distributed backend
    print("Rank intialized = {} :".format(rank))
    #dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=rank, world_size=world_size)

    # Set the device based on the rank
    device = torch.device("cuda:{}".format(rank))

    data = getData()
    print("Moving 'data' to GPUs.....")
    data = data.to(device)
    print("Completed moving 'data' to GPUs....Moving onto training...")

    # Gather model and set to eval mode
    model = getModel()
    model.eval()
    # Perform the forward pass on each GPU
    with torch.no_grad():
        output = model(x=data.x, edge_index=data.edge_index)

    # Gather the outputs from all GPUs
    all_outputs = [torch.zeros_like(output) for _ in range(world_size)]
    dist.all_gather(all_outputs, output)
    print("Completed gathering from all GPUs....")

    # Only the first GPU (rank 0) will save the complete output to the file
    if rank == 0:
        embeddings = torch.cat(all_outputs, dim=0)
        # Save the complete output to disk
        embeddingsDirectoryPath = dataBaseDirectory + "/saved_files/embeddings"
        embedding_file_name = '{}/{}'.format(embeddingsDirectoryPath, "embedding_file.pkl")
        print("Saving embeddings for data to : {}".format(embedding_file_name))
        with open(embedding_file_name, 'wb') as file:
            pickle.dump(embeddings, file)


def main_cpu(rank, world_size):
    # Initialize distributed backend
    print("Rank initialized = {} :".format(rank))

    master_ip = 'localhost'  # Replace with the IP address of the master node
    port = '12345'  # Replace with the desired port number
    address = 'tcp://{}:{}'.format(master_ip, port)
    dist.init_process_group(backend='gloo', init_method=address, rank=rank, world_size=world_size)

    dataBaseDirectory = '/scratch/siads699s23_class_root/siads699s23_class/shared_data/team_16_algorhythms/data/spark_table_warehouse'

    data = getData(dataBaseDirectory)
    print("Moving 'data' to CPUs.....")
    # Move the 'data' to the current rank's CPU
    data = data.to(rank)
    print("Completed moving 'data' to CPU....")

    # Gather model and set to eval mode
    model = getModel(dataBaseDirectory)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])  # Wrap the model with DDP for distributed data parallel

    # Set to non-training mode and generate embeddings
    model.eval()
    embeddings = model(x=data.x, edge_index=data.edge_index)
    print("The embeddings have the following shape: \n{}".format(embeddings.shape))

    # Save to disk
    # To-do: Assign version name or other metadata to the name of the file
    embeddingsDirectoryPath = dataBaseDirectory + "/saved_files/embeddings"
    embedding_file_name = '{}/{}'.format(embeddingsDirectoryPath, "embedding_file.pkl")
    print("Saving embeddings for data to : {}".format(embedding_file_name))
    with open(embedding_file_name, 'wb') as file:
        pickle.dump(embeddings, file)

def getData(dataBaseDirectory):
    # Load 'data' object directly from saved pickle file that was created earlier
    dataVersion = 'version_20230728_060743'
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

    with open(matching_file, 'rb') as file:
        data = pickle.load(file)
    print("The entire 'data' object has the below representation : \n{}".format(data))
    return data


def getModel(dataBaseDirectory):
    modelChoice = 'model_gcnconv_20230728_194226_using_data_v20230728_060743.pkl'
    modelDirectoryPath = dataBaseDirectory + "/saved_files/model_artefacts"
    file_path = '{}/{}'.format(modelDirectoryPath, modelChoice)

    model = GCN(num_features=data.x.size(1), hidden_size=16, embedding_size=10)
    model.load_state_dict(torch.load(file_path))
    return model


def trimDownEdges(data,fraction=0.6):
    num_elements = int(fraction * data.edge_weight.size(0)) # Generate number of edge pairs based on fraction choice
    selected_indices = random.sample(range(data.edge_index.size(1)), num_elements)

    # Use the selected indices to extract the desired elements
    edge_indices_out = data.edge_index[:, selected_indices]
    edge_weights_out = data.edge_weight[selected_indices]

    return Data(x=data.x, edge_index=edge_indices_out, edge_weight=edge_indices_out, num_nodes=data.x.size(0))

# Consider to place this into a model library to import when needed
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


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("""
        Error: Argument missing of whether this is 'distributedGPU' or 'distributedCPU' or 'non-distributed' mode. 
        """)
        sys.exit(1)  # Exit with a non-zero status code to indicate an error

    mode = sys.argv[1]
    print("Mode chosen : '{}'".format(mode))

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
    print("Is torch cuda available? : {}".format(torch.cuda.is_available()))

    if mode == 'distributedGPU':
        world_size = 2 # Set the number of GPUs available for distributed inference
        # Use torch.multiprocessing.spawn to launch a separate process for each GPU
        mp.spawn(main_gpu, args=(world_size,), nprocs=world_size)
    elif mode == 'distributedCPU':
        world_size = 2
        # Use torch.multiprocessing.spawn to launch a separate process for each CPU
        mp.spawn(main_cpu, args=(world_size,), nprocs=world_size)
    elif mode == 'non-distributed':
        data = getData(dataBaseDirectory)
        model = getModel(dataBaseDirectory)

        # Move the model and data to the GPU,if applicable
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        data = data.to(device)

        model = DataParallel(model)  # Set the model to use DataParallel for multi-CPU computation

        # Temporary step to trim down edges to be able to get some embeddings
        # data = trimDownEdges(data, fraction=0.6)

        # Set to non-training mode and generate embeddings
        model.eval()
        embeddings = model(x=data.x, edge_index=data.edge_index)
        print("The embeddings have the following shape: \n{}".format(embeddings.shape))

        # Save to disk
        # To-do: Assign version name or other metadata to name of file
        embeddingsDirectoryPath = dataBaseDirectory + "/saved_files/embeddings"
        embedding_file_name = '{}/{}'.format(embeddingsDirectoryPath, "embedding_file.pkl")
        print("Saving embeddings for data to : {}".format(embedding_file_name))
        with open(embedding_file_name, 'wb') as file:
            pickle.dump(embeddings, file)
    else:
        print("Error: Mode provided is invalid. Choose 'distributed' or 'regular'")
        sys.exit(1)
