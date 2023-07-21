import pandas as pd 
import glob
import json
import torch
from torch_geometric.data import Data

# Load the data
feature_df = pd.read_csv('train_spotifyapi_100k_sample_20230714_042509/part-00000-43d58907-2d4d-4e73-bb70-0f2dec6d74af-c000.csv')
feature_df.dropna(inplace=True)

csvPath = "train_graph_node_reltns_100k_sample_20230714_042509/part-00000-483c9c03-4fa3-4513-8df2-e3332e1d234a-c000.csv"
with open(csvPath, 'r') as file:
    lines = file.readlines()

# Now read in the lines to a DataFrame as plain lines
rel_df = pd.DataFrame({'Text': lines[1:]}) # Reading from line 2 onwards since line 1 is labels
rel_df[['track_uri', 'relationships']] = rel_df['Text'].str.split(',', n=1, expand=True)
rel_df.drop(columns='Text',inplace=True)
rel_df['relationships'] = rel_df['relationships'].apply(lambda x: json.loads(x))
# Process the data
uri_to_index = {uri: i for i, uri in enumerate(feature_df['track_uri'])}
features = torch.tensor(feature_df.iloc[:, 1:].values, dtype=torch.float)
# create empty lists for storing edge indices and weights
edge_index = []
edge_weight = []

# iterate over each row in the dataframe
for _, row in rel_df.iterrows():
    # check if the source URI is in the subset
    if row['track_uri'] not in uri_to_index:
        continue

    # get the index of the source node
    src = uri_to_index[row['track_uri']]
    
    # parse the relationships JSON
    relationships = json.loads(row['relationships'])
    
    for relationship in relationships:
        # check if the target URI is in the subset
        if relationship['to_track_uri'] not in uri_to_index:
            continue

        # get the index of the target node
        tgt = uri_to_index[relationship['to_track_uri']]
        
        # append the edge indices and weight for both directions (src->tgt and tgt->src)
        edge_index.extend([(src, tgt), (tgt, src)])
        edge_weight.extend([relationship['totalScore'], relationship['totalScore']])

# convert to PyTorch tensors
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
edge_weight = torch.tensor(edge_weight, dtype=torch.float)

data = Data(x=features, edge_index=edge_index, edge_attr=edge_weight)
# Save processed dataframes
feature_df.to_pickle("models/data/feature_df.pkl")
rel_df.to_pickle("models/data/rel_df.pkl")

# Save PyTorch tensors
torch.save(features, "models/data/features.pt")
torch.save(edge_index, "models/data/edge_index.pt")
torch.save(edge_weight, "models/data/edge_weight.pt")
torch.save(data, "models/data/data.pt")
