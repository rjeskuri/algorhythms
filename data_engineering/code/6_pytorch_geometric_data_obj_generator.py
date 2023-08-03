"""
This code will be used to generate the entire torch-geometric.data.Data object on entire dataset.
"""

import os
import datetime
import numpy as np
import glob
import pandas as pd
import configparser
from tqdm import tqdm
import pickle
import joblib
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import torch
from torch_geometric.data import Data


if __name__ == "__main__":
    config_options = configparser.ConfigParser()
    conf_dir = os.environ.get('SPARK_CONF_DIR') or 'conf'  # Options to support Spark CLuster and local modes
    config_options.read('{}/spark.conf'.format(conf_dir))  # Load entries defined in 'spark-start' shell script
    dataBaseDirectory = dict(config_options.items("SPARK_APP_CONFIGS")).get('spark.sql.warehouse.dir')

    file_choice = 'all'
    pair_wts_parquet_files = glob.glob('{}/*.parquet'.format(
        dataBaseDirectory + "/algorhythms_db.db" + "/track_track_{}_graph_pair_wts_tbl".format(file_choice)))
    print("Number of parquet files for pair-wise relations were found to be : {}".format(len(pair_wts_parquet_files)))

    unique_track_uri_name_and_artists_files = glob.glob(
        '{}/algorhythms_db.db/unique_track_uri_name_and_artist_tbl/*.parquet'.format(dataBaseDirectory))
    print("Number of parquet files for distinct tracks with their track name and artist name : {}".format(
        len(unique_track_uri_name_and_artists_files)))

    # Specify the path to the CSV file for Spotify data that contains ALL tracks
    spotifyCsvFile = "{}/{}" \
        .format(dataBaseDirectory, 'spotifyapi_tracks_all.csv')
    print("Spotify feature data CSV is : {}".format(spotifyCsvFile))

    featureDf = pd.read_csv(spotifyCsvFile)
    na_feature_tracks = featureDf[featureDf.isna().any(axis=1)]['track_uri'].values.tolist()
    print("There are {} Spotify songs that pulled up with 'NaN' features from the Spotify API calls.".format(
        len(na_feature_tracks)))

    # Drop any NaN values, since we don't need them and
    featureDf = featureDf.dropna()

    # OHE the 'key', 'mode' and 'time_signature' columns using scikit-learn OneHotEncoder
    non_ohe_columns = ['track_uri', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                       'instrumentalness', 'liveness',
                       'valence', 'tempo', 'duration_ms']
    ohe_columns = ['key', 'mode', 'time_signature']
    ohe_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe_df = pd.DataFrame(ohe_encoder.fit_transform(featureDf[ohe_columns])
                          , columns=ohe_encoder.get_feature_names_out(ohe_columns))

    # Generate a new 'featureOHEDf' with non-OHE and OHE columns concatenated
    featureOHEDf = pd.concat([featureDf[non_ohe_columns].reset_index(drop=True)
                              # Only the non_ohe_columns on left. Reset index to prevent issues in concat
                                 , ohe_df.reset_index(drop=True)],
                             axis=1)  # The ohe columns on right. Reset index to prevent issues in concat
    featureOHEDf.set_index('track_uri', inplace=True)

    # Min-Max scale the entire dataframe
    minMaxScaler = MinMaxScaler()

    '''
    for col in tqdm(featureOHEDf.columns, desc="Min Max scaling across columns..."):
        column_data = featureOHEDf[col].values.reshape(-1, 1)  # Reshape the column to a 2D array to fit the scaler
        featureOHEDf[col] = minMaxScaler.fit_transform(column_data)  # Fit and transform the column with Min-Max scaling
    '''
    featureOHEMinMaxDf = pd.DataFrame(minMaxScaler.fit_transform(featureOHEDf),columns=featureOHEDf.columns,index=featureOHEDf.index)

    # Generate the trackIndexDict that will be used to help create 'edge_index' references
    trackIndexDict = {node: index for index, node in enumerate(featureOHEMinMaxDf.index)}
    trackNodeFeatures = featureOHEMinMaxDf.values

    # Concatenate all the parquet file contents together to form a single pandas dataframe
    unique_tracks_name_artist_df = pd.concat([pd.read_parquet(file) for file in unique_track_uri_name_and_artists_files]
                                             , ignore_index=True)

    # To ensure that indexes in 'trackIndexDict' are consistently used to join to create 'indexTrackDict'
    # ,make a temporary Dataframe that preserves these indexes.This way no unintended effects will be seen with any index resets
    joinHelperDf = pd.DataFrame(list(trackIndexDict.items()), columns=['track_uri', 'track_index'])

    # Create a dataframe that has all the desired columns, index, track_uri, track_name, artist_name
    indexTrackDictDf = joinHelperDf.merge(unique_tracks_name_artist_df, on='track_uri', how='inner').set_index(
        'track_index')

    # Reverse the trackIndexDict since during inference we will need index to track lookup
    # Produces a dictionary of index: (track_uri,track_name,artist_name)
    # Note : The tuple value is actually a numpy.record, but can be called using index like a regular tuple
    indexTrackDict = dict(zip(indexTrackDictDf.index, indexTrackDictDf.to_records(index=False)))

    # Initialize to empty numpy array
    edge_index_src_np = np.empty(0, dtype=int)
    edge_index_dst_np = np.empty(0, dtype=int)
    edge_weights_np = np.empty(0, dtype=int)

    for parquet_file in tqdm(pair_wts_parquet_files):
        pairWtsDf = pd.read_parquet(parquet_file)
        edge_index_src_np = np.concatenate((edge_index_src_np, pairWtsDf['from_track_uri'].map(trackIndexDict).values))
        edge_index_dst_np = np.concatenate((edge_index_dst_np, pairWtsDf['to_track_uri'].map(trackIndexDict).values))
        edge_weights_np = np.concatenate((edge_weights_np, pairWtsDf['totalScore'].values))

    # Get the edge_indices_np by stacking source and destination arrays on top
    edge_indices_np = np.array([edge_index_src_np, edge_index_dst_np], dtype=np.int64)
    # Convert numpy arrays to torch tensors
    edge_index = torch.tensor(edge_indices_np, dtype=torch.long).contiguous()
    edge_weight = torch.tensor(edge_weights_np, dtype=torch.float)

    # IMPORTANT STEP: Duplicate the entries in edge_index (reversed) to ensure undirected edges.Also duplicate edge_weight to match the edge_index.
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_weight = torch.cat([edge_weight, edge_weight], dim=0)

    # Since featureOHEMinMaxDf is an ordered DataFrame that was used to generate 'trackIndexDict'
    # , we can be sure of using featureOHEMinMaxDf.values as array in same indexed order to define node_features
    node_features = torch.tensor(trackNodeFeatures, dtype=torch.float)

    # Create the graph data using torch-geometric
    data = Data(x=node_features, edge_index=edge_index, edge_weight=edge_weight, num_nodes=node_features.size(0))

    print("The 'data' object generated is : \n{}\n".format(data))
    # Commenting out below code since it takes a lot of time for a large dataset:
    # print("Is the 'data' object undirected? : {}".format(data.is_undirected()))

    """
    Save files to disk that can be used for training and inference
    """

    timeStamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Use subsequently
    versionDirectoryPath = os.path.join(dataBaseDirectory
                                        + "/saved_files/data_representations", 'version_{}'.format(timeStamp))
    os.makedirs(versionDirectoryPath, exist_ok=True)
    print("Files from this job will be saved to : {}".format(versionDirectoryPath))

    # Save 'data' and 'trackIndexDict' to storage for future use
    file_trackIndexDict = '{}/dict_of_track_to_index_{}.pkl'.format(versionDirectoryPath, timeStamp)
    file_indexTrackDict = '{}/dict_of_index_to_track_uri_and_names_{}.pkl'.format(versionDirectoryPath, timeStamp)
    file_data = '{}/data_obj_{}_nodes_{}_edges_{}.pkl'.format(
        versionDirectoryPath, data.num_nodes, data.edge_index.size(1), timeStamp)
    file_node_features = '{}/all_node_features_{}.pkl'.format(versionDirectoryPath, timeStamp)

    with open(file_trackIndexDict, 'wb') as file:
        pickle.dump(trackIndexDict, file)
    with open(file_indexTrackDict, 'wb') as file:
        pickle.dump(indexTrackDict, file)
    with open(file_data, 'wb') as file:
        pickle.dump(data, file)
    with open(file_node_features, 'wb') as file:
        pickle.dump(node_features, file)

    # Save the 'ohe_encoder' and 'minMaxScaler' to a 'joblib' each  file in 'versionDirectoryPath'
    # Load in model evaluation time to be able to use it in evaluation time.
    file_ohe_encoder = '{}/ohe_encoder_{}.joblib'.format(versionDirectoryPath, timeStamp)
    file_min_max_enc = '{}/minmax_scaler_{}.joblib'.format(versionDirectoryPath, timeStamp)

    joblib.dump(ohe_encoder, file_ohe_encoder)
    joblib.dump(minMaxScaler, file_min_max_enc)
