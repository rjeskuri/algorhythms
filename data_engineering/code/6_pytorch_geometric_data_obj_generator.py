"""
This code will be used to generate the entire torch-geometric.data.Data object on entire dataset.
The Spark script is designed to process in parallel transformations to enable generating ~1.7 billion edge representations
"""

import os
import sys
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
import configparser
from mylib.logger import Log4J
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import torch
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import datetime
import pickle
import joblib # To dump and load sklearn objects effectively

if __name__ == "__main__":

    spark_conf = SparkConf()
    config_options = configparser.ConfigParser()
    conf_dir = os.environ.get('SPARK_CONF_DIR') or 'conf'  # Options to support Spark CLuster and local modes
    config_options.read('{}/spark.conf'.format(conf_dir))  # Load entries defined in 'spark-start' shell script
    dataBaseDirectory = dict(config_options.items("SPARK_APP_CONFIGS")).get('spark.sql.warehouse.dir')
    for k, v in config_options.items("SPARK_APP_CONFIGS"):
        spark_conf.set(k, v)

    spark = SparkSession.builder \
        .config(conf=spark_conf) \
        .enableHiveSupport() \
        .getOrCreate()

    logger = Log4J(spark)
    logger.info("Starting Spark application....")
    logger.info("The SPARK_CONF_DIR is set to {}".format(conf_dir))

    conf_settings = spark.sparkContext.getConf().getAll()
    logger.info("Below is log of all the spark default settings:")
    for key, value in conf_settings:
        logger.info(f"{key}: {value}")

    #  Log information about database/s and set the database context
    logger.info(spark.catalog.listDatabases())
    logger.info(spark.catalog.listTables("ALGORHYTHMS_DB"))
    spark.catalog.setCurrentDatabase("ALGORHYTHMS_DB")

    # Note: when called from 'batch-6' shell script, it will always pass an argument from there with 'all' as default
    if len(sys.argv) > 1:
        file_choice = sys.argv[1]  # Get the file_choice value from the command-line argument
    else:
        file_choice = 'all'  # i.e. default file_choice is 'all'

    try:
        aggregatedSongPairScoresDf = spark.table("track_track_{}_graph_pair_wts_tbl".format(file_choice))
    except Exception as e:
        logger.error("Error: {}".format(e))
        logger.info("Stopping application due to error...")
        spark.stop()

    # Specify the path to the CSV file for Spotify data that contains ALL tracks
    spotifyCsvFile = "{}/{}".format(dataBaseDirectory, 'spotifyapi_tracks_all.csv' )
    logger.info("Spotify feature data CSV is : {}".format(spotifyCsvFile))

    # Load featureDf and drop any NaN rows
    featureDf = pd.read_csv(spotifyCsvFile)
    featureDf = featureDf.dropna()

    # OHE the 'key', 'mode' and 'time_signature' columns using scikit-learn OneHotEncoder
    non_ohe_columns = ['track_uri','danceability', 'energy','loudness','speechiness', 'acousticness', 'instrumentalness', 'liveness',
           'valence', 'tempo', 'duration_ms']
    ohe_columns = ['key', 'mode', 'time_signature']
    ohe_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    ohe_df = pd.DataFrame(ohe_encoder.fit_transform(featureDf[ohe_columns])
                          ,columns=ohe_encoder.get_feature_names_out(ohe_columns))

    # Generate a new 'featureOHEDf' with non-OHE and OHE columns concatenated
    featureOHEDf = pd.concat([featureDf[non_ohe_columns].reset_index(drop=True) # Only the non_ohe_columns on left. Reset index to prevent issues in concat
                           , ohe_df.reset_index(drop=True)], axis=1) # The ohe columns on right. Reset index to prevent issues in concat
    featureOHEDf.set_index('track_uri',inplace=True)

    # Min-Max scale the entire dataframe
    minMaxScaler = MinMaxScaler()
    for col in featureOHEDf.columns:
        column_data = featureOHEDf[col].values.reshape(-1, 1) # Reshape the column to a 2D array to fit the scaler
        featureOHEDf[col] = minMaxScaler.fit_transform(column_data) # Fit and transform the column with Min-Max scaling

    # Generate the trackIndexDict that will be used to help create 'edge_index' references
    trackIndexDict = {node:index for index,node in enumerate(featureOHEDf.index)}

    # User-Defined Function (UDF) to map track_uri to its index in trackIndexDict
    def map_track_uri_to_index(track_uri):
        return trackIndexDict[track_uri]

    # Register the UDF for Spark to be able to use
    udf_map_track_uri_to_index = udf(map_track_uri_to_index, IntegerType())

    # Generate new columns for the indexes
    aggregatedSongPairScoresDf = aggregatedSongPairScoresDf.withColumn('from_track_uri_idx',
                                                                       udf_map_track_uri_to_index('from_track_uri'))
    aggregatedSongPairScoresDf = aggregatedSongPairScoresDf.withColumn('to_track_uri_idx',
                                                                       udf_map_track_uri_to_index('to_track_uri'))

    from_track_uri_np = np.array(aggregatedSongPairScoresDf.select("from_track_uri_idx").collect()).squeeze()
    to_track_uri_np = np.array(aggregatedSongPairScoresDf.select("to_track_uri_idx").collect()).squeeze()
    edge_indices_np = np.array([from_track_uri_np, to_track_uri_np], dtype=np.int64)
    edge_weights = np.array(aggregatedSongPairScoresDf.select("totalScore").collect()).squeeze()

    # Convert numpy arrays to torch tensors
    edge_index = torch.tensor(edge_indices_np, dtype=torch.long).contiguous()
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)

    # IMPORTANT STEP:
    # Duplicate the entries in edge_index (reversed) to ensure undirected edges.Also duplicate edge_weight to match the edge_index.
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_weight = torch.cat([edge_weight, edge_weight], dim=0)

    # Since featureOHEDf is an ordered DataFrame that was used to generate 'trackIndexDict'
    # , we can be sure of using featureOHEDf.values as array in same indexed order to define node_features
    node_features = torch.tensor(featureOHEDf.values, dtype=torch.float)

    # Create the graph data using torch-geometric
    data = Data(x=node_features, edge_index=edge_index, edge_weight=edge_weight, num_nodes=node_features.size(0))

    logger.info("The 'data' object generated is : \n{}\n".format(data))
    logger.info("Is the 'data' object undirected? : {}".format(data.is_undirected()))

    """
    Save files to disk that can be used for training and inference
    """
    timeStamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Use subsequently
    versionDirectoryPath = os.path.join(dataBaseDirectory
                                        + "/saved_files/data_representations", 'version_{}'.format(timeStamp))
    os.makedirs(versionDirectoryPath, exist_ok=True)
    print("Files from this job will be saved to : {}".format(versionDirectoryPath))

    # Save 'data' and 'trackIndexDict' to storage for future use
    file_trackIndexDict = '{}/dict_of_track_index_location_{}.pkl'.format(versionDirectoryPath, timeStamp)
    file_data = '{}/data_obj_{}_nodes_{}_edges_{}.pkl'.format(
        versionDirectoryPath, data.num_nodes, data.edge_index.size(1), timeStamp)

    with open(file_trackIndexDict, 'wb') as file:
        pickle.dump(trackIndexDict, file)
    with open(file_data, 'wb') as file:
        pickle.dump(data, file)

    # Save the 'ohe_encoder' and 'minMaxScaler' to a 'joblib' each  file in 'versionDirectoryPath'
    # Load in model evaluation time to be able to use it in evaluation time.
    file_ohe_encoder = '{}/ohe_encoder_{}.joblib'.format(versionDirectoryPath, timeStamp)
    file_min_max_enc = '{}/minmax_scaler_{}.joblib'.format(versionDirectoryPath, timeStamp)

    joblib.dump(ohe_encoder, file_ohe_encoder)
    joblib.dump(minMaxScaler, file_min_max_enc)

    logger.info("Stopping application after successful completion...")
    spark.stop()