"""
Instructions:
This script is NOT to be run in the form of a Spark application to prevent having several worker threads.
This is because several worker threads may not respect the rate limit throttling in use.
If running in Great Lakes, UMich environment , launch a 'Jupyter Notebook' environment --> Open a terminal session
1. Since spotipy package is not installed in environment, ensure it is installed through a package manager.
2. Once done, run the script in Python command line terminal

Prompts:
Prompt 1 = File choice to be used. Default is 'all' if nothing is specified
Prompt 2 = Starting row number of the combined dataframe to start query to SpotifyAPI from
Prompt 3 = Number of records to pull up in this run (Empty means all records till end)
"""

import os
import sys
import time
import numpy as np
import glob
import pandas as pd
import configparser
from mylib.utils import groupListByBatch
from mylib.spotifyapi import getinfotracks, getinfoaudiofeatures
from spotipy.exceptions import SpotifyException
pd.set_option('display.max_columns', None)


def track_action(track,idx,cnt,tracks_uri):
    if isinstance(track, dict):
        return [track['uri'], float(track.get('danceability', np.nan)),
                             float(track.get('energy', np.nan)), track.get('key', -1),
                             float(track.get('loudness', np.nan)), track.get('mode', -1),
                             float(track.get('speechiness', np.nan)), float(track.get('acousticness', np.nan)),
                             float(track.get('instrumentalness', np.nan)), float(track.get('liveness', np.nan)),
                             float(track.get('valence', np.nan)), float(track.get('tempo', np.nan)),
                             float(track.get('duration_ms', np.nan)), track.get('time_signature', -1)]
    else:  # i.e. if the NoneType appears, pass back an empty list with nan values in the 13 spots
        print("Batch {} has one or more entries with NoneType".format(cnt+1))
        return [tracks_uri[idx]] + [np.nan,np.nan,-1,np.nan,-1,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,-1]


# Make API requests to Spotify with rate limit handling
def throttled_tracks_request(tracks_uri_list, tracksDf, tracksSchema):
    batch_size = 100  # Max. batch size to pass to 'several tracks' API
    # Iterate over the grouped items
    for cnt,batch in enumerate(groupListByBatch(tracks_uri_list, batch_size)):
        tracks_uri = [item for item in batch if item is not None]  # Last batch might have some None values to filter
        if len(tracks_uri) == 0:
            continue
        print("Batch {} is being started.".format(cnt + 1))
        return_flag = False  # Initially set to false for the batch
        batch_retry_cnt = 0  # Initially set to 0
        break_flag = False
        while not return_flag:  # While loop to ensure retries if rate limit is reached
            try:
                time.sleep(1.0)  # Half second wait time in-built between calls to prevent hitting rate limit early
                response = getinfoaudiofeatures(tracks_uri)  # Make API call to get result
                tempRows = [track_action(track,idx,cnt,tracks_uri) for idx,track in enumerate(response)]
                tempDf = pd.DataFrame(data=tempRows, columns=tracksSchema.keys()).astype(tracksSchema)
                tracksDf = pd.concat([tracksDf, tempDf],axis=0)  # Join temp result with existing Dataframe
                return_flag = True  # Will reach here only if no exception. Set batch to pass to leave while loop
                batch_retry_cnt = 0  # Reset batch retry count to 0
                print("Batch {} has been processed successfully.".format(cnt+1))
                if cnt % 50 == 0:
                    print("At Batch {}. Pushing data to temp storage 'temp_tracks_{}_spotifyapi.csv'".format(cnt + 1,file_choice))
                    tracksDf.to_csv(csvReadDirectory + "/temp_tracks_{}_spotifyapi.csv".format(file_choice), index=False)

            except SpotifyException as e:
                if e.http_status == 429:
                    if batch_retry_cnt == 5:
                        break_flag = True
                        print("Maximum attempts exceeded for this batch. Exiting....")
                        break
                    #  Handle rate limit exceeded error
                    print(e.headers)
                    reset_time = int(e.headers.get('Retry-After', 1))  # Get reset time from header or default as 1.
                    wait_time = reset_time + 2  # Wait for the reset time plus some additional buffer time
                    time.sleep(wait_time)
                    print("Rate limit exceeded. Waiting for {} seconds.".format(wait_time))
                    batch_retry_cnt += 1
                else:
                    # Handle other Spotify API errors
                    print("Spotify API error:", e)

            if break_flag:  # Break out of the 2nd loop as well
                print("Batch {} has FAILED due to 5 attempts for retries failing. Moving to next batch....".format(cnt + 1))
                break

    return tracksDf


if __name__ == "__main__":
    config_options = configparser.ConfigParser()
    conf_dir = os.environ.get('SPARK_CONF_DIR') or 'conf'  # Options to support Spark CLuster and local modes
    config_options.read('{}/spark.conf'.format(conf_dir))  # Load entries defined in 'spark-start' shell script
    csvReadDirectory = dict(config_options.items("SPARK_APP_CONFIGS")).get('spark.sql.warehouse.dir')

    #  Sample a few tracks to be able to send up to the 'several tracks' API
    #  sample_values = playlistDf.select('track_uri').sample(False, 0.1).limit(1010).collect()
    #  track_uri_list = [row['track_uri'] for row in sample_values]

    # From command-line, provide user ability to specify 'all', 'train' or any other folder that is format 'unique_<argument>_tracks_data'
    if len(sys.argv) > 1:
        file_choice = sys.argv[1] # Get the file_choice value from the command-line argument
    else:
        # Use the default sample_size value
        file_choice = 'all' # i.e. default size

    # Get a list of CSV files in the folder
    track_csv_files = glob.glob('{}/*.csv'.format(csvReadDirectory + "/unique_tracks" + "/unique_{}_tracks_data".format(file_choice)))
    # Read CSV files into separate DataFrames for each CSV.
    # Since data starts from 1st file in CSV produced by Spark, we will need to assign header
    dataframesList = [pd.read_csv(file, header=None, names=['track_uri']) for file in track_csv_files]
    combined_tracks_df = pd.concat(dataframesList,axis=0)  # Combine DataFrames into a single DataFrame
    # Add a column to be able to track the position of the rows, just in case it is needed
    combined_tracks_df['pos'] = list(range(1, combined_tracks_df.shape[0] + 1))

    # Note: sys.argv[1] is always the program name itself.
    startPos = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    endPos = int(sys.argv[3]) + startPos if len(sys.argv) > 3 else len(combined_tracks_df)

    print("Starting position with respect to Dataframe to begin queries = {}".format(startPos))
    print("End position with respect to Dataframe to end queries = {}".format(endPos))
    queryRangeDf = combined_tracks_df.iloc[startPos:endPos]

    track_uri_list = queryRangeDf['track_uri'].values  # Get track_uri in form of list
    print("Number of tracks to query (in batches of 100) = {}".format(len(track_uri_list)))

    '''
    # Define the schema for the JSON fields in Spark
    tracksSchema = StructType([
        StructField("track_uri", StringType(), nullable=False),
        StructField("danceability", DoubleType(), nullable=True),
        StructField("energy", DoubleType(), nullable=True),
        StructField("key", IntegerType(), nullable=True),
        StructField("loudness", DoubleType(), nullable=True),
        StructField("mode", IntegerType(), nullable=True),
        StructField("speechiness", DoubleType(), nullable=True),
        StructField("acousticness", DoubleType(), nullable=True),
        StructField("instrumentalness", DoubleType(), nullable=True),
        StructField("liveness", DoubleType(), nullable=True),
        StructField("valence", DoubleType(), nullable=True),
        StructField("tempo", DoubleType(), nullable=True),
        StructField("duration_ms", DoubleType(), nullable=True),
        StructField("time_signature", IntegerType(), nullable=True)
    ])
    '''

    # Define the data types for each column in Pandas
    tracksSchema = {
        "track_uri": str,
        "danceability": float,
        "energy": float,
        "key": int,
        "loudness": float,
        "mode": int,
        "speechiness": float,
        "acousticness": float,
        "instrumentalness": float,
        "liveness": float,
        "valence": float,
        "tempo": float,
        "duration_ms": float,
        "time_signature": int
    }

    # Create an empty DataFrame with the defined columns and assign schema
    tracksDf = pd.DataFrame(columns=tracksSchema.keys()).astype(tracksSchema)

    #  Throttle responses using 'Several tracks' API and  write to Spark tracks DataFrame
    tracksDf = throttled_tracks_request(track_uri_list, tracksDf, tracksSchema)  # Spark code
    print("Number of tracks in the tracksDf = {}".format(tracksDf['track_uri'].count()))
    print(tracksDf.head())

    tracksDf.to_csv(csvReadDirectory + "/spotifyapi_tracks_{}.csv".format(file_choice), index=False)
