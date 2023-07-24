from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list, struct, split
from pyspark.sql import functions as f  # Doing it this way to not affect standard 'sum', round' and 'abs' functions
import configparser
from mylib.logger import Log4J
import os
import sys
import pandas as pd

if __name__ == "__main__":
    # Note: when called from 'batch-3' shell script, it will always pass an argument from there with 'all' as default
    if len(sys.argv) > 1:
        file_choice = sys.argv[1]  # Get the file_choice value from the command-line argument
    else:
        file_choice = 'all'  # i.e. default file_choice is 'all'

    spark_conf = SparkConf()
    config_options = configparser.ConfigParser()
    conf_dir = os.environ.get('SPARK_CONF_DIR') or 'conf'  # Options to support Spark CLuster and local modes
    config_options.read('{}/spark.conf'.format(conf_dir))  # Load entries defined in 'spark-start' shell script
    for k, v in config_options.items("SPARK_APP_CONFIGS"):
        spark_conf.set(k, v)

    spark = SparkSession.builder  \
        .config(conf=spark_conf) \
        .enableHiveSupport()  \
        .getOrCreate()

    logger = Log4J(spark)
    logger.info("Starting Spark application....")
    logger.info("The SPARK_CONF_DIR is set to {}".format(conf_dir))

    # Load feature audio_tracks information dataframe and get list of tracks with 'NaN' features.
    csvReadDirectory = dict(config_options.items("SPARK_APP_CONFIGS")).get('spark.sql.warehouse.dir')
    spotifyCsvFile = "{}/{}" \
        .format(csvReadDirectory, 'spotifyapi_tracks_all.csv')
    logger.info("Spotify feature data CSV is : {}".format(spotifyCsvFile))
    featureDf = pd.read_csv(spotifyCsvFile)
    na_feature_tracks = featureDf[featureDf.isna().any(axis=1)]['track_uri'].values.tolist()
    logger.info("There are {} Spotify songs that pulled up with 'NaN' features from the Spotify API calls.".format(len(na_feature_tracks)))

    #  Load data from the algorhythms_db --> playlist_{}_data_tbl
    logger.info(spark.catalog.listDatabases())
    logger.info(spark.catalog.listTables("ALGORHYTHMS_DB"))
    spark.catalog.setCurrentDatabase("ALGORHYTHMS_DB")
    logger.info("Will attempt to read from playlist_{}_data_tbl".format(file_choice))
    try:
        # Attempt to read from the table
        playlistDf = spark.table("playlist_{}_data_tbl".format(file_choice))
        logger.info("Successfully read from playlist_{}_data_tbl".format(file_choice))
    except Exception as e:
        logger.error("Error: {}".format(e))
        logger.info("Stopping application due to error...")
        spark.stop()

    # Select only the required columns to process further
    playlistDf = playlistDf.select(
        "playListName",
        "numberTracks",
        "positionInPlaylist",
        "track_uri"
    )

    # Repartition to be able to reduce shuffle-sorting during joins and calculations
    playlistDf = playlistDf.repartition("playListName")

    # Remove any entries where the track_uri does not have valid features (i.e. are rows in featureDf with 'NaN' values)
    playlistDf = playlistDf.filter(~col('track_uri').isin(na_feature_tracks))

    # Generate all unique pairs using join within playlists that also prevents duplicates 'from' and 'to' pairs
    playlistSongsPairsDf = playlistDf.alias("from").join(
        playlistDf.select("playListName", "track_uri", "positionInPlaylist").alias("to"),
        (col("from.track_uri") < col("to.track_uri")) & (col("from.playListName") == col("to.playListName"))
    )

    # Calculate the score value based on difference between pairs
    # Intuition for score calculation =(a) penalize for larger playlists, (b) reward songs that are closer to each other
    songPairScoresDf = playlistSongsPairsDf.withColumn("weightedScore",
                                   f.round(1 / (f.abs(
                                       col("from.positionInPlaylist") - col("to.positionInPlaylist")) * col(
                                       "from.numberTracks")), 5))  # Round score down to 5 decimals
    songPairScoresDf = songPairScoresDf.select(
        col("from.track_uri").alias("from_track_uri"),
        col("to.track_uri").alias("to_track_uri"),
        col("weightedScore")
    )

    # For rows that share exact same combination of 'from_track_uri' and 'to_track_uri' aggregate the weightedScore
    aggregatedSongPairScoresDf = songPairScoresDf.groupBy('from_track_uri', 'to_track_uri').agg(f.sum('weightedScore').alias('totalScore'))
    # Note: The below .count() is intentionally commented out since it initiates a full DAG for the transformation, which is an expensive process
    #logger.info("Number of unique relations mined from song pairs = {}".format(aggregatedSongPairScoresDf.count()))

    # Swap column names and copy the DataFrame
    # i.e. this is done since the aggregatedSongPairScoresDf only has one row for each association
    # We will need the reverse of this as well to be able to do 'denseGroupedTracksDf' transformation later
    aggregatedSongPairScoresSwappedDf = aggregatedSongPairScoresDf.select(
        col("to_track_uri").alias("from_track_uri"),
        col("from_track_uri").alias("to_track_uri"),
        col("totalScore")
    )

    # Vertically concatenate the two DataFrames
    trackUriDf = aggregatedSongPairScoresDf.unionByName(aggregatedSongPairScoresSwappedDf, allowMissingColumns=False)

    # Optimize to dense representation
    """
    What is happening in the code below:
    a) .agg is used to group by the 'from_track_uri'
    b) Inside each selection of group, 'struct' will essentially form tuples of the to_track_uri and totalScore
    c) collect_list will bunch these tuples to a list and place this into a single field for each group.
    d) We end up naming it 'relationships'
    """
    denseGroupedTracksDf = trackUriDf.groupBy('from_track_uri').agg(
        collect_list(struct('to_track_uri', 'totalScore')).alias('relationships')
    )
    denseGroupedTracksDf = denseGroupedTracksDf.withColumnRenamed('from_track_uri', 'track')
    # Note: The below .count() is intentionally commented out since it initiates a full DAG for the transformation, which is an expensive process
    #logger.info("Number of rows in dense representation for each unique song as a row = {}".format(denseGroupedTracksDf.count()))

    # Write to Spark database as a backup incase it is needed if the Neo4J instance is shut down and new one is created
    logger.info("Starting to write to track_track_{}_graph_node_rltns_tbl.....".format(file_choice))
    denseGroupedTracksDf.write \
        .mode("overwrite") \
        .saveAsTable("track_track_{}_graph_node_rltns_tbl".format(file_choice))
    logger.info("Finished writing to track_track_{}_graph_node_rltns_tbl.....".format(file_choice))

    # Optional: Write also the 'aggregatedSongPairScoresDf' to storage just in case it is useful
    logger.info("Starting to write to track_track_{}_graph_pair_wts_tbl.....".format(file_choice))
    aggregatedSongPairScoresDf.write \
        .mode("overwrite") \
        .saveAsTable("track_track_{}_graph_pair_wts_tbl".format(file_choice))
    logger.info("Finished writing to track_track_{}_graph_pair_wts_tbl.....".format(file_choice))

    logger.info("Stopping application...")
    spark.stop()
