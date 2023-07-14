#  This file will be used to load the Spotify Million Playlist (abbreviated as 'SMP')
#  Once loaded, it will be transformed to a RDBMS-friendly view and stored in Spark DB

from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, col, udf
import configparser
from pyspark.sql.types import StringType
from mylib.logger import Log4J
from mylib.utils import hasherfunc
import os

if __name__ == "__main__":
    spark_conf = SparkConf()
    config_options = configparser.ConfigParser()
    spark_conf_dir = os.environ.get('SPARK_CONF_DIR')
    config_options.read('{}/spark.conf'.format(spark_conf_dir))  # Load entries defined in 'spark-start' shell script
    for k, v in config_options.items("SPARK_APP_CONFIGS"):
        spark_conf.set(k, v)

    #  Create the SparkSession using the retrieved configuration
    spark = SparkSession.builder \
        .config(conf=spark_conf) \
        .enableHiveSupport()  \
        .getOrCreate()

    #  Initiate logger class
    logger = Log4J(spark)

    # Get the value of the SPARK_CONF_DIR environment variable
    logger.info("Starting Spark application.....")
    logger.info("SPARK_CONF_DIR is set to: '{}' folder".format(spark_conf_dir))
    logger.info("Start-up configurations are....")
    logger.info(spark.sparkContext.getConf().toDebugString())

    #  Task 1 : Create Spark temporary table view to view each of the songs from 'tempsampledata'
    #  Read in dataset and only keep playlists information,however explode 'playlist' vertically to create playlist rows
    #  Output after this step : Several rows, one each for each playlist
    playlistDf = spark.read \
        .option("multiline", "true") \
        .option("inferSchema", "true") \
        .json(path='{}/{}'.format(spark.conf.get("spark.sql.playlist.dir")
                                  , spark.conf.get("spark.sql.playlist.pathtofiles"))) \
        .select('playlists') \
        .selectExpr("explode(playlists) as playlist")

    # Apply hash function to playlist name to ensure no duplicate names exist
    hasher_udf = udf(hasherfunc, StringType())  # Register a UDF with the 'hasherfunc' created for this purpose
    # Create a new column 'playlistName' by applying the UDF
    playlistDf = playlistDf.withColumn('playlistName',
                    hasher_udf(playlistDf['playlist.name'], playlistDf['playlist.tracks']))

    #  Leave only the playlist name, number of tracks and vertically explode/'melt' the tracks array
    #  Output after this step : Several rows, one each for each track
    playlistDf = playlistDf.select(col("playlistName"),
                                   col("playlist.num_tracks").alias("numberTracks"),
                                   explode("playlist.tracks").alias("track"))

    #  Retain only the columns of interest include JSON keys from 'track' that are of interest. Discard the rest
    #  Output after this step : Several rows, one each for each track, with only relevant information retained.
    playlistDf = playlistDf.select(
        "playlistName",
        "numberTracks",
        col("track.pos").alias("positionInPlaylist"),
        col("track.track_uri").alias("track_uri"),
        col("track.album_uri").alias("album_uri"),
        col("track.artist_uri").alias("artist_uri")
    )

    playlistTrainDf, playlistEvalDf, playlistTestDf = playlistDf.randomSplit([0.8, 0.1, 0.1],seed=699)

    logger.info("Completed split to train, evaluation and test datasets.....")
    logger.info("Train data set : A total of {} rows have been generated".format(playlistTrainDf.count()))
    logger.info("Evaluation data set : A total of {} rows have been generated".format(playlistEvalDf.count()))
    logger.info("Test data set : A total of {} rows have been generated".format(playlistTestDf.count()))

    logger.info("Moving onto saving information into Spark tables.....")

    #  Load Playlist data to Spark tables
    spark.sql("CREATE DATABASE IF NOT EXISTS ALGORHYTHMS_DB")
    spark.catalog.setCurrentDatabase("ALGORHYTHMS_DB")

    logger.info(spark.catalog.currentDatabase())
    logger.info(spark.catalog.listDatabases())

    playlistTrainDf.write \
        .mode("overwrite") \
        .saveAsTable("playlist_train_data_tbl")

    playlistEvalDf.write \
        .mode("overwrite") \
        .saveAsTable("playlist_eval_data_tbl")

    playlistTestDf.write \
        .mode("overwrite") \
        .saveAsTable("playlist_test_data_tbl")

    #  Gather the unique tracks and save to csv (Will be used in standard Python script to process)
    tracksTrainDf = playlistTrainDf.select(col('track_uri')).distinct()
    tracksTrainDf.write  \
        .csv("file://" + spark.conf.get("spark.sql.warehouse.dir") + "/unique_train_tracks_data")

    logger.info("Stopping application...")
    spark.stop()
