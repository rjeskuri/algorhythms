"""
This script will be used to read from playlist table
to be able to gather unique combinations of spotify track URI , track name and artist name.
Once done, it will store it in memory

"""

from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from pyspark.sql.functions import explode, col, udf
import configparser
from mylib.logger import Log4J
import os


if __name__ == "__main__":
    spark_conf = SparkConf()
    config_options = configparser.ConfigParser()
    spark_conf_dir = os.environ.get('SPARK_CONF_DIR') or 'conf'
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

    playlistDf = spark.read \
        .option("multiline", "true") \
        .option("inferSchema", "true") \
        .json(path='{}/{}'.format(spark.conf.get("spark.sql.playlist.dir")
                                  , spark.conf.get("spark.sql.playlist.pathtofiles"))) \
        .select('playlists') \
        .selectExpr("explode(playlists) as playlist")

    #  Leave only the playlist name, number of tracks and vertically explode/'melt' the tracks array
    #  Output after this step : Several rows, one each for each track
    playlistDf = playlistDf.select(col('playlist.name').alias("playlistName"),
                                   col("playlist.num_tracks").alias("numberTracks"),
                                   explode("playlist.tracks").alias("track"))

    #  Retain only distinct combinations of track_uri, track_name and artist_name
    # Coalesce down to 10 file since we will use pandas in next parts to read the files
    trackUniqueGroupsDf = playlistDf.select(
        col("track.track_uri").alias("track_uri"),
        col("track.track_name").alias("track_name"),
        col("track.artist_name").alias("artist_name")
    ).distinct().coalesce(10)

    logger.info(spark.catalog.listDatabases())
    logger.info(spark.catalog.listTables("ALGORHYTHMS_DB"))
    spark.catalog.setCurrentDatabase("ALGORHYTHMS_DB")

    # Save to Spark tables
    trackUniqueGroupsDf.write \
        .mode("overwrite") \
        .saveAsTable("unique_track_uri_name_and_artist_tbl")

    logger.info("Stopping application...")
    spark.stop()
