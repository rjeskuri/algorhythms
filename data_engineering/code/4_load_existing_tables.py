from pyspark import SparkConf
from pyspark.sql import SparkSession
import configparser
from mylib.logger import Log4J
import os

if __name__ == "__main__":
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
    logger.info("SPARK_CONF_DIR is set to: '{}' folder".format(os.environ.get('SPARK_CONF_DIR')))

    #  Load data from the algorhythms_db --> playlist_data_tbl
    logger.info(spark.catalog.listDatabases())
    logger.info(spark.catalog.listTables("ALGORHYTHMS_DB"))

    spark.catalog.setCurrentDatabase("ALGORHYTHMS_DB")
    tracksTrainDf = spark.table("tracks_train_data_tbl")
    playlistTrainDf = spark.table("playlist_train_data_tbl")
    tracktrackNodeRelationsDf = spark.table("track_track_train_graph_node_rltns_tbl")
    tracktrackWtsTrainDf = spark.table("track_track_train_graph_pair_wts_tbl")

    logger.info("A total of {} rows of playlist songs data is present in train dataset".format(playlistTrainDf.count()))
    playlistTrainDf.show(30)

    logger.info("Number of tracks from train dataset with Spotify audio-features = {}".format(tracksTrainDf.count()))
    tracksTrainDf.show(30)

    logger.info("Number of rows from train dataset in dense representation for each unique song as a row= {}".format(tracktrackNodeRelationsDf.count()))
    tracktrackNodeRelationsDf.show(30)

    logger.info("Number of rows from train dataset in pair wise representation for each unique song-pair as a row= {}".format(tracktrackWtsTrainDf.count()))
    tracktrackWtsTrainDf.show(30)

    logger.info("Stopping application...")
    spark.stop()
