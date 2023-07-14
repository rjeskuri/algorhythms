from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat, lit, col, to_json
import configparser
from mylib.logger import Log4J
import os
import datetime

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

    tracktrackNodeRelationsDf = spark.table("track_track_train_graph_node_rltns_tbl")  # Load full DF
    tracksTrainDf = spark.table("tracks_train_data_tbl") # Load full DataFrame

    # Create node relationships sample
    tracktrackNodeRelationsDf_sample = tracktrackNodeRelationsDf.sample(False, 0.1, seed=699).limit(100000)

    # IMPORTANT STEP: Cache the DataFrame. Else, the directed acyclic graph will restart after .count() or .show() in the logger calls
    # If the DAG restarts, it will make a new .sample() which will not be the same 100,000 rows it intially produced.
    tracktrackNodeRelationsDf_sample = tracktrackNodeRelationsDf_sample.cache()

    # Need to convert the schema of the 'relationships' column to be able to save to JSON
    tracktrackNodeRelationsDf_sample = tracktrackNodeRelationsDf_sample.withColumn('relationships', to_json(col('relationships')))
    logger.info("Number of records in the sample dataset for graph node relationship data = {}".format(tracktrackNodeRelationsDf_sample.count()))
    logger.info(tracktrackNodeRelationsDf_sample.show(30))

    # Create Spotify tracks sample through join
    tracksTrainDf_sample = tracktrackNodeRelationsDf_sample.alias('reltn').join(
        tracksTrainDf.alias('spotify'),
        (col("reltn.track") == col("spotify.track_uri"))
    ).select(tracksTrainDf.columns)

    # Building as cache to present re-sampling DAG actions
    tracksTrainDf_sample = tracksTrainDf_sample.cache()

    logger.info("Number of records in the sample dataset for Spotify audio-features tracks = {}".format(tracksTrainDf_sample.count()))
    logger.info(tracksTrainDf_sample.show(30))

    timeStamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    tracksTrainDf_sample.coalesce(1).write  \
        .csv(spark.conf.get("spark.sql.warehouse.dir") + '/100k_samples/train_spotifyapi_100k_sample_{}'.format(timeStamp)
             , header=True, mode='overwrite')

    tracktrackNodeRelationsDf_sample.coalesce(1).write  \
        .csv(spark.conf.get("spark.sql.warehouse.dir") + '/100k_samples/train_graph_node_reltns_100k_sample_{}'.format(timeStamp)
             , header=True, mode='overwrite')
