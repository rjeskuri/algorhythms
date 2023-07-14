from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
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
    csvReadDirectory = dict(config_options.items("SPARK_APP_CONFIGS")).get('spark.sql.warehouse.dir')

    spark = SparkSession.builder  \
        .config(conf=spark_conf) \
        .enableHiveSupport()  \
        .getOrCreate()

    logger = Log4J(spark)
    logger.info("Starting Spark application....")
    logger.info("The SPARK_CONF_DIR is set to {}".format(conf_dir))

    #  Log information about database/s and set the database context
    logger.info(spark.catalog.listDatabases())
    logger.info(spark.catalog.listTables("ALGORHYTHMS_DB"))
    spark.catalog.setCurrentDatabase("ALGORHYTHMS_DB")

    # Define the schema for the JSON fields (i.e. Spark Schema corresponding to pandas schema in 2.1)
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

    # Create the DataFrame by reading the CSV with the defined schema
    tracksTrainDf = spark.read.csv(csvReadDirectory + "/spotifyapi_tracks_train.csv"
                                   , schema=tracksSchema, header=True)
    logger.info("The number of entries in the Dataframe is : {}".format(tracksTrainDf.count()))

    # Write to table in the Spark Database
    tracksTrainDf.write \
        .mode("overwrite") \
        .saveAsTable("tracks_train_data_tbl")

    logger.info("Stopping application...")
    spark.stop()
