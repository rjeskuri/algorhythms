from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
import configparser
from mylib.logger import Log4J
import os
import sys


if __name__ == "__main__":
    # Note: when called from 'batch-2.2' shell script, it will always pass an argument from there with 'all' as default
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
    try:
        tracksDf = spark.read.csv(csvReadDirectory + "/spotifyapi_tracks_{}.csv".format(file_choice)
                                       , schema=tracksSchema, header=True)
        logger.info("The number of entries in the Dataframe is : {}".format(tracksDf.count()))
        # Write to table in the Spark Database
        tracksDf.write \
            .mode("overwrite") \
            .saveAsTable("tracks_{}_data_tbl".format(file_choice))
    except Exception as e:
        logger.error("Error: {}".format(e))
        logger.info("Stopping application due to error...")
        spark.stop()

    logger.info("Stopping application after successful completion...")
    spark.stop()
