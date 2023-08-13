from pyspark import SparkConf
from pyspark.sql import SparkSession
import configparser
from mylib.logger import Log4J
from pyspark.sql.functions import size, col, when, count
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
    logger.info("Start-up configurations are....")
    logger.info(spark.sparkContext.getConf().toDebugString())

    #  Load data from the algorhythms_db --> playlist_data_tbl
    logger.info(spark.catalog.listDatabases())
    logger.info(spark.catalog.listTables("ALGORHYTHMS_DB"))

    spark.catalog.setCurrentDatabase("ALGORHYTHMS_DB")

    file_choice = 'all'
    tracktrackNodeRelationsDf = spark.table("track_track_{}_graph_node_rltns_tbl".format(file_choice))

    tracktrackReltnSize = tracktrackNodeRelationsDf.withColumn("relationships_length", size("relationships")).drop('relationships')

    # Define the buckets and ranges
    buckets = [
        (0, 10, "0-10"),
        (10, 50, "10-50"),
        (50, 100, "50-100"),
        (100, 200, "100-200"),
        (200, 500, "200-500"),
        (500, 1000, "500-1000"),
        (1000, 2000, "1000-2000"),
        (2000, 5000, "2000-5000"),
        (5000, 10000, "5000-10000"),
        (10000, float("inf"), ">10000"),
    ]

    # Create a new column 'buckets' based on the relationship_length
    tracktrackReltnSize = tracktrackReltnSize.withColumn(
        "buckets",
        when(col("relationships_length") >= buckets[9][0], buckets[9][2])
        .when(col("relationships_length") >= buckets[8][0], buckets[8][2])
        .when(col("relationships_length") >= buckets[7][0], buckets[7][2])
        .when(col("relationships_length") >= buckets[6][0], buckets[6][2])
        .when(col("relationships_length") >= buckets[5][0], buckets[5][2])
        .when(col("relationships_length") >= buckets[4][0], buckets[4][2])
        .when(col("relationships_length") >= buckets[3][0], buckets[3][2])
        .when(col("relationships_length") >= buckets[2][0], buckets[2][2])
        .when(col("relationships_length") >= buckets[1][0], buckets[1][2])
        .when(col("relationships_length") >= buckets[0][0], buckets[0][2])
        .otherwise(None)
    )

    # Group by 'buckets' and count the occurrences in each bucket
    bucket_counts = tracktrackReltnSize.groupBy("buckets").agg(count("*").alias("count"))

    csvReadDirectory = dict(config_options.items("SPARK_APP_CONFIGS")).get('spark.sql.warehouse.dir')
    bucket_counts.coalesce(1).write.parquet(csvReadDirectory+'/spotify_track_node_degree_buckets')

    logger.info("Stopping application...")
    spark.stop()
