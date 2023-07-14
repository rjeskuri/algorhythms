from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list, struct, split
from pyspark.sql import functions as f  # Doing it this way to not affect standard 'sum', round' and 'abs' functions
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
    logger.info("The SPARK_CONF_DIR is set to {}".format(conf_dir))

    #  Load data from the algorhythms_db --> playlist_data_tbl
    logger.info(spark.catalog.listDatabases())
    logger.info(spark.catalog.listTables("ALGORHYTHMS_DB"))
    spark.catalog.setCurrentDatabase("ALGORHYTHMS_DB")
    playlistTrainDf = spark.table("playlist_train_data_tbl")

    # Select only the required columns to process further
    playlistTrainDf = playlistTrainDf.select(
        "playListName",
        "numberTracks",
        "positionInPlaylist",
        "track_uri"
    )

    # Repartition to be able to reduce shuffle-sorting during joins and calculations
    playlistTrainDf = playlistTrainDf.repartition("playListName")

    # Generate all unique pairs using join within playlists that also prevents duplicates 'from' and 'to' pairs
    playlistSongsPairsDf = playlistTrainDf.alias("from").join(
        playlistTrainDf.select("playListName", "track_uri", "positionInPlaylist").alias("to"),
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
    logger.info("Number of unique relations mined from song pairs = {}".format(aggregatedSongPairScoresDf.count()))

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
    logger.info("Number of rows in dense representation for each unique song as a row = {}".format(denseGroupedTracksDf.count()))

    #  Neo4J Boiler plate code for Spark Connector
    '''
    # Set the Neo4j connection properties
    neo4j_connection = {
        "url": "bolt://localhost:7687",
        "user": "neo4j",
        "password": "password",
        "authentication.type": "basic"
    }

    # Write the denseGroupedTracksDf DataFrame to Neo4j
    denseGroupedTracksDf.write.format("org.neo4j.spark.DataSource") \
        .option("url", neo4j_connection["url"]) \
        .option("authentication.type", neo4j_connection["authentication.type"]) \
        .option("user", neo4j_connection["user"]) \
        .option("password", neo4j_connection["password"]) \
        .option("labels", ":Track") \
        .option("relationship", "RELATED_TO") \
        .option("relationship.save.strategy", "keys") \
        .mode("Overwrite") \
        .save()
    '''

    # Write to Spark database as a backup incase it is needed if the Neo4J instance is shut down and new one is created
    denseGroupedTracksDf.write \
        .mode("overwrite") \
        .saveAsTable("track_track_train_graph_node_rltns_tbl")

    # Optional: Write also the 'aggregatedSongPairScoresDf' to storage just in case it is useful
    aggregatedSongPairScoresDf.write \
        .mode("overwrite") \
        .saveAsTable("track_track_train_graph_pair_wts_tbl")

    logger.info("Stopping application...")
    spark.stop()
