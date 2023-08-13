class Log4J:
    def __init__(self, spark):
        log4j = spark._jvm.org.apache.log4j  # Get log4j instance from Java Virtual machine
        root_class = "siads699.capstone.spark.app"  # From log.properties, to log to a certain higher level grouping
        app_name = spark.sparkContext.getConf().get("spark.app.name") # Get the appName defined in SparkSession
        self.logger = log4j.LogManager.getLogger(root_class + "." + app_name)  # Append with app name

    def warn(self, message):
        self.logger.warn(message)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)
