import os
import sys
from pyspark.sql import SparkSession

def get_spark_session(app_name: str = "Urban_Mobility_Analytics") -> SparkSession:
    """
    Initializes and returns a configured SparkSession for local execution.
    Configured with appropriate memory and shuffle settings for laptop execution.
    """
    # Ensure worker Python matches driver Python environment
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.driver.maxResultSize", "2g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    return spark

if __name__ == "__main__":
    spark = get_spark_session("Spark_Session_Test")
    print(f"SparkSession successfully initialized. Version: {spark.version}")
    spark.stop()
