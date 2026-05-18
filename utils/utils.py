import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def compute_agreement(sg1_path: str, sg2_path: str, id_col: str, label_col: str) -> pd.DataFrame:
    """
    Computes majority voting and consensus labels from raw annotator files
    (SG1 and SG2) using PySpark, returning a clean pandas DataFrame for GNN consumption.
    """
    print(f"--- [PySpark] Initializing Spark Session to merge {os.path.basename(sg1_path)} and {os.path.basename(sg2_path)} ---")
    
    # Initialize a local SparkSession
    spark = SparkSession.builder \
        .appName("MediaBiasConsensus") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .getOrCreate()

    try:
        # Load datasets with PySpark (semi-colon separated files)
        print("  Loading raw datasets in Spark...")
        sg1_df = spark.read.option("header", "true").option("sep", ";").csv(sg1_path)
        sg2_df = spark.read.option("header", "true").option("sep", ";").csv(sg2_path)

        # Keep only required columns and cast ID to string
        sg1_sel = sg1_df.select(F.col(id_col).cast("string").alias("article_id"), F.col(label_col))
        sg2_sel = sg2_df.select(F.col(id_col).cast("string").alias("article_id"), F.col(label_col))

        # Union annotations
        all_annotations = sg1_sel.union(sg2_sel).filter(F.col("article_id").isNotNull())

        # Count occurrences of each label per article_id
        counts_df = all_annotations.groupBy("article_id", label_col).count()

        # Count total annotations per article_id
        totals_df = all_annotations.groupBy("article_id").agg(F.count("*").alias("total_count"))

        # Define window spec to rank labels by vote count per article_id
        window_spec = Window.partitionBy("article_id").orderBy(F.col("count").desc())

        # Pick the majority label (rank 1)
        ranked_df = counts_df.withColumn("rank", F.row_number().over(window_spec))
        majority_df = ranked_df.filter(F.col("rank") == 1).drop("rank")

        # Join to compute agreement: (votes for majority label) / (total votes)
        consensus_spark_df = majority_df.join(totals_df, "article_id") \
            .withColumn("agreement", F.col("count") / F.col("total_count")) \
            .select(
                F.col("article_id"),
                F.col(label_col).alias("consensus_label"),
                F.col("agreement").cast("double")
            )

        print("  Aggregation completed in PySpark. Converting back to Pandas...")
        # Convert back to Pandas for GNN/Database pipeline compatibility
        pandas_df = consensus_spark_df.toPandas()
        
    finally:
        # Stop Spark Session to free resources
        spark.stop()
        print("  Spark Session stopped successfully.")

    return pandas_df
