# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC # Enables autoreload; learn more at https://docs.databricks.com/en/files/workspace-modules.html#autoreload-for-python-modules
# MAGIC # To disable autoreload; run %autoreload 0

# COMMAND ----------

from common_functions import *

# COMMAND ----------

from pyspark.sql import SparkSession
from sklearn.model_selection import KFold
import pandas as pd
import logging


delta_source_location = "abfss://raw@cloudinfrastg.dfs.core.windows.net/02_bronze/ml_delta/penguins"
output_delta_path = "abfss://raw@cloudinfrastg.dfs.core.windows.net/02_bronze/ml_delta/penguins_cross_validation"
def cross_validation_pipeline(delta_source_location, output_delta_path):
    try:
        # Initialize Spark session
        spark = SparkSession.builder.getOrCreate()

        # Load dataset from Delta Lake
        logging.info(f"Loading data from Delta Lake: {delta_source_location}")
        data = spark.read.format("delta").load(delta_source_location).toPandas()

        # Prepare cross-validation folds
        logging.info("Preparing cross-validation folds...")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        folds = [{"fold": i, "train_indices": train.tolist(), "test_indices": test.tolist()} 
                 for i, (train, test) in enumerate(kf.split(data))]

        # Convert folds to Spark DataFrame
        folds_df = spark.createDataFrame(pd.DataFrame(folds))

        # Save folds as Delta table
        logging.info(f"Saving cross-validation folds as Delta table to: {output_delta_path}")
        folds_df.write.format("delta").mode("overwrite").save(output_delta_path)

    except Exception as e:
        logging.error(f"Error in cross-validation pipeline: {e}")
        raise

# COMMAND ----------

cross_validation_pipeline(delta_source_location, output_delta_path)

# COMMAND ----------

       data = spark.read.format("delta").load(output_delta_path).display()