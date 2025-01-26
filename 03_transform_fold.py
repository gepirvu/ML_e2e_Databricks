# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC # Enables autoreload; learn more at https://docs.databricks.com/en/files/workspace-modules.html#autoreload-for-python-modules
# MAGIC # To disable autoreload; run %autoreload 0

# COMMAND ----------

from common_functions import *

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pandas as pd
import numpy as np
import logging
import json

# COMMAND ----------

folds_path = "abfss://raw@cloudinfrastg.dfs.core.windows.net/02_bronze/ml_delta/penguins_cross_validation"
data_path = "abfss://raw@cloudinfrastg.dfs.core.windows.net/02_bronze/ml_delta/penguins"


logging.info(f"Loading fold indices from Delta table: {folds_path}")
folds_df = spark.read.format("delta").load(folds_path).toPandas()
data = spark.read.format("delta").load(data_path).toPandas()
folds_df.display()

fold_row = folds_df[folds_df["fold"] == fold].iloc[0]
train_indices = json.loads(fold_row["train_indices"])
test_indices = json.loads(fold_row["test_indices"])

# COMMAND ----------

 target_transformer = build_target_transformer()
 y_train = target_transformer.fit_transform(folds_df.train_indices)

# COMMAND ----------



# COMMAND ----------

data_path = "abfss://raw@cloudinfrastg.dfs.core.windows.net/02_bronze/ml_delta/penguins"
folds_path = "abfss://raw@cloudinfrastg.dfs.core.windows.net/02_bronze/ml_delta/penguins_cross_validation"
output_base_path = "abfss://raw@cloudinfrastg.dfs.core.windows.net/02_bronze/ml_delta/penguins_transformed_folds"

def transform_fold(fold, data_path, folds_path):
    """
    Transform the data to build a model during the cross-validation process.

    Args:
        fold (int): The fold number to process.
        data_path (str): Path to the dataset Delta table.
        folds_path (str): Path to the folds Delta table.

    Returns:
        tuple: Transformed training and testing datasets (X_train, y_train, X_test, y_test).
    """
    try:
        # Initialize Spark session
        spark = SparkSession.builder.getOrCreate()

        # Configure logging
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s",
            level=logging.INFO,
        )

        logging.info(f"Transforming fold {fold}...")

        # Load dataset from Delta table
        logging.info(f"Loading dataset from Delta table: {data_path}")
        data = spark.read.format("delta").load(data_path).toPandas()

        # Load fold indices from Delta table
        logging.info(f"Loading fold indices from Delta table: {folds_path}")
        folds_df = spark.read.format("delta").load(folds_path).toPandas()

       

    except Exception as e:
        logging.error(f"Error transforming fold {fold}: {e}")
        raise

# COMMAND ----------

fold = 0  # Fold number to process
data_path = "abfss://raw@cloudinfrastg.dfs.core.windows.net/02_bronze/ml_delta/penguins"
folds_path = "abfss://raw@cloudinfrastg.dfs.core.windows.net/02_bronze/ml_delta/penguins_cross_validation"

X_train, y_train, X_test, y_test = transform_fold(fold, data_path, folds_path)