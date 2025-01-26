# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC # Enables autoreload; learn more at https://docs.databricks.com/en/files/workspace-modules.html#autoreload-for-python-modules
# MAGIC # To disable autoreload; run %autoreload 0

# COMMAND ----------

# Import the common functions notebook
from common_functions import *

# COMMAND ----------

test_import()

# COMMAND ----------

import mlflow
import logging
from pyspark.sql import SparkSession


def start_pipeline():
    """
    Start the data pipeline by loading the dataset, saving it as Delta, 
    and logging metadata to MLflow.
    """
    try:
        experiment_path = "/Users/d.t.georgian.pirvu@axpo.com/translated_experiment"
        initialize_mlflow(experiment_path)

        data_location = "abfss://raw@cloudinfrastg.dfs.core.windows.net/00_data_source/"
        delta_location = "abfss://raw@cloudinfrastg.dfs.core.windows.net/02_bronze/ml_delta/penguins"
        file_name = "penguins.csv"
        dataset_path = data_location + file_name

        logging.info(f"Loading dataset from: {dataset_path}")
        spark = SparkSession.builder.getOrCreate()
        data = load_dataset(dataset_path)

        logging.info(f"Saving dataset to Delta Lake at: {delta_location}")
        data.write.format("delta").mode("overwrite").save(delta_location)

        logging.info("Logging dataset information to MLflow.")
        pandas_data = data.toPandas()
        with mlflow.start_run(run_name="start_pipeline"):
            mlflow.log_param("dataset_path", dataset_path)
            mlflow.log_metric("data_size", data.count())
            mlflow.log_table(data=pandas_data, artifact_file="dataset.json" )  # Log table to MLflow (Databricks feature)
    except Exception as e:
        logging.error(f"Error in start_pipeline: {e}")
        raise

# COMMAND ----------

start_pipeline()