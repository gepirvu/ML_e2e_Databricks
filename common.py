# Databricks notebook source
pip install jq

# COMMAND ----------

from pyspark.sql.functions import when, col
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from keras import Input, layers, models, optimizers

# COMMAND ----------

# Configure logging
def configure_logging():
    """
    Configure logging for the Databricks environment.
    """
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )
    logging.info("Logging is configured.")

# COMMAND ----------

def load_dataset(dataset_path="abfss://raw@cloudinfrastg.dfs.core.windows.net/00_data_source/penguins.csv"):
    """
    Load and prepare the dataset in Databricks.

    Args:
        dataset_path (str): Path to the dataset in Databricks File System (DBFS).

    Returns:
        PySpark DataFrame: Prepared dataset.
    """
    spark = SparkSession.builder.getOrCreate()

    # Load dataset from DBFS
    data = spark.read.csv(dataset_path, header=True, inferSchema=True)
    # Handle missing values in the 'sex' column
    data = data.withColumn("sex", when(col("sex") == ".", None).otherwise(col("sex")))

    logging.info(f"Loaded dataset with {data.count()} samples.")

    return data

# COMMAND ----------

load_dataset()

# COMMAND ----------

# Build Target Transformer
def build_target_transformer():
    """
    Build a Scikit-learn transformer to preprocess the target column.

    Returns:
        OrdinalEncoder: Transformer for target encoding.
    """
    return OrdinalEncoder()

# COMMAND ----------

# Build Features Transformer
def build_features_transformer():
    """
    Build a Scikit-learn pipeline for feature preprocessing.

    Returns:
        ColumnTransformer: Transformer for feature preprocessing.
    """
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
    )

    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"),
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, make_column_selector(dtype_exclude="object")),
            ("categorical", categorical_transformer, ["island", "sex"]),
        ],
    )


# COMMAND ----------

# Build Neural Network Model
def build_model(input_shape, learning_rate=0.01):
    """
    Build and compile a neural network model.

    Args:
        input_shape (int): Number of input features.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        keras.Model: Compiled Keras model.
    """
    model = models.Sequential([
        Input(shape=(input_shape,)),
        layers.Dense(10, activation="relu"),
        layers.Dense(8, activation="relu"),
        layers.Dense(3, activation="softmax"),
    ])

    model.compile(
        optimizer=optimizers.SGD(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

# COMMAND ----------

# Initialize MLflow
def initialize_mlflow(experiment_name):
    """
    Initialize MLflow for tracking.

    Args:
        experiment_name (str): Path to the MLflow experiment.
    """
    import mlflow
    mlflow.set_experiment(experiment_name)
    logging.info(f"MLflow experiment set to: {experiment_name}")

# COMMAND ----------

def test_import():
    print("Common functions successfully imported.")