# Databricks notebook source
import logging
import os
from typing import Dict, Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from tensorflow import keras
from tensorflow.keras import layers, optimizers
import keras_tuner as kt  #

# COMMAND ----------

# Configure logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)


# COMMAND ----------

mlflow.set_experiment("/Users/d.t.georgian.pirvu@axpo.com/penguin-classification-tuning")

# COMMAND ----------

def load_data():
    """Load the penguins dataset."""
    # Define schema
    schema = StructType([
        StructField("species", StringType(), True),
        StructField("island", StringType(), True),
        StructField("culmen_length_mm", DoubleType(), True),
        StructField("culmen_depth_mm", DoubleType(), True),
        StructField("flipper_length_mm", DoubleType(), True),
        StructField("body_mass_g", DoubleType(), True),
        StructField("sex", StringType(), True)
    ])
    
    # Read from storage
    data_location = "abfss://raw@cloudinfrastg.dfs.core.windows.net/00_data_source/"
    file_name = "penguins.csv"
    dataset_path = data_location + file_name
    
    logging.info(f"Loading dataset from: {dataset_path}")
    data = spark.read.csv(dataset_path, header=True, schema=schema)
    
    # Convert to pandas
    data = data.toPandas()
    
    # Clean data
    # Replace '.' with NaN in the sex column
    data["sex"] = data["sex"].replace(".", np.nan)
    
    # Drop rows with missing values
    data = data.dropna()
    
    return data

# COMMAND ----------

def build_model(hp: kt.HyperParameters, input_shape: int) -> keras.Model:
    """Build a tunable model with hyperparameters."""
    # Tunable parameters
    learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
    units = hp.Int("units", 32, 256, step=32)
    dropout_rate = hp.Float("dropout_rate", 0.1, 0.5, step=0.1)
    
    # Model architecture
    model = keras.Sequential([
        layers.Dense(units, activation="relu", input_shape=(input_shape,)),
        layers.Dropout(dropout_rate),
        layers.Dense(units // 2, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(3, activation="softmax")
    ])
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

# COMMAND ----------

def build_transformers():
    """Build the feature and target transformers."""
    # Feature transformer
    numeric_features = [
        "culmen_length_mm",
        "culmen_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]
    categorical_features = ["island", "sex"]
    
    features_transformer = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_features),
        ]
    )
    
    # Target transformer
    target_transformer = ColumnTransformer(
        transformers=[("species", OneHotEncoder(sparse_output=False), [0])]
    )
    
    return features_transformer, target_transformer

# COMMAND ----------

class MLflowTuner(kt.RandomSearch):
    """Custom tuner that logs results to MLflow."""
    
    def on_trial_end(self, trial):
        """Log trial results to MLflow."""
        with mlflow.start_run(run_name=f"trial-{trial.trial_id}", nested=True):
            # Log hyperparameters
            for hp, value in trial.hyperparameters.values.items():
                mlflow.log_param(hp, value)
            
            # Log metrics
            for metric, value in trial.metrics.metrics.items():
                if len(value.values) > 0:
                    mlflow.log_metric(metric, value.values[-1])
        
        super().on_trial_end(trial)

# COMMAND ----------

def run_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    """Run hyperparameter tuning with MLflow tracking."""
    
    with mlflow.start_run(run_name="hyperparameter-tuning") as run:
        tuner = MLflowTuner(
            hypermodel=lambda hp: build_model(hp, X_train.shape[1]),
            objective="val_accuracy",
            max_trials=10,
            executions_per_trial=2,
            directory="/dbfs/tmp/keras_tuner",
            project_name="penguin_classification"
        )
        
        # Log tuning parameters
        mlflow.log_params({
            "max_trials": 10,
            "executions_per_trial": 2,
            "input_shape": X_train.shape[1]
        })
        
        # Search for best hyperparameters
        tuner.search(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=32
        )
        
        # Get and log best hyperparameters
        best_hp = tuner.get_best_hyperparameters()[0]
        best_model = tuner.get_best_models()[0]
        
        # Log best hyperparameters
        mlflow.log_params({
            "best_" + name: value 
            for name, value in best_hp.values.items()
        })
        
        # Log best model
        mlflow.keras.log_model(
            best_model,
            "best_model",
            registered_model_name="penguin_classifier_tuned"
        )
        
        return best_hp, best_model

# COMMAND ----------

# Load and prepare data
data = load_data()
X = data.drop("species", axis=1)
y = data["species"]

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Transform data
features_transformer, target_transformer = build_transformers()


# Transform features
features_transformer, target_transformer = build_transformers()
X_train_transformed = features_transformer.fit_transform(X_train)
X_val_transformed = features_transformer.transform(X_val)
X_test_transformed = features_transformer.transform(X_test)

# Transform targets - reshape for correct dimensions
y_train_transformed = target_transformer.fit_transform(y_train.to_numpy().reshape(-1, 1))
y_val_transformed = target_transformer.transform(y_val.to_numpy().reshape(-1, 1))
y_test_transformed = target_transformer.transform(y_test.to_numpy().reshape(-1, 1))
logging.info(f"Training data shapes - X: {X_train_transformed.shape}, y: {y_train_transformed.shape}")
logging.info(f"Validation data shapes - X: {X_val_transformed.shape}, y: {y_val_transformed.shape}")


# COMMAND ----------

# Run hyperparameter tuning
best_hp, best_model = run_hyperparameter_tuning(
    X_train_transformed, y_train_transformed,
    X_val_transformed, y_val_transformed
)

# Print results
print("\nBest Hyperparameters:")
for name, value in best_hp.values.items():
    print(f"{name}: {value}")

# Evaluate on test set
test_loss, test_accuracy = best_model.evaluate(X_test_transformed, y_test_transformed)
print(f"\nTest Accuracy: {test_accuracy:.4f}")