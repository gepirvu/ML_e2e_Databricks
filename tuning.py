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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
import keras_tuner

# COMMAND ----------

# Configure logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)

# COMMAND ----------

# Set memory growth for GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

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

def build_model(hp: keras_tuner.HyperParameters, input_shape: int) -> keras.Model:
    """Build a tunable model with hyperparameters."""
    # Tunable parameters - reduced search space for stability
    learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
    units = hp.Int("units", 32, 128, step=32)
    dropout_rate = hp.Float("dropout_rate", 0.1, 0.3, step=0.1)
    
    # Simpler model architecture
    model = keras.Sequential([
        layers.Dense(units, activation="relu", input_shape=(input_shape,)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(3, activation="softmax")
    ])
    
    # Compile model with mixed precision
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
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

class MLflowTuner(keras_tuner.Tuner):
    """Custom tuner that logs results to MLflow."""
    
    def __init__(
        self,
        hypermodel,
        objective=None,
        max_trials=10,
        executions_per_trial=1,
        directory=None,
        project_name=None,
        **kwargs
    ):
        self.objective = objective or "val_accuracy"
        directory = directory or "/dbfs/tmp/keras_tuner"
        project_name = project_name or "default_project"
        
        super().__init__(
            hypermodel=hypermodel,
            oracle=keras_tuner.oracles.RandomSearchOracle(
                objective=keras_tuner.Objective(self.objective, "max"),
                max_trials=max_trials,
            ),
            directory=directory,
            project_name=project_name,
            **kwargs
        )
        self.executions_per_trial = executions_per_trial

    def run_trial(self, trial, *args, **kwargs):
        kwargs["batch_size"] = trial.hyperparameters.Int("batch_size", 32, 64, step=32)
        kwargs["epochs"] = trial.hyperparameters.Int("epochs", 10, 20, step=5)
        
        results = []
        for execution in range(self.executions_per_trial):
            copied_kwargs = kwargs.copy()
            model = self.hypermodel(trial.hyperparameters)
            history = model.fit(*args, **copied_kwargs)
            results.append(history.history)
            
        # Log to MLflow
        with mlflow.start_run(run_name=f"trial-{trial.trial_id}", nested=True):
            for name, value in trial.hyperparameters.values.items():
                mlflow.log_param(name, value)
            
            # Log the best metrics
            for metric in ["accuracy", "val_accuracy", "loss", "val_loss"]:
                if metric in results[0]:
                    best_value = max([result[metric][-1] for result in results])
                    mlflow.log_metric(metric, best_value)
        
        # Return the results of the best execution
        return results[0]

def run_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    """Run hyperparameter tuning with MLflow tracking."""
    
    with mlflow.start_run(run_name="hyperparameter-tuning") as run:
        tuner = MLflowTuner(
            hypermodel=lambda hp: build_model(hp, X_train.shape[1]),
            objective="val_accuracy",
            max_trials=5,  # Reduced number of trials
            executions_per_trial=1,  # Reduced executions
            directory="/dbfs/tmp/keras_tuner",
            project_name="penguin_classification"
        )
        
        # Log tuning parameters
        mlflow.log_params({
            "max_trials": 5,
            "executions_per_trial": 1,
            "input_shape": X_train.shape[1]
        })
        
        # Search for best hyperparameters with early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )
        
        # Search with reduced epochs and batch size
        tuner.search(
            X_train, y_train,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            epochs=10,  # Reduced epochs
            batch_size=32,
            verbose=1
        )
        
        # Get and log best hyperparameters
        best_hp = tuner.get_best_hyperparameters()[0]
        best_model = tuner.hypermodel(best_hp)
        
        # Train the best model
        history = best_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )
        
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