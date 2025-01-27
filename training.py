# Databricks notebook source
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.model_selection import KFold
import mlflow
import pickle
import scipy.sparse

# COMMAND ----------

# Enable MLflow autologging for Keras
mlflow.keras.autolog()

# COMMAND ----------

# Pipeline Configuration
TRAINING_BATCH_SIZE = 32
TRAINING_EPOCHS = 50
ACCURACY_THRESHOLD = 0.7

# Configure logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

# COMMAND ----------

def build_target_transformer():
    """Build a Scikit-Learn transformer to preprocess the target column."""
    return OrdinalEncoder()


# COMMAND ----------

def build_features_transformer():
    """Build a Scikit-Learn transformer to preprocess the feature columns."""
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
            (
                "numeric",
                numeric_transformer,
                make_column_selector(dtype_exclude="object"),
            ),
            (
                "categorical",
                categorical_transformer,
                ["island", "sex"],
            ),
        ],
    )


# COMMAND ----------

def build_model(input_shape, learning_rate=0.01):
    """Build and compile the neural network to predict the species of a penguin."""
    from keras import Input, layers, models, optimizers

    model = models.Sequential(
        [
            Input(shape=(input_shape,)),
            layers.Dense(10, activation="relu"),
            layers.Dense(8, activation="relu"),
            layers.Dense(3, activation="softmax"),
        ],
    )

    model.compile(
        optimizer=optimizers.SGD(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, DoubleType

def load_dataset():
    """Load the penguins dataset from Delta table and prepare it."""
       # Define explicit schema
    schema = StructType([
        StructField("species", StringType(), True),
        StructField("island", StringType(), True),
        StructField("culmen_length_mm", DoubleType(), True),
        StructField("culmen_depth_mm", DoubleType(), True),
        StructField("flipper_length_mm", DoubleType(), True),
        StructField("body_mass_g", DoubleType(), True),
        StructField("sex", StringType(), True)
    ])

    # Read from Delta table
    data_location = "abfss://raw@cloudinfrastg.dfs.core.windows.net/00_data_source/"
    #delta_location = "abfss://raw@cloudinfrastg.dfs.core.windows.net/02_bronze/ml_delta/penguins"
    file_name = "penguins.csv"
    dataset_path = data_location + file_name

    spark.sql("DROP TABLE IF EXISTS penguins")
   
    logging.info(f"Loading dataset from: {dataset_path}")
    data = spark.read.csv(dataset_path, header=True, schema=schema)
    data.describe()

    logging.info(f"Saving dataset to Delta Lake at: default catalog")
    data.write.format("delta").mode("overwrite").saveAsTable("penguins")

    data = spark.read.table("penguins").toPandas()
    
    # Replace extraneous values in the sex column with NaN
    data["sex"] = data["sex"].replace(".", np.nan)

    # Shuffle the dataset
    seed = int(time.time() * 1000)
    generator = np.random.default_rng(seed=seed)
    data = data.sample(frac=1, random_state=generator)

    logging.info("Loaded dataset with %d samples", len(data))
    return data

# COMMAND ----------

# MAGIC %sql
# MAGIC delete from default.penguins

# COMMAND ----------

data = load_dataset()

# COMMAND ----------

# Set up cross-validation
kfold = KFold(n_splits=5, shuffle=True)
folds = list(enumerate(kfold.split(data)))

# Initialize metrics storage
fold_metrics = []
experiment_name = "/Users/d.t.georgian.pirvu@axpo.com/mlpenguins_experiments"
mlflow.set_experiment(experiment_name)

# COMMAND ----------

with mlflow.start_run(run_name="cross-validation") as parent_run:
    parent_run_id = parent_run.info.run_id
    for fold, (train_indices, test_indices) in folds:
            # Start a child run for this fold
        with mlflow.start_run(run_name=f"cross-validation-fold-{fold}", nested=True) as child_run:
            logging.info(f"Training fold {fold + 1}/5...")
            # Log fold information
        
        # Verify data shapes and content
            logging.info(f"Train indices shape: {train_indices.shape}, Test indices shape: {test_indices.shape}")
            logging.info(f"Data shape: {data.shape}")
        
        # Transform target with error checking
            target_data = data.species.to_numpy()
            if target_data is None or len(target_data) == 0:
                raise ValueError("Target data is empty or None")
        
            target_data = target_data.reshape(-1, 1)
            logging.info(f"Target data shape before split: {target_data.shape}")
        
            target_transformer = build_target_transformer()
            y_train = target_transformer.fit_transform(target_data[train_indices])
            y_test = target_transformer.transform(target_data[test_indices])
            logging.info(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
        
        # Transform features with error checking
            features_transformer = build_features_transformer()
            try:
                x_train = features_transformer.fit_transform(data.iloc[train_indices])
                x_test = features_transformer.transform(data.iloc[test_indices])
                logging.info(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")
            except Exception as e:
                logging.error(f"Error in feature transformation: {str(e)}")
                logging.error(f"Data columns: {data.columns.tolist()}")
                logging.error(f"Data types: {data.dtypes}")
                raise
        
        # Build and train model
            input_shape = x_train.shape[1]
            logging.info(f"Building model with input shape: {input_shape}")
            model = build_model(input_shape=input_shape)
        
            history = model.fit(
                x_train, y_train,
                epochs=TRAINING_EPOCHS,
                batch_size=TRAINING_BATCH_SIZE,
                validation_data=(x_test, y_test),
                verbose=1
            )
        
            # Evaluate model
            test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("test_loss", test_loss)

            # Log training history metrics
            for epoch, metrics in enumerate(history.history['accuracy']):
                mlflow.log_metric("train_accuracy", metrics, step=epoch)
                mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
                mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
                mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
        
            fold_metrics.append({
                'fold': fold,
                'test_accuracy': test_accuracy,
                'test_loss': test_loss
            })
        
            logging.info(f"Fold {fold + 1} - Test Accuracy: {test_accuracy:.4f}")
       # Log average metrics in the parent run
    avg_accuracy = np.mean([m['test_accuracy'] for m in fold_metrics])
    avg_loss = np.mean([m['test_loss'] for m in fold_metrics])
    mlflow.log_metric("avg_test_accuracy", avg_accuracy)
    mlflow.log_metric("avg_test_loss", avg_loss)     

# COMMAND ----------

avg_accuracy = np.mean([m['test_accuracy'] for m in fold_metrics])
avg_loss = np.mean([m['test_loss'] for m in fold_metrics])

print(f"Average Test Accuracy: {avg_accuracy:.4f}")
print(f"Average Test Loss: {avg_loss:.4f}")

# COMMAND ----------

fold_metrics

# COMMAND ----------

# Train final model if cross-validation performance is good enough
if avg_accuracy >= ACCURACY_THRESHOLD:
    with mlflow.start_run(run_name="final-model") as parent_run:
        logging.info("Training final model on full dataset...")
        
        with mlflow.start_run(run_name="data-preprocessing", nested=True):
            # Transform target
            target_data = data.species.to_numpy().reshape(-1, 1)
            target_transformer = build_target_transformer()
            y = target_transformer.fit_transform(target_data)
            
            # Get and log target classes
            target_classes = target_transformer.categories_[0].tolist()
            mlflow.log_param("target_classes", target_classes)
            
            # Transform features
            features_transformer = build_features_transformer()
            X = features_transformer.fit_transform(data)
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param("n_samples", X.shape[0])
            
            # Log feature names
            numeric_features = features_transformer.named_transformers_["numeric"].get_feature_names_out()
            categorical_features = features_transformer.named_transformers_["categorical"].get_feature_names_out()
            feature_names = list(numeric_features) + list(categorical_features)
            mlflow.log_param("feature_names", feature_names)
        
        with mlflow.start_run(run_name="model-training", nested=True):
            # Build and train final model
            model = build_model(input_shape=X.shape[1])
            
            history = model.fit(
                X, y,
                epochs=TRAINING_EPOCHS,
                batch_size=TRAINING_BATCH_SIZE,
                validation_split=0.2,
                verbose=1
            )
            
            # Log final metrics
            final_loss, final_accuracy = model.evaluate(X, y, verbose=0)
            mlflow.log_metric("final_accuracy", final_accuracy)
            mlflow.log_metric("final_loss", final_loss)
            
            # Log training history
            for epoch, metrics in enumerate(history.history['accuracy']):
                mlflow.log_metric("train_accuracy", metrics, step=epoch)
                mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
                mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
                mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
        
        # Save the final model and transformers
        with mlflow.start_run(run_name="model-saving", nested=True):
            # Log the Keras model - MLflow autolog will handle the model artifacts
            mlflow.keras.log_model(
                model,
                "model",
                registered_model_name="penguin_classifier"
            )
            
            # Log transformers as artifacts
            mlflow.sklearn.log_model(
                target_transformer,
                "transformers/target_transformer",
                registered_model_name="penguin_target_transformer"
            )
            
            mlflow.sklearn.log_model(
                features_transformer,
                "transformers/features_transformer",
                registered_model_name="penguin_features_transformer"
            )
            
            # Log additional metadata including raw input/output schemas
            metadata = {
                "feature_names": feature_names,
                "target_classes": target_classes,
                "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "raw_input_schema": {
                    "culmen_length_mm": "double",
                    "culmen_depth_mm": "double",
                    "flipper_length_mm": "double",
                    "body_mass_g": "double",
                    "island": "string",
                    "sex": "string"
                },
                "raw_output_schema": {
                    "prediction": "string",
                    "confidence": "double"
                },
                "input_features_info": {
                    "numeric_features": [
                        "culmen_length_mm",
                        "culmen_depth_mm",
                        "flipper_length_mm",
                        "body_mass_g"
                    ],
                    "categorical_features": ["island", "sex"]
                }
            }
            mlflow.log_dict(metadata, "metadata.json")
            
            logging.info("Final model and transformers saved successfully to MLflow Model Registry")
else:
    logging.warning(
        f"Cross-validation accuracy {avg_accuracy:.4f} below threshold {ACCURACY_THRESHOLD}. "
        "Final model will not be trained."
    )


# COMMAND ----------

mlflow.end_run()