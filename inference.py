# Databricks notebook source
import logging
import os
import time
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

# COMMAND ----------

# Configure logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)

# COMMAND ----------


def load_registered_models():
    """Load the latest versions of all required models from MLflow Model Registry."""
    logging.info("Loading registered models from MLflow Model Registry...")
    
    # Load the main classifier model
    classifier = mlflow.keras.load_model("models:/penguin_classifier/latest")
    
    # Load the transformers
    target_transformer = mlflow.sklearn.load_model("models:/penguin_target_transformer/latest")
    features_transformer = mlflow.sklearn.load_model("models:/penguin_features_transformer/latest")
    
    # Load metadata
    client = mlflow.tracking.MlflowClient()
    latest_run = client.search_model_versions("name='penguin_classifier'")[0]
    metadata = mlflow.artifacts.load_dict(f"runs:/{latest_run.run_id}/metadata.json")
    
    return classifier, target_transformer, features_transformer, metadata

# COMMAND ----------

model=load_registered_models()

# COMMAND ----------

class PenguinClassifier:
    """Penguin species classifier using models from MLflow Registry."""
    
    def __init__(self):
        """Initialize the classifier by loading models from MLflow Registry."""
        self.model, self.target_transformer, self.features_transformer, self.metadata = load_registered_models()
        self.feature_names = self.metadata["feature_names"]
        self.target_classes = self.metadata["target_classes"]
        logging.info("Classifier initialized with %d features and %d classes", 
                    len(self.feature_names), len(self.target_classes))
    
    def predict(self, input_data: pd.DataFrame) -> list:
        """Make predictions for the input data.
        
        Args:
            input_data: DataFrame with the same schema as the training data
            
        Returns:
            List of dictionaries containing predictions and confidence scores
        """
        logging.info("Received prediction request with %d samples", len(input_data))
        
        # Transform features
        try:
            X = self.features_transformer.transform(input_data)
            logging.info("Features transformed successfully")
        except Exception as e:
            logging.error("Error transforming features: %s", str(e))
            return None
        
        # Make predictions
        predictions = self.model.predict(X, verbose=0)
        
        # Process predictions
        prediction_indices = np.argmax(predictions, axis=1)
        confidence_scores = np.max(predictions, axis=1)
        
        # Convert indices to class names
        predicted_classes = [self.target_classes[idx] for idx in prediction_indices]
        
        # Format output
        results = [
            {
                "prediction": species,
                "confidence": float(confidence)  # Convert numpy float to Python float
            }
            for species, confidence in zip(predicted_classes, confidence_scores)
        ]
        
        logging.info("Predictions completed successfully")
        return results

# COMMAND ----------


def load_test_data():
    """Load a sample of data for testing the inference pipeline."""
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
    
    # Read from Delta table
    data_location = "abfss://raw@cloudinfrastg.dfs.core.windows.net/00_data_source/"
    file_name = "penguins.csv"
    dataset_path = data_location + file_name
    
    logging.info(f"Loading test data from: {dataset_path}")
    data = spark.read.csv(dataset_path, header=True, schema=schema)
    
    # Convert to pandas and take a small sample
    pdf = data.toPandas()
    test_sample = pdf.sample(n=5, random_state=42)
    
    return test_sample


# COMMAND ----------

classifier = PenguinClassifier()

# COMMAND ----------

# Load test data
test_data = load_test_data()
logging.info("Test data shape: %s", test_data.shape)


# COMMAND ----------

predictions = classifier.predict(test_data)

# COMMAND ----------


# Display results
for i, (pred, true) in enumerate(zip(predictions, test_data['species'])):
    print(f"\nSample {i+1}:")
    print(f"True species: {true}")
    print(f"Predicted species: {pred['prediction']} (confidence: {pred['confidence']:.3f})")