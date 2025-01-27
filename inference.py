# Databricks notebook source
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
import tensorflow as tf

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

def load_registered_models() -> Tuple[tf.keras.Model, Any, Any, Dict]:
    """Load the latest versions of all required models from MLflow Model Registry.
    
    Returns:
        Tuple containing:
        - classifier: The main Keras model
        - target_transformer: Transformer for target variable
        - features_transformer: Transformer for input features
        - metadata: Model metadata dictionary
    """
    logging.info("Loading registered models from MLflow Model Registry...")
    
    try:
        # Load the main classifier model
        classifier = mlflow.keras.load_model("models:/penguin_classifier/latest")
        
        # Load the transformers
        target_transformer = mlflow.sklearn.load_model("models:/penguin_target_transformer/latest")
        features_transformer = mlflow.sklearn.load_model("models:/penguin_features_transformer/latest")
        
        # Load metadata
        client = mlflow.tracking.MlflowClient()
        latest_version = client.search_model_versions("name='penguin_classifier'")[0]
        metadata = mlflow.artifacts.load_dict(f"runs:/{latest_version.run_id}/metadata.json")
        
        logging.info("Successfully loaded all models and transformers")
        return classifier, target_transformer, features_transformer, metadata
        
    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")
        raise RuntimeError("Failed to load required models from MLflow Registry") from e

# COMMAND ----------

class PenguinClassifier:
    """Penguin species classifier using models from MLflow Registry with data capture capability."""
    
    def __init__(self, enable_data_capture: bool = True):
        """Initialize the classifier.
        
        Args:
            enable_data_capture: If True, will log predictions to Delta table
        """
        self.model, self.target_transformer, self.features_transformer, self.metadata = load_registered_models()
        self.feature_names = self.metadata.get("feature_names", [])
        self.target_classes = self.metadata.get("target_classes", [])
        self.enable_data_capture = enable_data_capture
        
        # Initialize Spark for data capture
        self.spark = SparkSession.builder.getOrCreate()
        
        logging.info("Classifier initialized with %d features and %d classes", 
                    len(self.feature_names), len(self.target_classes))
    
    def predict(self, input_data: pd.DataFrame, 
               capture_data: Optional[bool] = None) -> Optional[List[Dict[str, Any]]]:
        """Make predictions for the input data.
        
        Args:
            input_data: DataFrame with the same schema as the training data
            capture_data: Override instance-level data capture setting
            
        Returns:
            List of dictionaries containing predictions and confidence scores
        """
        logging.info("Received prediction request with %d samples", len(input_data))
        
        try:
            # Transform features
            X = self.process_input(input_data)
            if X is None:
                return None
                
            # Make predictions
            predictions = self.model.predict(X, verbose=0)
            
            # Process predictions
            results = self.process_output(predictions)
            
            # Capture data if enabled
            should_capture = capture_data if capture_data is not None else self.enable_data_capture
            if should_capture:
                self.capture_data(input_data, results)
            
            return results
            
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            return None
    
    def process_input(self, payload: pd.DataFrame) -> Optional[np.ndarray]:
        """Process the input data for prediction."""
        try:
            # Validate input columns
            missing_features = set(self.feature_names) - set(payload.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Transform features
            result = self.features_transformer.transform(payload)
            logging.info("Features transformed successfully")
            return result
            
        except Exception as e:
            logging.error(f"Error processing input: {str(e)}")
            return None
    
    def process_output(self, predictions: np.ndarray) -> List[Dict[str, Any]]:
        """Process model predictions into human-readable format."""
        prediction_indices = np.argmax(predictions, axis=1)
        confidence_scores = np.max(predictions, axis=1)
        
        # Convert indices to class names
        predicted_classes = [self.target_classes[idx] for idx in prediction_indices]
        
        # Format output
        results = [
            {
                "prediction": species,
                "confidence": float(confidence),
                "prediction_time": datetime.now(timezone.utc).isoformat()
            }
            for species, confidence in zip(predicted_classes, confidence_scores)
        ]
        
        return results
    
    def capture_data(self, input_data: pd.DataFrame, predictions: List[Dict[str, Any]]) -> None:
        """Capture input data and predictions to Delta table."""
        try:
            # Create a copy of input data
            data = input_data.copy()
            
            # Add prediction information
            data['prediction'] = [p['prediction'] for p in predictions]
            data['confidence'] = [p['confidence'] for p in predictions]
            data['prediction_time'] = [p['prediction_time'] for p in predictions]
            
            # Convert to Spark DataFrame
            prediction_schema = StructType([
                *[StructField(name, DoubleType(), True) for name in self.feature_names],
                StructField("prediction", StringType(), True),
                StructField("confidence", DoubleType(), True),
                StructField("prediction_time", TimestampType(), True)
            ])
            
            spark_df = self.spark.createDataFrame(data, schema=prediction_schema)
            
            # Write to Delta table
            table_name = "penguin_predictions"
            spark_df.write.format("delta").mode("append").saveAsTable(table_name)
            
            logging.info(f"Successfully captured predictions to Delta table: {table_name}")
            
        except Exception as e:
            logging.error(f"Error capturing prediction data: {str(e)}")

# COMMAND ----------

def load_test_data() -> pd.DataFrame:
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
    
    try:
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
        
    except Exception as e:
        logging.error(f"Error loading test data: {str(e)}")
        raise

# COMMAND ----------

# Initialize classifier
classifier = PenguinClassifier(enable_data_capture=True)

# COMMAND ----------

# Load and make predictions on test data
try:
    test_data = load_test_data()
    logging.info("Test data shape: %s", test_data.shape)
    
    predictions = classifier.predict(test_data)
    
    # Display results
    if predictions:
        for i, (pred, true) in enumerate(zip(predictions, test_data['species'])):
            print(f"\nSample {i+1}:")
            print(f"True species: {true}")
            print(f"Predicted species: {pred['prediction']} (confidence: {pred['confidence']:.3f})")
            print(f"Prediction time: {pred['prediction_time']}")
    else:
        print("No predictions were generated")