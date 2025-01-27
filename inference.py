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
        # Define expected input schema
        input_schema = {
            "inputs": [
                {"name": "culmen_length_mm", "type": "double"},
                {"name": "culmen_depth_mm", "type": "double"},
                {"name": "flipper_length_mm", "type": "double"},
                {"name": "body_mass_g", "type": "double"},
                {"name": "island", "type": "string"},
                {"name": "sex", "type": "string"}
            ]
        }
        
        # Define expected output schema
        output_schema = {
            "outputs": [
                {"name": "prediction", "type": "string"},
                {"name": "confidence", "type": "double"}
            ]
        }
        
        # Load the main classifier model
        classifier = mlflow.keras.load_model("models:/penguin_classifier/latest")
        
        # Check if model has signature
        try:
            model_info = mlflow.models.get_model_info("models:/penguin_classifier/latest")
            if not model_info.signature:
                logging.warning("Model loaded without schema signature. Input/output validation may be limited.")
        except Exception as e:
            logging.warning(f"Could not validate model signature: {str(e)}")
        
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
        
        # Define required input features (before transformation)
        self.required_features = [
            "culmen_length_mm",
            "culmen_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
            "island",
            "sex"
        ]
        
        # Define input schema for validation
        self.input_schema = StructType([
            StructField("culmen_length_mm", DoubleType(), True),
            StructField("culmen_depth_mm", DoubleType(), True),
            StructField("flipper_length_mm", DoubleType(), True),
            StructField("body_mass_g", DoubleType(), True),
            StructField("island", StringType(), True),
            StructField("sex", StringType(), True)
        ])
        
        # Define output schema for predictions table
        self.prediction_schema = StructType([
            StructField("island", StringType(), True),
            StructField("culmen_length_mm", DoubleType(), True),
            StructField("culmen_depth_mm", DoubleType(), True),
            StructField("flipper_length_mm", DoubleType(), True),
            StructField("body_mass_g", DoubleType(), True),
            StructField("sex", StringType(), True),
            StructField("prediction", StringType(), True),
            StructField("confidence", DoubleType(), True),
            StructField("prediction_time", TimestampType(), True)
        ])
        
        self.target_classes = self.metadata.get("target_classes", [])
        self.enable_data_capture = enable_data_capture
        
        # Initialize Spark for data capture
        self.spark = SparkSession.builder.getOrCreate()
        
        logging.info("Classifier initialized with %d input features and %d classes", 
                    len(self.required_features), len(self.target_classes))
    
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
            # Make a copy to avoid modifying the original data
            data = payload.copy()
            
            # Validate input columns
            missing_features = set(self.required_features) - set(data.columns)
            if missing_features:
                raise ValueError(f"Missing required input features: {missing_features}")
            
            # Ensure numeric columns are float
            numeric_columns = [
                "culmen_length_mm",
                "culmen_depth_mm",
                "flipper_length_mm",
                "body_mass_g"
            ]
            for col in numeric_columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Handle missing values in categorical columns
            data['island'] = data['island'].fillna('NA')
            data['sex'] = data['sex'].fillna('NA')
            
            # Validate data types using Spark schema
            try:
                spark_df = self.spark.createDataFrame(data, schema=self.input_schema)
                data = spark_df.toPandas()
            except Exception as e:
                logging.error(f"Schema validation failed: {str(e)}")
                return None
            
            # Transform features using the transformer
            result = self.features_transformer.transform(data)
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
            # Create a new DataFrame with the correct column order and types using the prediction schema
            data_dict = {
                "island": input_data["island"].astype(str),
                "culmen_length_mm": input_data["culmen_length_mm"].astype(float),
                "culmen_depth_mm": input_data["culmen_depth_mm"].astype(float),
                "flipper_length_mm": input_data["flipper_length_mm"].astype(float),
                "body_mass_g": input_data["body_mass_g"].astype(float),
                "sex": input_data["sex"].astype(str),
                "prediction": [p["prediction"] for p in predictions],
                "confidence": [float(p["confidence"]) for p in predictions],
                "prediction_time": pd.to_datetime([p["prediction_time"] for p in predictions])
            }
            
            # Create pandas DataFrame
            pdf = pd.DataFrame(data_dict)
            
            # Convert to Spark DataFrame with schema
            spark_df = self.spark.createDataFrame(pdf, schema=self.prediction_schema)
            
            # Write to Delta table
            table_name = "penguin_predictions"
            
            # Create the table if it doesn't exist
            spark_df.write \
                .format("delta") \
                .mode("append") \
                .option("mergeSchema", "true") \
                .saveAsTable(table_name)
            
            logging.info(f"Successfully captured predictions to Delta table: {table_name}")
            
        except Exception as e:
            logging.error(f"Error capturing prediction data: {str(e)}")
            logging.debug("Data types of input columns:")
            for col in input_data.columns:
                logging.debug(f"{col}: {input_data[col].dtype}")

# COMMAND ----------

def load_test_data() -> pd.DataFrame:
    """Load a sample of data for testing the inference pipeline."""
    try:
        # Read from Delta table
        data_location = "abfss://raw@cloudinfrastg.dfs.core.windows.net/00_data_source/"
        file_name = "penguins.csv"
        dataset_path = data_location + file_name
        
        logging.info(f"Loading test data from: {dataset_path}")
        
        # Read data with explicit schema
        schema = StructType([
            StructField("species", StringType(), True),
            StructField("island", StringType(), True),
            StructField("culmen_length_mm", DoubleType(), True),
            StructField("culmen_depth_mm", DoubleType(), True),
            StructField("flipper_length_mm", DoubleType(), True),
            StructField("body_mass_g", DoubleType(), True),
            StructField("sex", StringType(), True)
        ])
        
        data = spark.read.csv(dataset_path, header=True, schema=schema)
        
        # Convert to pandas and take a small sample
        pdf = data.toPandas()
        
        # Store true species for evaluation
        true_species = pdf['species'].copy()
        
        # Drop the species column as it's not needed for inference
        test_sample = pdf.drop('species', axis=1).sample(n=5, random_state=42)
        
        # Store true species for comparison
        test_sample.attrs['true_species'] = true_species[test_sample.index]
        
        logging.info("Test data loaded successfully")
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
        true_species = test_data.attrs.get('true_species', None)
        for i, pred in enumerate(predictions):
            print(f"\nSample {i+1}:")
            if true_species is not None:
                print(f"True species: {true_species.iloc[i]}")
            print(f"Predicted species: {pred['prediction']} (confidence: {pred['confidence']:.3f})")
            print(f"Prediction time: {pred['prediction_time']}")
    else:
        print("No predictions were generated")