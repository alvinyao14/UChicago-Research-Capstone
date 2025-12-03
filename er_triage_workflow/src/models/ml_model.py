"""ML Model loading and prediction."""

import re
import json
import joblib
import pandas as pd
from typing import Dict

from ...config import get_config


class MLModel:
    """Wrapper for ML model and preprocessor."""
    
    def __init__(self, config=None):
        """Load ML model, preprocessor, and feature names."""
        if config is None:
            config = get_config()
        
        self.config = config
        self.model = joblib.load(str(config.ml_model_path))
        self.preprocessor = joblib.load(str(config.ml_preprocessor_path))
        
        with open(str(config.ml_features_path), 'r') as f:
            self.feature_names = json.load(f)
        
        print(f"Successfully loaded ML model and preprocessor ({len(self.feature_names)} features).")
    
    def predict_proba(self, patient_data: Dict) -> float:
        """Run prediction on patient data."""
        input_df = pd.DataFrame([patient_data])
        transformed_data = self.preprocessor.transform(input_df)
        input_transformed_df = pd.DataFrame(
            transformed_data.toarray(),
            columns=self.feature_names
        )
        prob_admit = self.model.predict_proba(input_transformed_df)[0][1]
        return float(prob_admit)


# Global ML model instance (lazy loaded)
_ml_model_instance = None


def get_ml_model() -> MLModel:
    """Get or create global ML model instance."""
    global _ml_model_instance
    if _ml_model_instance is None:
        _ml_model_instance = MLModel()
    return _ml_model_instance


def clean_text_for_ml(text: str) -> str:
    """
    Cleans the triage notes for TF-IDF.
    This version removes the [AGE] tags and all non-alpha characters.
    (Copied from 3.1.1-Traditional_ML_Training.ipynb)
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\\[.*?\\]', ' ', text)    # Remove tags like [AGE]
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Remove special characters
    text = text.lower()                       # Convert to lowercase
    text = re.sub(r'\s+', ' ', text).strip()  # Consolidate whitespace
    return text


def ml_predict_proba(patient_data: Dict) -> float:
    """Runs the raw data through the ML preprocessor and model."""
    model = get_ml_model()
    return model.predict_proba(patient_data)

