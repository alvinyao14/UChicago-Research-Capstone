"""LLM Classifier model loading and prediction."""

import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForSequenceClassification
from typing import Dict

from ...config import get_config


class LLMClassifier:
    """Wrapper for LLM classifier model."""
    
    def __init__(self, config=None):
        """Load LLM classifier model and tokenizer."""
        if config is None:
            config = get_config()
        
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(str(config.llm_model_path))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoPeftModelForSequenceClassification.from_pretrained(
            str(config.llm_model_path),
            num_labels=2,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            device_map="auto",  # Requires a GPU!
            offload_folder=str(config.llm_offload_path)
        )
        self.model.eval()
        
        print("\nSuccessfully loaded fine-tuned OpenBioLLM model.")
    
    def predict_proba(self, text: str) -> float:
        """Run prediction on formatted text."""
        device = self.model.device
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            return float(probs[1].cpu().numpy())


# Global LLM classifier instance (lazy loaded)
_llm_classifier_instance = None


def get_llm_classifier() -> LLMClassifier:
    """Get or create global LLM classifier instance."""
    global _llm_classifier_instance
    if _llm_classifier_instance is None:
        _llm_classifier_instance = LLMClassifier()
    return _llm_classifier_instance


def format_for_llm_classifier(patient_data: Dict) -> str:
    """Formats the raw DB row for the CLASSIFICATION model."""
    return (
        f"age range: {patient_data.get('age_bucket')} / "
        f"sex: {patient_data.get('sex')} / "
        f"heart rate: {patient_data.get('heart_rate')} / "
        f"systolic blood pressure: {patient_data.get('bp_systolic')} / "
        f"diastolic blood pressure: {patient_data.get('bp_diastolic')} / "
        f"respiratory rate: {patient_data.get('resp_rate')} / "
        f"temperature in Celsius: {patient_data.get('temperature_C')} / "
        f"oxygen saturation: {patient_data.get('oxygen_saturation')} / "
        f"ESI: {int(patient_data.get('ESI', 0))} / "
        f"recent admissions (in 30 days): {int(patient_data.get('recent_admissions_30d', 0))} / "
        f"{patient_data.get('triage_notes_redacted', '')}"
    )


def llm_predict_proba(text: str) -> float:
    """Runs the formatted text through the CLASSIFIER and returns P(Admit)."""
    classifier = get_llm_classifier()
    return classifier.predict_proba(text)

