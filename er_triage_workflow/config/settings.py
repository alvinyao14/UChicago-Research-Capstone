"""Configuration settings for ER Admission Agentic AI."""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration class for ER Admission Agentic AI."""
    
    # Base paths
    base_path: Path
    data_path: Path
    model_training_path: Path
    logs_path: Path
    
    # Database
    db_path: Path
    
    # ML Model paths
    ml_model_path: Path
    ml_preprocessor_path: Path
    ml_features_path: Path
    
    # LLM Model paths
    llm_model_id: str = "aaditya/Llama3-OpenBioLLM-8B"
    llm_model_path: Path = None
    llm_offload_path: Path = None
    
    # Workflow settings
    admission_threshold: float = 0.5
    max_retries: int = 2
    retry_delay: float = 0.5
    
    # Logging
    log_path: Optional[Path] = None
    error_log_path: Optional[Path] = None
    
    def __post_init__(self):
        """Validate paths after initialization."""
        if self.llm_model_path is None:
            self.llm_model_path = self.model_training_path / "3.2-LLM_Classification" / "3.2.0-FineTune_OpenBioLLM" / "OpenBioLLM_Final"
        if self.llm_offload_path is None:
            self.llm_offload_path = self.model_training_path / "3.2-LLM_Classification" / "3.2.0-FineTune_OpenBioLLM" / "OpenBioLLM_Offload"
        if self.log_path is None:
            self.log_path = self.logs_path / "trace_log.jsonl"
        if self.error_log_path is None:
            self.error_log_path = self.logs_path / "error_log.jsonl"
        
        # Create directories if they don't exist
        self.logs_path.mkdir(parents=True, exist_ok=True)
        self.llm_offload_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls, base_path: Optional[str] = None) -> "Config":
        """
        Create configuration from environment variables or default paths.
        
        Args:
            base_path: Base path to the project. If None, uses environment variable or default.
        
        Returns:
            Config instance
        """
        if base_path is None:
            base_path = os.getenv("BASE_PATH", "/content/drive/MyDrive/Work/Capstone-TeamFolder/Capstone_Organized")
        
        base = Path(base_path)
        
        return cls(
            base_path=base,
            data_path=base / "1-Data",
            model_training_path=base / "3-Model_Training",
            logs_path=base / "4-LangGraph" / "4.0-LangGraph_Logs",
            db_path=base / "1-Data" / "ED_Simulated_Database_Fixed.db",
            ml_model_path=base / "3-Model_Training" / "3.1-Traditional_ML" / "3.1.0-Traditional_ML_Artifacts" / "gb_model.joblib",
            ml_preprocessor_path=base / "3-Model_Training" / "3.1-Traditional_ML" / "3.1.0-Traditional_ML_Artifacts" / "ml_preprocessor.joblib",
            ml_features_path=base / "3-Model_Training" / "3.1-Traditional_ML" / "3.1.0-Traditional_ML_Artifacts" / "ml_feature_columns.json",
            admission_threshold=float(os.getenv("ADMISSION_THRESHOLD", "0.5")),
            max_retries=int(os.getenv("MAX_RETRIES", "2")),
        )
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validate that all required paths exist.
        
        Returns:
            (is_valid, error_message)
        """
        required_paths = {
            "Database": self.db_path,
            "ML Model": self.ml_model_path,
            "ML Preprocessor": self.ml_preprocessor_path,
            "ML Features": self.ml_features_path,
            "LLM Model": self.llm_model_path,
        }
        
        missing = []
        for name, path in required_paths.items():
            if not path.exists():
                missing.append(f"{name}: {path}")
        
        if missing:
            return False, f"Missing required paths:\n" + "\n".join(missing)
        
        return True, None


# Global config instance (can be overridden)
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def set_config(config: Config):
    """Set the global configuration instance."""
    global _config
    _config = config

