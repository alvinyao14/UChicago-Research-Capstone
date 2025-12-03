# Notebook to Python Package Conversion Guide

This guide explains how the notebook code has been organized into a Python package structure.

## 📦 Module Organization

### 1. Configuration (`config/settings.py`)
**From Notebook**: Path definitions and constants (Cell ~337-358)
- Extracted all path definitions
- Created `Config` dataclass for centralized configuration
- Supports environment variables

### 2. Utilities (`src/utils/`)

#### `json_parser.py`
**From Notebook**: Cell 5 (parse_json_with_fallback function)
- Robust JSON parsing with multiple fallback strategies

#### `risk_scoring.py`
**From Notebook**: Cell 5 (extract_temporal_features, calculate_patient_risk_score)
- Patient risk scoring
- Temporal feature extraction

#### `logging.py` (TO BE CREATED)
**From Notebook**: Cells ~500-600
- Execution context management
- Event logging
- Error logging
- Performance tracking
- Node wrapper with retry logic

### 3. Models (`src/models/`)

#### `ml_model.py` (TO BE CREATED)
**From Notebook**: Cells ~750-800
- ML model loading
- ML prediction function
- Text cleaning for ML

#### `llm_model.py` (TO BE CREATED)
**From Notebook**: Cells ~800-850
- LLM classifier loading
- LLM prediction function
- Text formatting for LLM

#### `fusion_agent.py` (TO BE CREATED)
**From Notebook**: Cell ~1055 (run_fusion_agent function)
- Fusion agent LLM loading
- Fusion agent execution with retry logic

### 4. Database (`src/database/queries.py`) (TO BE CREATED)
**From Notebook**: fetch_data_node function
- Patient data fetching
- Patient history queries
- Database connection management

### 5. Workflow (`src/workflow/`)

#### `state.py` (TO BE CREATED)
**From Notebook**: Cell ~1200
- ERState TypedDict definition
- VitalSigns Pydantic model
- State validation functions

#### `nodes.py` (TO BE CREATED)
**From Notebook**: Cells ~1400-1900
- All workflow node functions:
  - fetch_data_node
  - severity_gate_node
  - ml_model_node
  - llm_model_node
  - human_input_node
  - fusion_node
  - confidence_check_node
  - human_review_node
  - finalize_node

#### `routing.py` (TO BE CREATED)
**From Notebook**: Conditional routing functions
- conditional_severity_gate
- conditional_confidence_routing

#### `graph.py` (TO BE CREATED)
**From Notebook**: Cells ~2000-2100
- Graph construction
- Node registration
- Edge definitions
- Graph compilation

### 6. Scripts (`scripts/`)

#### `run_workflow.py` (TO BE CREATED)
**From Notebook**: Cell ~2200 (test runs)
- Main entry point
- Workflow invocation
- Command-line interface

#### `evaluate.py` (TO BE CREATED)
**From Notebook**: Cells ~2300+ (evaluation section)
- Evaluation on test set
- Metrics calculation
- Results export

## 🔄 Next Steps

1. **Extract remaining modules** from the notebook
2. **Update imports** to use the new package structure
3. **Create unit tests** for each module
4. **Update documentation** with examples

## 📝 Notes

- The notebook uses Colab-specific paths - these should be configurable via `Config`
- Some global variables (like model instances) need to be managed as singletons or passed through state
- The logging system writes to files - ensure log directories exist

