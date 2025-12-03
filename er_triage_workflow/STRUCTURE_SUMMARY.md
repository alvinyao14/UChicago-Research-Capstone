# Repository Structure Summary

## ✅ Completed

### 1. Repository Structure
- ✅ Created folder structure (`src/`, `config/`, `scripts/`, `tests/`)
- ✅ Created `.gitignore` for Python projects
- ✅ Created `requirements.txt` with all dependencies

### 2. Configuration
- ✅ `config/settings.py` - Centralized configuration management
- ✅ `config/__init__.py` - Package initialization

### 3. Utilities
- ✅ `src/utils/json_parser.py` - Robust JSON parsing
- ✅ `src/utils/risk_scoring.py` - Risk scoring and temporal features
- ✅ `src/utils/__init__.py` - Package exports

### 4. Documentation
- ✅ `README.md` - Main project documentation
- ✅ `CONVERSION_GUIDE.md` - Guide for converting notebook code

## 🚧 To Be Created (Extract from Notebook)

### 1. Utilities (`src/utils/logging.py`)
**Source**: Notebook cells ~500-600
**Contains**:
- `init_execution_context(visit_id)`
- `get_execution_id()`
- `log_event(...)`
- `log_error(...)`
- `track_performance(step_name)`
- `make_logged_node(fn, name, max_retries, retry_delay)`

### 2. Models (`src/models/`)

#### `ml_model.py`
**Source**: Notebook cells ~750-800
**Contains**:
- `clean_text_for_ml(text)` function
- ML model loading logic
- `ml_predict_proba(patient_data)` function

#### `llm_model.py`
**Source**: Notebook cells ~800-850
**Contains**:
- `format_for_llm_classifier(patient_data)` function
- LLM classifier loading logic
- `llm_predict_proba(text)` function

#### `fusion_agent.py`
**Source**: Notebook cell ~1055
**Contains**:
- Fusion agent LLM loading logic
- `run_fusion_agent(ml_prob, llm_prob, human_note, max_retries)` function

### 3. Database (`src/database/queries.py`)
**Source**: `fetch_data_node` function in notebook
**Contains**:
- Database connection management
- Patient data query with history
- Query execution and result processing

### 4. Workflow (`src/workflow/`)

#### `state.py`
**Source**: Notebook cell ~1200
**Contains**:
- `VitalSigns` Pydantic model
- `ERState` TypedDict definition
- `validate_initial_state(state)` function
- `validate_state_transition(state, node_name)` function

#### `nodes.py`
**Source**: Notebook cells ~1400-1900
**Contains**:
- `fetch_data_node(state)`
- `severity_gate_node(state)`
- `ml_model_node(state)`
- `llm_model_node(state)`
- `human_input_node(state)`
- `fusion_node(state)`
- `confidence_check_node(state)`
- `human_review_node(state)`
- `finalize_node(state)`
- `run_models_node(state)`

#### `routing.py`
**Source**: Conditional routing functions
**Contains**:
- `conditional_severity_gate(state)`
- `conditional_confidence_routing(state)`

#### `graph.py`
**Source**: Notebook cells ~2000-2100
**Contains**:
- Graph construction logic
- Node registration
- Edge definitions
- Graph compilation

### 5. Scripts (`scripts/`)

#### `run_workflow.py`
**Source**: Notebook cells ~2200 (test runs)
**Contains**:
- Main entry point
- Command-line argument parsing
- Workflow invocation
- Result display

#### `evaluate.py`
**Source**: Notebook cells ~2300+ (evaluation section)
**Contains**:
- Test set loading
- Batch evaluation
- Metrics calculation
- Results export

## 📋 Extraction Checklist

- [ ] Extract logging utilities (`src/utils/logging.py`)
- [ ] Extract ML model code (`src/models/ml_model.py`)
- [ ] Extract LLM model code (`src/models/llm_model.py`)
- [ ] Extract fusion agent code (`src/models/fusion_agent.py`)
- [ ] Extract database queries (`src/database/queries.py`)
- [ ] Extract state definitions (`src/workflow/state.py`)
- [ ] Extract workflow nodes (`src/workflow/nodes.py`)
- [ ] Extract routing logic (`src/workflow/routing.py`)
- [ ] Extract graph construction (`src/workflow/graph.py`)
- [ ] Create main script (`scripts/run_workflow.py`)
- [ ] Create evaluation script (`scripts/evaluate.py`)
- [ ] Update all imports to use new package structure
- [ ] Test the complete workflow

## 🔧 How to Extract

1. **Open the notebook** in Jupyter/Colab
2. **Locate the cell** mentioned in the "Source" section
3. **Copy the code** from that cell
4. **Paste into the corresponding Python file**
5. **Update imports** to use the new package structure
6. **Remove Colab-specific code** (e.g., `drive.mount()`)
7. **Replace hard-coded paths** with `Config` references
8. **Test the module** independently

## 📝 Import Updates Needed

When extracting code, update imports from:
```python
# Old (notebook)
from some_module import something

# New (package)
from src.utils.logging import log_event
from src.models.ml_model import ml_predict_proba
from config import get_config
```

## 🎯 Priority Order

1. **High Priority** (Core functionality):
   - `src/utils/logging.py` - Needed by all nodes
   - `src/workflow/state.py` - Defines data structures
   - `src/workflow/nodes.py` - Core workflow logic
   - `src/workflow/graph.py` - Workflow assembly

2. **Medium Priority** (Model inference):
   - `src/models/ml_model.py`
   - `src/models/llm_model.py`
   - `src/models/fusion_agent.py`

3. **Lower Priority** (Supporting code):
   - `src/database/queries.py`
   - `src/workflow/routing.py`
   - `scripts/run_workflow.py`
   - `scripts/evaluate.py`

