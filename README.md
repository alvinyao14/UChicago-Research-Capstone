# ER Triage LangGraph Workflow

A comprehensive LangGraph-based workflow for Emergency Room (ER) triage admission decisions, integrating traditional Machine Learning models, Large Language Model (LLM) classifiers, and a fusion agent with human-in-the-loop (HITL) capabilities.

## 📁 Repository Structure

```
Capstone_Organized/
├── 1-Data/                          # Data files
│   ├── ED_Simulated_Database_Fixed.db
│   └── ED_Model_Training_Dataset.csv
├── 2-PreWorkflow_Dataset_Prep/     # Data preparation notebooks
├── 3-Model_Training/               # Model training artifacts
│   ├── 3.1-Traditional_ML/
│   │   └── 3.1.0-Traditional_ML_Artifacts/
│   │       ├── gb_model.joblib          # ⚠️ Not in repo (too large)
│   │       ├── ml_preprocessor.joblib   # ⚠️ Not in repo (too large)
│   │       └── ml_feature_columns.json  # ✅ Included
│   └── 3.2-LLM_Classification/
│       └── 3.2.0-FineTune_OpenBioLLM/
│           └── OpenBioLLM_Final/        # ⚠️ Not in repo (too large)
├── 4-LangGraph/                    # LangGraph workflow notebooks
│   ├── 4.0-LangGraph_Logs/        # Execution logs
│   └── 4.1.4-LangGraph_Agent_with_Reasoning-Optimal_Threshold_Finder.ipynb
├── 5-Evaluation_Reports/          # Evaluation results
└── er_triage_workflow/             # ✅ Python package (main code)
    ├── config/                     # Configuration
    ├── src/                        # Source code
    │   ├── database/              # Database queries
    │   ├── models/                # Model loading & inference
    │   ├── utils/                 # Utilities
    │   ├── workflow/              # LangGraph workflow
    │   └── main.py                # Entry point
    ├── tests/                      # Unit tests
    ├── scripts/                    # Utility scripts
    └── requirements.txt           # Dependencies
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.9+**
2. **Model Artifacts**: You need to obtain the trained models separately (they're too large for Git):
   - ML Model: `3-Model_Training/3.1-Traditional_ML/3.1.0-Traditional_ML_Artifacts/gb_model.joblib`
   - ML Preprocessor: `3-Model_Training/3.1-Traditional_ML/3.1.0-Traditional_ML_Artifacts/ml_preprocessor.joblib`
   - LLM Model: `3-Model_Training/3.2-LLM_Classification/3.2.0-FineTune_OpenBioLLM/OpenBioLLM_Final/`

3. **Database**: `1-Data/ED_Simulated_Database_Fixed.db` (not in repo due to size)

### Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd Capstone_Organized
   ```

2. Set up Python environment:
   ```bash
   cd er_triage_workflow
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set the base path (if different from default):
   ```bash
   export BASE_PATH="/path/to/Capstone_Organized"
   ```

4. Ensure model artifacts and database are in place (see structure above)

### Running the Workflow

```bash
cd er_triage_workflow
python -m src.main
```

Or use the workflow programmatically:

```python
from er_triage_workflow.src.main import run_simulation

result = run_simulation(
    visit_id=1,
    human_prompt="Patient is 70yo, frail, and on chemotherapy."
)
print(f"Decision: {result['decision']}")
```

## 📝 Important Notes

### Files Not in Repository

Due to GitHub file size limits, the following are **NOT** included in the repository:

- **Database files** (`.db`, `.sqlite`): Place in `1-Data/`
- **Model files** (`.joblib`, `.pkl`, `.safetensors`, `.bin`): Place in `3-Model_Training/`
- **Large CSV files**: Place in `1-Data/`
- **Log files** (`.jsonl`): Generated at runtime in `4-LangGraph/4.0-LangGraph_Logs/`

### Directory Structure

The code expects the following directory structure relative to `BASE_PATH`:

```
BASE_PATH/
├── 1-Data/
│   └── ED_Simulated_Database_Fixed.db
├── 3-Model_Training/
│   ├── 3.1-Traditional_ML/
│   │   └── 3.1.0-Traditional_ML_Artifacts/
│   │       ├── gb_model.joblib
│   │       ├── ml_preprocessor.joblib
│   │       └── ml_feature_columns.json
│   └── 3.2-LLM_Classification/
│       └── 3.2.0-FineTune_OpenBioLLM/
│           └── OpenBioLLM_Final/
└── 4-LangGraph/
    └── 4.0-LangGraph_Logs/
```

## 🔧 Configuration

Configuration is managed in `er_triage_workflow/config/settings.py`. You can:

1. Set `BASE_PATH` environment variable
2. Modify paths in the config file
3. Use `Config.from_env(base_path="/your/path")` programmatically

## 📚 Documentation

- **Package Structure**: See `er_triage_workflow/STRUCTURE_SUMMARY.md`
- **Conversion Guide**: See `er_triage_workflow/CONVERSION_GUIDE.md`
- **Extraction Status**: See `er_triage_workflow/EXTRACTION_STATUS.md`
- **Quick Start**: See `er_triage_workflow/QUICK_START.md`

## 🧪 Evaluation

The evaluation script is in the notebook `4-LangGraph/4.1.4-LangGraph_Agent_with_Reasoning-Optimal_Threshold_Finder.ipynb` (cells 38-48). This will be extracted to `er_triage_workflow/src/evaluation/evaluate.py` in a future update.

## 📄 License

[Add your license here]

## 👥 Contributors

[Add contributors here]
