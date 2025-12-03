# ER Triage Workflow

A LangGraph-based workflow system for Emergency Room triage decision-making that combines traditional ML models, LLM classifiers, and human-in-the-loop feedback.

## 📁 Project Structure

```
er_triage_workflow/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup.py                 # Package setup (optional)
├── .gitignore              # Git ignore rules
│
├── config/                  # Configuration
│   ├── __init__.py
│   └── settings.py         # Configuration management
│
├── src/                    # Source code
│   ├── __init__.py
│   │
│   ├── models/             # Model loading and inference
│   │   ├── __init__.py
│   │   ├── ml_model.py    # Traditional ML model
│   │   ├── llm_model.py   # LLM classifier
│   │   └── fusion_agent.py # Fusion agent LLM
│   │
│   ├── database/           # Database operations
│   │   ├── __init__.py
│   │   └── queries.py     # Database queries
│   │
│   ├── workflow/           # LangGraph workflow
│   │   ├── __init__.py
│   │   ├── state.py       # State definitions
│   │   ├── nodes.py       # Workflow nodes
│   │   ├── routing.py     # Conditional routing
│   │   └── graph.py       # Graph construction
│   │
│   ├── utils/              # Utilities
│   │   ├── __init__.py
│   │   ├── json_parser.py # JSON parsing utilities
│   │   ├── logging.py     # Logging utilities
│   │   └── risk_scoring.py # Risk scoring
│   │
│   └── evaluation/         # Evaluation tools
│       ├── __init__.py
│       └── metrics.py     # Evaluation metrics
│
├── scripts/                # Executable scripts
│   ├── run_workflow.py    # Main workflow runner
│   └── evaluate.py        # Evaluation script
│
└── tests/                  # Unit tests
    ├── __init__.py
    └── test_*.py
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd er_triage_workflow

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Set the `BASE_PATH` environment variable to point to your project root:

```bash
export BASE_PATH="/path/to/Capstone_Organized"
```

Or modify `config/settings.py` directly.

### Running the Workflow

```bash
python scripts/run_workflow.py --visit_id 1 --human_prompt "Patient is 70yo, frail, and on chemotherapy."
```

## 📋 Features

- **Multi-Model Fusion**: Combines traditional ML and LLM predictions
- **Human-in-the-Loop**: Supports human review and override
- **Patient History**: Incorporates historical visit patterns
- **Risk Scoring**: Calculates patient risk based on multiple factors
- **Robust Error Handling**: Comprehensive logging and retry logic
- **Clinical Reasoning**: Fusion agent provides explainable decisions

## 🔧 Components

### Models
- **ML Model**: Gradient Boosting Classifier trained on structured features
- **LLM Classifier**: Fine-tuned OpenBioLLM for text-based classification
- **Fusion Agent**: Generative LLM that synthesizes all inputs

### Workflow Nodes
1. **fetch_data**: Retrieves patient data and history
2. **severity_gate**: Early exit for critical cases
3. **ml_model**: ML model prediction
4. **llm_model**: LLM classifier prediction
5. **human_input**: Processes human notes
6. **fusion**: Combines all inputs with fusion agent
7. **confidence_check**: Routes based on confidence
8. **human_review**: Human-in-the-loop review (optional)
9. **finalize**: Final decision with rationale

## 📊 Evaluation

Run evaluation on test set:

```bash
python scripts/evaluate.py --test_csv path/to/test_data.csv
```

## 📝 License

[Your License Here]

## 👥 Authors

[Your Name/Team]

