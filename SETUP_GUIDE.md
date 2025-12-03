# Setup Guide for Cloned Repository

This guide helps you set up the repository after cloning, especially if you need to add the large files that aren't in Git.

## Step 1: Clone and Navigate

```bash
git clone <your-repo-url>
cd Capstone_Organized
```

## Step 2: Set Up Python Environment

```bash
cd er_triage_workflow
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step 3: Add Required Files

The following files are **not** in the repository due to size limits. You need to add them manually:

### Database File
- **Location**: `1-Data/ED_Simulated_Database_Fixed.db`
- **Size**: ~496KB
- **Source**: Your original data directory

### ML Model Artifacts
Place these files in `3-Model_Training/3.1-Traditional_ML/3.1.0-Traditional_ML_Artifacts/`:
- `gb_model.joblib` - Gradient Boosting model
- `ml_preprocessor.joblib` - Preprocessing pipeline
- `ml_feature_columns.json` - Feature names (✅ already in repo)

### LLM Model
Place the fine-tuned model in:
- `3-Model_Training/3.2-LLM_Classification/3.2.0-FineTune_OpenBioLLM/OpenBioLLM_Final/`

This should contain:
- `adapter_model.safetensors`
- `adapter_config.json`
- `tokenizer_config.json`
- Other model files

### Training Dataset (Optional)
- **Location**: `1-Data/ED_Model_Training_Dataset.csv`
- **Size**: ~558KB
- **Used for**: Evaluation and testing

## Step 4: Verify Structure

After adding files, verify your structure matches:

```bash
# Check database exists
ls -lh 1-Data/ED_Simulated_Database_Fixed.db

# Check ML artifacts
ls -lh 3-Model_Training/3.1-Traditional_ML/3.1.0-Traditional_ML_Artifacts/

# Check LLM model
ls -lh 3-Model_Training/3.2-LLM_Classification/3.2.0-FineTune_OpenBioLLM/OpenBioLLM_Final/
```

## Step 5: Set Base Path (if needed)

If your repository is in a different location, set the `BASE_PATH` environment variable:

```bash
export BASE_PATH="/path/to/Capstone_Organized"
```

Or modify `er_triage_workflow/config/settings.py` directly.

## Step 6: Test Installation

```bash
cd er_triage_workflow
python -c "from config import get_config; cfg = get_config(); print('Config loaded:', cfg.base_path)"
```

## Troubleshooting

### "Database file not found"
- Ensure `1-Data/ED_Simulated_Database_Fixed.db` exists
- Check `BASE_PATH` is set correctly

### "ML Model not found"
- Ensure model files are in `3-Model_Training/3.1-Traditional_ML/3.1.0-Traditional_ML_Artifacts/`
- Check file names match exactly

### "LLM Checkpoint not found"
- Ensure LLM model is in `3-Model_Training/3.2-LLM_Classification/3.2.0-FineTune_OpenBioLLM/OpenBioLLM_Final/`
- Verify the directory structure matches

### Import Errors
- Make sure you're in the `er_triage_workflow` directory or have installed the package
- Try: `pip install -e .` from the `er_triage_workflow` directory

