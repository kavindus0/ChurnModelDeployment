# Project Structure

## Directory Organization
```
ChurnModelDeployment/
├── .git/                 # Git version control
├── .kiro/               # Kiro AI assistant configuration
├── .venv/               # Python virtual environment (UV managed)
├── .vscode/             # VS Code settings
├── data/                # Dataset storage
│   └── dataset.csv      # Customer churn dataset
├── model/               # Model artifacts and saved models
├── .gitignore           # Git ignore patterns
├── .python-version      # Python version specification
├── pyproject.toml       # Project configuration and dependencies
├── requirements.txt     # Pip-style requirements (generated)
├── uv.lock             # Dependency lock file
├── README.md           # Project documentation
└── *.ipynb             # Jupyter notebooks for analysis
```

## File Naming Conventions
- **Notebooks**: Use numbered prefixes for sequential workflow (e.g., `0_data_prep.ipynb`, `1_eda.ipynb`, `2_modeling.ipynb`)
- **Data files**: Store in `data/` directory with descriptive names
- **Models**: Save trained models in `model/` directory with version/timestamp suffixes

## Data Pipeline Structure
Based on the existing preprocessing pipeline:
- **Feature Categories**:
  - `numerical_features`: Age, Tenure, Balance, EstimatedSalary
  - `nominal_features`: Gender, Geography (one-hot encoded)
  - `ordinal_features`: CreditScoreBins
  - `remainder_features`: NumOfProducts, HasCrCard, IsActiveMember

## Development Flow
1. **Data Preparation** (`0_data_prep.ipynb`): Data loading, cleaning, preprocessing
2. **EDA** (future notebooks): Exploratory data analysis
3. **Modeling** (future notebooks): Model training and evaluation
4. **Model Storage**: Serialized models in `model/` directory

## Code Organization Principles
- Keep notebooks focused on specific stages of the ML pipeline
- Use scikit-learn ColumnTransformer for consistent preprocessing
- Maintain separation between data, code, and model artifacts
- Follow consistent variable naming for feature categories