# ChurnModelDeployment

---
inclusion: always
---

# ChurnModelDeployment Product Guide

Banking customer churn prediction system using binary classification to identify customers likely to exit.

## Mandatory Feature Schema
ALWAYS use these exact variable names and categories in all code:

```python
numerical_features = ['Age', 'Tenure', 'Balance', 'EstimatedSalary']
nominal_features = ['Gender', 'Geography']  
ordinal_features = ['CreditScoreBins']
remainder_features = ['NumOfProducts', 'HasCrCard', 'IsActiveMember']
target_variable = 'Exited'
```

## Required Preprocessing Pipeline
Use this exact ColumnTransformer structure:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('nom', OneHotEncoder(drop='first'), nominal_features),
        ('ord', OrdinalEncoder(), ordinal_features),
        ('remainder', 'passthrough', remainder_features)
    ]
)
```

## Model Development Standards
- **Class Imbalance**: MUST use imbalanced-learn techniques (SMOTE preferred)
- **Evaluation Priority**: Recall > Precision (business critical for churn detection)
- **Model Selection**: LogisticRegression, RandomForest, or XGBoost only
- **Validation**: StratifiedKFold with n_splits=5
- **Performance Check**: Validate across all Geography segments

## Code Architecture Patterns
- Define feature lists as module-level constants at top of notebooks
- Always use sklearn.pipeline.Pipeline for preprocessing + model
- Save both preprocessor and model together using joblib
- Include feature importance visualization for business stakeholders

## File Organization Rules
- **Models**: `model/{algorithm}_{YYYYMMDD_HHMMSS}.joblib`
- **Notebooks**: Sequential numbering: `0_data_prep.ipynb`, `1_eda.ipynb`, `2_modeling.ipynb`
- **Processed Data**: Store in `data/` with descriptive names

## Business Requirements
- **Target Variable**: `Exited` (1=churned customer, 0=retained customer)
- **Geographic Markets**: France, Germany, Spain (treat as nominal categorical)
- **Credit Assessment**: Use `CreditScoreBins` not raw `CreditScore`
- **Compliance**: Model must be interpretable for regulatory requirements
- **Performance Threshold**: Minimum recall of 0.70 on test set

## Validation Checklist
Before finalizing any model:
- [ ] Feature schema matches exactly
- [ ] Class imbalance addressed
- [ ] Performance validated per geography
- [ ] Feature importance documented
- [ ] Pipeline saved with model

# Technology Stack

## Build System & Package Management
- **Package Manager**: UV (modern Python package manager)
- **Python Version**: >=3.11
- **Project Configuration**: pyproject.toml (modern Python packaging)
- **Virtual Environment**: .venv (managed by UV)

## Core Libraries & Frameworks
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, imbalanced-learn
- **Data Visualization**: matplotlib, seaborn, plotly
- **Statistical Analysis**: statsmodels
- **Development Environment**: Jupyter notebooks, ipykernel
- **Utilities**: python-dotenv, pyyaml, joblib

## Data Science Stack
- **EDA & Profiling**: pandas-profiling, missingno
- **Feature Engineering**: scikit-learn preprocessing (StandardScaler, OneHotEncoder, OrdinalEncoder)
- **Model Pipeline**: sklearn.pipeline, sklearn.compose.ColumnTransformer
- **Class Imbalance**: imbalanced-learn

## Common Commands
```bash
# Environment setup
uv sync                    # Install dependencies from lock file
uv add <package>          # Add new dependency
uv remove <package>       # Remove dependency

# Development
jupyter notebook          # Start Jupyter server
jupyter lab              # Start JupyterLab

# Python execution
uv run python script.py  # Run Python script in virtual environment
uv run jupyter notebook  # Run Jupyter in virtual environment
```

## Development Workflow
- Use Jupyter notebooks for exploratory data analysis and prototyping
- Follow scikit-learn pipeline patterns for preprocessing
- Maintain reproducible environments with uv.lock

---
inclusion: always
---

# ChurnModelDeployment Product Guide

Banking customer churn prediction system using binary classification to identify customers likely to exit.

## Mandatory Feature Schema
ALWAYS use these exact variable names and categories in all code:

```python
numerical_features = ['Age', 'Tenure', 'Balance', 'EstimatedSalary']
nominal_features = ['Gender', 'Geography']  
ordinal_features = ['CreditScoreBins']
remainder_features = ['NumOfProducts', 'HasCrCard', 'IsActiveMember']
target_variable = 'Exited'
```

## Required Preprocessing Pipeline
Use this exact ColumnTransformer structure:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('nom', OneHotEncoder(drop='first'), nominal_features),
        ('ord', OrdinalEncoder(), ordinal_features),
        ('remainder', 'passthrough', remainder_features)
    ]
)
```

## Model Development Standards
- **Class Imbalance**: MUST use imbalanced-learn techniques (SMOTE preferred)
- **Evaluation Priority**: Recall > Precision (business critical for churn detection)
- **Model Selection**: LogisticRegression, RandomForest, or XGBoost only
- **Validation**: StratifiedKFold with n_splits=5
- **Performance Check**: Validate across all Geography segments

## Code Architecture Patterns
- Define feature lists as module-level constants at top of notebooks
- Always use sklearn.pipeline.Pipeline for preprocessing + model
- Save both preprocessor and model together using joblib
- Include feature importance visualization for business stakeholders

## File Organization Rules
- **Models**: `model/{algorithm}_{YYYYMMDD_HHMMSS}.joblib`
- **Notebooks**: Sequential numbering: `0_data_prep.ipynb`, `1_eda.ipynb`, `2_modeling.ipynb`
- **Processed Data**: Store in `data/` with descriptive names

## Business Requirements
- **Target Variable**: `Exited` (1=churned customer, 0=retained customer)
- **Geographic Markets**: France, Germany, Spain (treat as nominal categorical)
- **Credit Assessment**: Use `CreditScoreBins` not raw `CreditScore`
- **Compliance**: Model must be interpretable for regulatory requirements
- **Performance Threshold**: Minimum recall of 0.70 on test set

