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