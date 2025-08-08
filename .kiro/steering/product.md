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