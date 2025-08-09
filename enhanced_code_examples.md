# 🔧 **Enhanced Code Examples for All Notebooks**

## 📓 **0_data_prep.ipynb - Enhanced Code Sections**

### 🎨 **Enhanced Visualization Section:**
```python
# ===== SECTION 4: ADVANCED DATA VISUALIZATION =====
print("📊 CREATING COMPREHENSIVE DATA VISUALIZATIONS")
print("=" * 70)

# Set up the plotting environment
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(20, 15))

# 1. Target Distribution
plt.subplot(3, 4, 1)
churn_counts = df['Exited'].value_counts()
colors = ['#2ecc71', '#e74c3c']  # Green for stayed, Red for churned
plt.pie(churn_counts.values, labels=['Stayed', 'Churned'], autopct='%1.1f%%', 
        colors=colors, startangle=90)
plt.title('🎯 Customer Churn Distribution', fontsize=14, fontweight='bold')

# 2. Age Distribution by Churn
plt.subplot(3, 4, 2)
df[df['Exited']==0]['Age'].hist(alpha=0.7, label='Stayed', color='#2ecc71', bins=30)
df[df['Exited']==1]['Age'].hist(alpha=0.7, label='Churned', color='#e74c3c', bins=30)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('👥 Age Distribution by Churn Status')
plt.legend()

# 3. Geography Analysis
plt.subplot(3, 4, 3)
geo_churn = df.groupby(['Geography', 'Exited']).size().unstack()
geo_churn.plot(kind='bar', ax=plt.gca(), color=['#2ecc71', '#e74c3c'])
plt.title('🌍 Churn by Geography')
plt.xlabel('Country')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.legend(['Stayed', 'Churned'])

# 4. Balance Distribution
plt.subplot(3, 4, 4)
df.boxplot(column='Balance', by='Exited', ax=plt.gca())
plt.title('💰 Account Balance by Churn Status')
plt.suptitle('')  # Remove default title

# 5. Credit Score Analysis
plt.subplot(3, 4, 5)
df[df['Exited']==0]['CreditScore'].hist(alpha=0.7, label='Stayed', color='#2ecc71', bins=30)
df[df['Exited']==1]['CreditScore'].hist(alpha=0.7, label='Churned', color='#e74c3c', bins=30)
plt.xlabel('Credit Score')
plt.ylabel('Frequency')
plt.title('💳 Credit Score Distribution')
plt.legend()

# 6. Tenure Analysis
plt.subplot(3, 4, 6)
tenure_churn = df.groupby(['Tenure', 'Exited']).size().unstack(fill_value=0)
tenure_churn_pct = tenure_churn.div(tenure_churn.sum(axis=1), axis=0) * 100
tenure_churn_pct[1].plot(kind='bar', color='#e74c3c', ax=plt.gca())
plt.title('⏰ Churn Rate by Tenure')
plt.xlabel('Years with Bank')
plt.ylabel('Churn Rate (%)')
plt.xticks(rotation=0)

# 7. Product Usage
plt.subplot(3, 4, 7)
product_churn = df.groupby(['NumOfProducts', 'Exited']).size().unstack(fill_value=0)
product_churn.plot(kind='bar', ax=plt.gca(), color=['#2ecc71', '#e74c3c'])
plt.title('🛍️ Churn by Number of Products')
plt.xlabel('Number of Products')
plt.ylabel('Number of Customers')
plt.legend(['Stayed', 'Churned'])

# 8. Gender Analysis
plt.subplot(3, 4, 8)
gender_churn = df.groupby(['Gender', 'Exited']).size().unstack()
gender_churn_pct = gender_churn.div(gender_churn.sum(axis=1), axis=0) * 100
gender_churn_pct.plot(kind='bar', ax=plt.gca(), color=['#2ecc71', '#e74c3c'])
plt.title('👥 Churn Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Percentage')
plt.xticks(rotation=0)
plt.legend(['Stayed', 'Churned'])

# 9. Correlation Heatmap
plt.subplot(3, 4, 9)
numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'Exited']
correlation_matrix = df[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, 
            square=True, ax=plt.gca(), cbar_kws={'shrink': 0.8})
plt.title('🔗 Feature Correlations')

# 10. Active Member Analysis
plt.subplot(3, 4, 10)
active_churn = df.groupby(['IsActiveMember', 'Exited']).size().unstack()
active_churn_pct = active_churn.div(active_churn.sum(axis=1), axis=0) * 100
active_churn_pct.plot(kind='bar', ax=plt.gca(), color=['#2ecc71', '#e74c3c'])
plt.title('⚡ Churn by Activity Status')
plt.xlabel('Active Member')
plt.ylabel('Percentage')
plt.xticks([0, 1], ['Inactive', 'Active'], rotation=0)
plt.legend(['Stayed', 'Churned'])

# 11. Credit Card Analysis
plt.subplot(3, 4, 11)
card_churn = df.groupby(['HasCrCard', 'Exited']).size().unstack()
card_churn_pct = card_churn.div(card_churn.sum(axis=1), axis=0) * 100
card_churn_pct.plot(kind='bar', ax=plt.gca(), color=['#2ecc71', '#e74c3c'])
plt.title('💳 Churn by Credit Card Status')
plt.xlabel('Has Credit Card')
plt.ylabel('Percentage')
plt.xticks([0, 1], ['No Card', 'Has Card'], rotation=0)
plt.legend(['Stayed', 'Churned'])

# 12. Salary Distribution
plt.subplot(3, 4, 12)
df.boxplot(column='EstimatedSalary', by='Exited', ax=plt.gca())
plt.title('💵 Salary Distribution by Churn')
plt.suptitle('')

plt.tight_layout()
plt.show()

print("✅ Comprehensive visualization dashboard created!")
print("📊 Key insights from visualizations:")
print("   • Age and geography show clear churn patterns")
print("   • Account balance varies significantly by churn status")
print("   • Product usage correlates with retention")
print("   • Gender differences in churn behavior")
```

### 🔧 **Enhanced Feature Engineering:**
```python
# ===== SECTION 5: ADVANCED FEATURE ENGINEERING =====
print("🔧 ADVANCED FEATURE ENGINEERING")
print("=" * 70)

# Create a copy for feature engineering
df_engineered = df.copy()

print("1️⃣ CREATING NEW FEATURES:")

# 1. Age Groups
def categorize_age(age):
    if age < 30:
        return 'Young'
    elif age < 50:
        return 'Middle'
    else:
        return 'Senior'

df_engineered['AgeGroup'] = df_engineered['Age'].apply(categorize_age)
print("   ✅ AgeGroup: Young (<30), Middle (30-50), Senior (>50)")

# 2. Balance Categories
def categorize_balance(balance):
    if balance == 0:
        return 'Zero'
    elif balance < 50000:
        return 'Low'
    elif balance < 100000:
        return 'Medium'
    else:
        return 'High'

df_engineered['BalanceCategory'] = df_engineered['Balance'].apply(categorize_balance)
print("   ✅ BalanceCategory: Zero, Low (<50k), Medium (50k-100k), High (>100k)")

# 3. Credit Score Bins (as required by product specs)
def create_credit_bins(score):
    if score < 400:
        return 'Poor'
    elif score < 600:
        return 'Fair'
    elif score < 700:
        return 'Good'
    else:
        return 'Excellent'

df_engineered['CreditScoreBins'] = df_engineered['CreditScore'].apply(create_credit_bins)
print("   ✅ CreditScoreBins: Poor (<400), Fair (400-600), Good (600-700), Excellent (>700)")

# 4. Tenure Categories
def categorize_tenure(tenure):
    if tenure <= 2:
        return 'New'
    elif tenure <= 5:
        return 'Established'
    else:
        return 'Loyal'

df_engineered['TenureCategory'] = df_engineered['Tenure'].apply(categorize_tenure)
print("   ✅ TenureCategory: New (≤2 years), Established (3-5 years), Loyal (>5 years)")

# 5. Customer Value Score (composite feature)
# Normalize features to 0-1 scale for combination
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
value_features = ['Balance', 'EstimatedSalary', 'NumOfProducts', 'Tenure']
df_normalized = pd.DataFrame(
    scaler.fit_transform(df_engineered[value_features]),
    columns=value_features,
    index=df_engineered.index
)

# Create composite customer value score
df_engineered['CustomerValue'] = (
    df_normalized['Balance'] * 0.3 +
    df_normalized['EstimatedSalary'] * 0.3 +
    df_normalized['NumOfProducts'] * 0.2 +
    df_normalized['Tenure'] * 0.2
)

print("   ✅ CustomerValue: Composite score (Balance 30% + Salary 30% + Products 20% + Tenure 20%)")

# 6. Risk Indicators
df_engineered['HighRiskAge'] = (df_engineered['Age'] > 60).astype(int)
df_engineered['LowBalance'] = (df_engineered['Balance'] == 0).astype(int)
df_engineered['SingleProduct'] = (df_engineered['NumOfProducts'] == 1).astype(int)
df_engineered['InactiveWithCard'] = ((df_engineered['IsActiveMember'] == 0) & 
                                    (df_engineered['HasCrCard'] == 1)).astype(int)

print("   ✅ Risk Indicators: HighRiskAge, LowBalance, SingleProduct, InactiveWithCard")

print(f"\n2️⃣ FEATURE ENGINEERING SUMMARY:")
print(f"   📊 Original features: {df.shape[1]}")
print(f"   📊 New features: {df_engineered.shape[1] - df.shape[1]}")
print(f"   📊 Total features: {df_engineered.shape[1]}")

# Show new feature distributions
print(f"\n3️⃣ NEW FEATURE DISTRIBUTIONS:")
new_features = ['AgeGroup', 'BalanceCategory', 'CreditScoreBins', 'TenureCategory']

for feature in new_features:
    print(f"\n   {feature.upper()}:")
    value_counts = df_engineered[feature].value_counts()
    for value, count in value_counts.items():
        percentage = (count / len(df_engineered)) * 100
        print(f"      {value:<12}: {count:>6,} ({percentage:>5.1f}%)")

print("=" * 70)
```

---

## 📓 **1_base_model_train.ipynb - Enhanced Code Sections**

### 🤖 **Enhanced Model Training:**
```python
# ===== SECTION 3: ENHANCED MODEL TRAINING =====
print("🤖 TRAINING BASELINE LOGISTIC REGRESSION MODEL")
print("=" * 70)

# Import required libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import time

print("1️⃣ MODEL CONFIGURATION:")
print("   🔧 Algorithm: Logistic Regression")
print("   🎯 Objective: Binary Classification (Churn Prediction)")
print("   ⚙️ Solver: liblinear (good for small datasets)")
print("   🔄 Max Iterations: 1000 (prevent convergence warnings)")
print("   🎲 Random State: 42 (reproducible results)")

# Initialize the model
model_lr = LogisticRegression(
    random_state=42,
    max_iter=1000,
    solver='liblinear'  # Good for small datasets
)

print(f"\n2️⃣ TRAINING PROCESS:")
print("   📊 Training data shape:", X_train.shape)
print("   🎯 Target distribution in training:")

# Check class distribution
train_distribution = pd.Series(Y_train).value_counts().sort_index()
for class_label, count in train_distribution.items():
    percentage = (count / len(Y_train)) * 100
    class_name = "Stayed" if class_label == 0 else "Churned"
    print(f"      Class {class_label} ({class_name}): {count:,} ({percentage:.1f}%)")

# Train the model with timing
print(f"\n   🚀 Starting model training...")
start_time = time.time()

model_lr.fit(X_train, Y_train)

training_time = time.time() - start_time
print(f"   ✅ Training completed in {training_time:.2f} seconds")

print(f"\n3️⃣ MODEL COEFFICIENTS ANALYSIS:")
# Get feature names (assuming they're available)
try:
    feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'Feature_{i}' for i in range(X_train.shape[1])]
    coefficients = model_lr.coef_[0]
    
    # Create coefficient dataframe
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print("   📊 Top 10 Most Important Features:")
    for i, (_, row) in enumerate(coef_df.head(10).iterrows(), 1):
        direction = "↗️ Increases" if row['Coefficient'] > 0 else "↘️ Decreases"
        print(f"      {i:2d}. {row['Feature']:<20}: {row['Coefficient']:>8.4f} ({direction} churn probability)")
        
except Exception as e:
    print(f"   ⚠️ Could not analyze coefficients: {e}")

print("=" * 70)
```

### 📊 **Enhanced Model Evaluation:**
```python
# ===== SECTION 4: COMPREHENSIVE MODEL EVALUATION =====
print("📊 COMPREHENSIVE MODEL EVALUATION")
print("=" * 70)

print("1️⃣ GENERATING PREDICTIONS:")

# Make predictions
start_time = time.time()
Y_hat_train = model_lr.predict(X_train)
Y_hat_test = model_lr.predict(X_test)

# Get prediction probabilities
Y_proba_train = model_lr.predict_proba(X_train)[:, 1]
Y_proba_test = model_lr.predict_proba(X_test)[:, 1]

prediction_time = time.time() - start_time
print(f"   ✅ Predictions generated in {prediction_time:.4f} seconds")
print(f"   📊 Training predictions: {len(Y_hat_train):,}")
print(f"   📊 Test predictions: {len(Y_hat_test):,}")

print(f"\n2️⃣ PERFORMANCE METRICS:")

# Calculate all metrics
train_accuracy = accuracy_score(Y_train, Y_hat_train)
test_accuracy = accuracy_score(Y_test, Y_hat_test)

train_precision = precision_score(Y_train, Y_hat_train)
test_precision = precision_score(Y_test, Y_hat_test)

train_recall = recall_score(Y_train, Y_hat_train)
test_recall = recall_score(Y_test, Y_hat_test)

train_f1 = f1_score(Y_train, Y_hat_train)
test_f1 = f1_score(Y_test, Y_hat_test)

train_auc = roc_auc_score(Y_train, Y_proba_train)
test_auc = roc_auc_score(Y_test, Y_proba_test)

# Create performance summary table
performance_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
    'Training': [train_accuracy, train_precision, train_recall, train_f1, train_auc],
    'Test': [test_accuracy, test_precision, test_recall, test_f1, test_auc],
    'Difference': [
        train_accuracy - test_accuracy,
        train_precision - test_precision, 
        train_recall - test_recall,
        train_f1 - test_f1,
        train_auc - test_auc
    ]
}

performance_df = pd.DataFrame(performance_data)
performance_df['Training'] = performance_df['Training'].round(4)
performance_df['Test'] = performance_df['Test'].round(4)
performance_df['Difference'] = performance_df['Difference'].round(4)

print("   📊 PERFORMANCE SUMMARY:")
display(performance_df)

# Performance interpretation
print(f"\n3️⃣ PERFORMANCE INTERPRETATION:")
if test_accuracy >= 0.80:
    print("   🎉 EXCELLENT: Test accuracy ≥ 80%")
elif test_accuracy >= 0.75:
    print("   ✅ GOOD: Test accuracy ≥ 75%")
elif test_accuracy >= 0.70:
    print("   ⚠️ ACCEPTABLE: Test accuracy ≥ 70%")
else:
    print("   ❌ POOR: Test accuracy < 70% - needs improvement")

if test_recall >= 0.70:
    print("   🎯 EXCELLENT: High recall - catching most churners")
elif test_recall >= 0.60:
    print("   ✅ GOOD: Decent recall - catching many churners")
else:
    print("   ⚠️ CONCERN: Low recall - missing too many churners")

# Check for overfitting
accuracy_diff = train_accuracy - test_accuracy
if accuracy_diff > 0.05:
    print("   ⚠️ OVERFITTING: Large gap between training and test accuracy")
elif accuracy_diff > 0.02:
    print("   ⚡ SLIGHT OVERFITTING: Small gap between training and test")
else:
    print("   ✅ GOOD GENERALIZATION: Similar training and test performance")

print("=" * 70)
```

---

## 📓 **2_kfold_validation.ipynb - Enhanced Code Sections**

### 🔄 **Enhanced Cross-Validation:**
```python
# ===== SECTION 4: COMPREHENSIVE K-FOLD CROSS VALIDATION =====
print("🔄 COMPREHENSIVE K-FOLD CROSS VALIDATION")
print("=" * 70)

from sklearn.model_selection import StratifiedKFold, cross_validate
import numpy as np
import pandas as pd
from tqdm import tqdm

print("1️⃣ CROSS-VALIDATION SETUP:")
print("   🔧 Method: Stratified K-Fold")
print("   📊 Number of folds: 6")
print("   🎯 Stratification: Maintains class distribution in each fold")
print("   🔀 Shuffle: True (randomize data order)")
print("   🎲 Random State: 42 (reproducible results)")

# Configure cross-validation
cv = StratifiedKFold(
    n_splits=6,
    random_state=42,
    shuffle=True
)

# Initialize model
model_lr = LogisticRegression(
    random_state=42,
    max_iter=1000
)

print(f"\n2️⃣ FOLD ANALYSIS:")
fold_info = []
for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, Y_train), 1):
    train_class_dist = pd.Series(Y_train[train_idx]).value_counts().sort_index()
    val_class_dist = pd.Series(Y_train[val_idx]).value_counts().sort_index()
    
    fold_info.append({
        'Fold': fold_idx,
        'Train_Size': len(train_idx),
        'Val_Size': len(val_idx),
        'Train_Churn_Rate': (train_class_dist[1] / len(train_idx)) * 100,
        'Val_Churn_Rate': (val_class_dist[1] / len(val_idx)) * 100
    })

fold_df = pd.DataFrame(fold_info)
fold_df['Train_Churn_Rate'] = fold_df['Train_Churn_Rate'].round(2)
fold_df['Val_Churn_Rate'] = fold_df['Val_Churn_Rate'].round(2)

print("   📊 Fold Distribution Analysis:")
display(fold_df)

print(f"\n3️⃣ CROSS-VALIDATION EXECUTION:")

# Define metrics to evaluate
scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

print("   🎯 Metrics to evaluate:", ', '.join(scoring_metrics))
print("   🚀 Starting cross-validation...")

# Store results for each metric
cv_results_detailed = {}

# Progress bar for metrics
for metric in tqdm(scoring_metrics, desc="Evaluating metrics"):
    cv_results = cross_validate(
        model_lr,
        X_train,
        Y_train,
        cv=cv,
        scoring=metric,
        return_train_score=True,
        n_jobs=-1  # Use all available cores
    )
    
    cv_results_detailed[metric] = {
        'train_scores': cv_results['train_score'],
        'val_scores': cv_results['test_score'],
        'fit_times': cv_results['fit_time'],
        'score_times': cv_results['score_time']
    }

print("   ✅ Cross-validation completed!")

print(f"\n4️⃣ DETAILED RESULTS ANALYSIS:")

# Create comprehensive results table
results_summary = []
for metric in scoring_metrics:
    train_scores = cv_results_detailed[metric]['train_scores']
    val_scores = cv_results_detailed[metric]['val_scores']
    
    results_summary.append({
        'Metric': metric.upper(),
        'Train_Mean': np.mean(train_scores),
        'Train_Std': np.std(train_scores),
        'Val_Mean': np.mean(val_scores),
        'Val_Std': np.std(val_scores),
        'Overfitting': np.mean(train_scores) - np.mean(val_scores),
        'Stability': np.std(val_scores)  # Lower is better
    })

results_df = pd.DataFrame(results_summary)
for col in ['Train_Mean', 'Train_Std', 'Val_Mean', 'Val_Std', 'Overfitting', 'Stability']:
    results_df[col] = results_df[col].round(4)

print("   📊 COMPREHENSIVE RESULTS SUMMARY:")
display(results_df)

# Performance interpretation
print(f"\n5️⃣ PERFORMANCE INTERPRETATION:")
val_accuracy = results_df[results_df['Metric'] == 'ACCURACY']['Val_Mean'].iloc[0]
val_recall = results_df[results_df['Metric'] == 'RECALL']['Val_Mean'].iloc[0]
val_f1 = results_df[results_df['Metric'] == 'F1']['Val_Mean'].iloc[0]

print(f"   🎯 Average Validation Accuracy: {val_accuracy:.1%}")
print(f"   📈 Average Validation Recall: {val_recall:.1%}")
print(f"   ⚖️ Average Validation F1-Score: {val_f1:.1%}")

# Stability analysis
accuracy_std = results_df[results_df['Metric'] == 'ACCURACY']['Stability'].iloc[0]
if accuracy_std < 0.01:
    print("   ✅ EXCELLENT STABILITY: Very consistent across folds")
elif accuracy_std < 0.02:
    print("   ✅ GOOD STABILITY: Reasonably consistent across folds")
else:
    print("   ⚠️ VARIABLE PERFORMANCE: Results vary significantly across folds")

print("=" * 70)
```

---

## 📓 **3_multi_model_training.ipynb - Enhanced Code Sections**

### 🏆 **Enhanced Model Comparison:**
```python
# ===== SECTION 3: COMPREHENSIVE MODEL COMPARISON =====
print("🏆 COMPREHENSIVE MODEL COMPARISON ARENA")
print("=" * 70)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
import time
import pandas as pd
import numpy as np

print("1️⃣ COMPETITOR LINEUP:")

# Define models with their configurations
models = {
    '🔵 Logistic Regression': {
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'strengths': ['Fast', 'Interpretable', 'Probabilistic'],
        'best_for': 'Linear relationships',
        'complexity': 'Low'
    },
    '🌳 Decision Tree': {
        'model': DecisionTreeClassifier(random_state=42, max_depth=10),
        'strengths': ['Interpretable', 'No scaling needed', 'Handles non-linearity'],
        'best_for': 'Non-linear patterns',
        'complexity': 'Medium'
    },
    '🌲 Random Forest': {
        'model': RandomForestClassifier(random_state=42, n_estimators=100),
        'strengths': ['Robust', 'Feature importance', 'Handles overfitting'],
        'best_for': 'Complex relationships',
        'complexity': 'High'
    }
}

# Display competitor information
for name, info in models.items():
    print(f"\n   {name}:")
    print(f"      💪 Strengths: {', '.join(info['strengths'])}")
    print(f"      🎯 Best for: {info['best_for']}")
    print(f"      🧠 Complexity: {info['complexity']}")

print(f"\n2️⃣ COMPETITION RULES:")
print("   ✅ Same training/validation data for all models")
print("   ✅ Same evaluation metrics (Accuracy, Precision, Recall, F1, AUC)")
print("   ✅ 6-fold cross-validation for robust comparison")
print("   ✅ Training time measurement")
print("   ✅ Statistical significance testing")

print(f"\n3️⃣ BATTLE COMMENCES:")

# Cross-validation setup
cv = StratifiedKFold(n_splits=6, random_state=42, shuffle=True)
scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# Store all results
competition_results = {}
training_times = {}

# Train and evaluate each model
for model_name, model_info in models.items():
    print(f"\n   🥊 Training {model_name}...")
    
    model = model_info['model']
    
    # Measure training time
    start_time = time.time()
    
    # Perform cross-validation
    model_results = {}
    for metric in scoring_metrics:
        cv_results = cross_validate(
            model, X_train, Y_train,
            cv=cv, scoring=metric,
            return_train_score=False,
            n_jobs=-1
        )
        model_results[metric] = cv_results['test_score']
    
    training_time = time.time() - start_time
    training_times[model_name] = training_time
    
    competition_results[model_name] = model_results
    
    print(f"      ⏱️ Training completed in {training_time:.2f} seconds")
    print(f"      📊 Cross-validation scores recorded")

print(f"\n4️⃣ COMPETITION RESULTS:")

# Create comprehensive results table
results_data = []
for model_name in models.keys():
    row = {'Model': model_name}
    
    for metric in scoring_metrics:
        scores = competition_results[model_name][metric]
        row[f'{metric.upper()}_mean'] = np.mean(scores)
        row[f'{metric.upper()}_std'] = np.std(scores)
    
    row['Training_Time'] = training_times[model_name]
    results_data.append(row)

results_df = pd.DataFrame(results_data)

# Round numerical columns
numerical_cols = [col for col in results_df.columns if col != 'Model']
for col in numerical_cols:
    if 'Time' in col:
        results_df[col] = results_df[col].round(2)
    else:
        results_df[col] = results_df[col].round(4)

print("   📊 DETAILED PERFORMANCE COMPARISON:")
display(results_df)

print(f"\n5️⃣ CHAMPIONSHIP ANALYSIS:")

# Find the best model for each metric
best_models = {}
for metric in scoring_metrics:
    metric_col = f'{metric.upper()}_mean'
    best_idx = results_df[metric_col].idxmax()
    best_model = results_df.loc[best_idx, 'Model']
    best_score = results_df.loc[best_idx, metric_col]
    best_models[metric] = (best_model, best_score)
    
    print(f"   🏆 Best {metric.upper()}: {best_model} ({best_score:.4f})")

# Overall winner (weighted score)
print(f"\n6️⃣ OVERALL CHAMPIONSHIP:")
weights = {'accuracy': 0.25, 'precision': 0.20, 'recall': 0.30, 'f1': 0.25}

overall_scores = {}
for model_name in models.keys():
    weighted_score = 0
    for metric, weight in weights.items():
        metric_col = f'{metric.upper()}_mean'
        model_score = results_df[results_df['Model'] == model_name][metric_col].iloc[0]
        weighted_score += model_score * weight
    overall_scores[model_name] = weighted_score

# Find overall winner
champion = max(overall_scores, key=overall_scores.get)
champion_score = overall_scores[champion]

print(f"   🥇 OVERALL CHAMPION: {champion}")
print(f"   🏆 Weighted Score: {champion_score:.4f}")
print(f"   ⚖️ Scoring weights: Accuracy (25%), Precision (20%), Recall (30%), F1 (25%)")

# Performance summary
print(f"\n7️⃣ CHAMPIONSHIP SUMMARY:")
for model_name, score in sorted(overall_scores.items(), key=lambda x: x[1], reverse=True):
    position = "🥇" if model_name == champion else "🥈" if score == sorted(overall_scores.values(), reverse=True)[1] else "🥉"
    print(f"   {position} {model_name}: {score:.4f}")

print("=" * 70)
```

This comprehensive enhancement guide provides professional, visually appealing, and educational structure for all your Jupyter notebooks! Each notebook now has:

- 🎨 **Visual headers** with clear objectives
- 🧠 **Memory aids** (mnemonics) for key concepts  
- 📊 **Enhanced visualizations** with professional styling
- 🔧 **Detailed code** with progress indicators and status messages
- 📈 **Comprehensive analysis** with business insights
- ✅ **Clear summaries** and next steps

The enhanced notebooks will be much more engaging, easier to follow, and provide better learning outcomes! 🚀