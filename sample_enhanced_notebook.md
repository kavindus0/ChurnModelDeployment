# 📚 **Sample Enhanced Notebook Section**

## 🔧 **0_data_prep.ipynb - Complete Enhanced Version**

```markdown
# 🔧 **Step 0: Data Preparation & Preprocessing**
## *Building the Foundation for Churn Prediction*

---

### 📋 **What We'll Accomplish Today:**

| Step | Task | Goal | Time |
|------|------|------|------|
| 1️⃣ | **Load Data** | Get our customer dataset | 2 min |
| 2️⃣ | **Explore Data** | Understand what we have | 5 min |
| 3️⃣ | **Clean Data** | Fix missing values & errors | 10 min |
| 4️⃣ | **Transform Features** | Prepare data for ML | 15 min |
| 5️⃣ | **Split Data** | Create train/test sets | 3 min |
| 6️⃣ | **Balance Data** | Handle class imbalance | 5 min |

**Total Time:** ~40 minutes

---

### 🧠 **Memory Aid - "PREP" Method:**
- **P**ick up the data (Load)
- **R**eview what you have (Explore) 
- **E**liminate problems (Clean)
- **P**repare for modeling (Transform)

---

### 🎯 **Learning Objectives:**
By the end of this notebook, you will:
- ✅ Load and explore customer churn data
- ✅ Handle missing values and outliers
- ✅ Transform categorical and numerical features
- ✅ Create balanced train/test datasets
- ✅ Save preprocessed data for modeling
```

```python
# ===== SECTION 1: LIBRARY IMPORTS =====

# 🧠 Memory Aid: "DATA-VIZ-ML-PREP-BALANCE"
# DATA: numpy, pandas
# VIZ: matplotlib  
# ML: sklearn pipeline
# PREP: preprocessing tools
# BALANCE: SMOTE for class balance

# ===== CORE DATA LIBRARIES =====
import numpy as np                    # 🔢 Numbers & arrays
import pandas as pd                   # 📊 Data tables & analysis
import warnings                       # ⚠️ Control warning messages
warnings.filterwarnings('ignore')

# ===== VISUALIZATION =====
import matplotlib.pyplot as plt       # 📈 Charts & graphs
import seaborn as sns                 # 🎨 Beautiful statistical plots
plt.style.use('seaborn-v0_8')        # 🎨 Set attractive plot style

# ===== MACHINE LEARNING PIPELINE =====
from sklearn.pipeline import Pipeline              # 🔧 ML workflow organizer
from sklearn.compose import ColumnTransformer      # 🎯 Feature transformer
from sklearn.model_selection import train_test_split # ✂️ Data splitter

# ===== DATA PREPROCESSING =====
from sklearn.impute import SimpleImputer           # 🔧 Fill missing values
from sklearn.preprocessing import (                 # 🎨 Data transformers
    OneHotEncoder,      # For categories (Gender, Geography)
    OrdinalEncoder,     # For ordered categories (CreditScoreBins)  
    StandardScaler      # For numbers (Age, Balance, etc.)
)

# ===== CLASS BALANCING =====
from imblearn.over_sampling import SMOTE           # ⚖️ Balance churned vs non-churned

# ===== SUCCESS MESSAGE =====
print("🎉 SUCCESS: All libraries loaded!")
print("📚 Libraries imported:")
print("   📊 Data: numpy, pandas")  
print("   📈 Visualization: matplotlib, seaborn")
print("   🤖 ML: sklearn pipeline & preprocessing")
print("   ⚖️ Balancing: SMOTE")
print("🚀 Ready to start data preparation!")
```

```markdown
## 1️⃣ **Load Raw Data**

### 📂 **Getting Our Customer Dataset**

**What we're doing:** Loading bank customer data to predict who might leave (churn)

**Memory Aid - "LOAD":**
- **L**ocate the file
- **O**pen with pandas  
- **A**ssess the shape
- **D**isplay first few rows

### 📋 **Expected Dataset Structure:**
| Column | Type | Description |
|--------|------|-------------|
| CustomerId | int | Unique customer identifier |
| Surname | object | Customer last name |
| CreditScore | int | Credit rating (300-850) |
| Geography | object | Country (France, Germany, Spain) |
| Gender | object | Male or Female |
| Age | int | Customer age |
| Tenure | int | Years with bank |
| Balance | float | Account balance |
| NumOfProducts | int | Number of bank products |
| HasCrCard | int | Has credit card (0/1) |
| IsActiveMember | int | Active member (0/1) |
| EstimatedSalary | float | Estimated annual salary |
| Exited | int | **TARGET**: Left bank (0/1) |
```

```python
# ===== SECTION 2: DATA LOADING =====

print("📂 LOADING CUSTOMER CHURN DATASET...")
print("=" * 60)

# 📂 Load the customer churn dataset
try:
    df = pd.read_csv('data/dataset.csv')
    print("✅ SUCCESS: Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ ERROR: Dataset file not found!")
    print("💡 TIP: Make sure 'data/dataset.csv' exists")
    raise

# 📊 Dataset Overview
print(f"📏 Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"💾 Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# 🎯 Target Variable Check
if 'Exited' in df.columns:
    churn_rate = df['Exited'].mean() * 100
    print(f"🎯 Churn Rate: {churn_rate:.1f}% ({df['Exited'].sum():,} out of {len(df):,} customers)")
    
    # Class distribution
    class_counts = df['Exited'].value_counts()
    print(f"   📊 Stayed: {class_counts[0]:,} customers ({(class_counts[0]/len(df)*100):.1f}%)")
    print(f"   📊 Churned: {class_counts[1]:,} customers ({(class_counts[1]/len(df)*100):.1f}%)")
else:
    print("⚠️ WARNING: 'Exited' column not found!")

print("=" * 60)
```

```python
# 👀 First look at our data
print("🔍 FIRST 5 CUSTOMERS:")
display(df.head())

print("\n📋 DATASET INFORMATION:")
print("-" * 60)
print(f"{'Column':<20} {'Type':<15} {'Non-Null':<10} {'Unique':<8} {'Sample'}")
print("-" * 60)

for col in df.columns:
    non_null = df[col].count()
    unique_count = df[col].nunique()
    sample_val = str(df[col].iloc[0])
    if len(sample_val) > 15:
        sample_val = sample_val[:12] + "..."
    
    print(f"{col:<20} {str(df[col].dtype):<15} {non_null:<10} {unique_count:<8} {sample_val}")

print("-" * 60)
```

```markdown
## 2️⃣ **Explore Data Quality**

### 🔍 **Data Quality Assessment**

**Memory Aid - "CHECK":**
- **C**ount missing values
- **H**unt for duplicates  
- **E**xamine data types
- **C**atch outliers
- **K**eep notes on issues

### 📊 **Quality Metrics We'll Check:**
| Metric | What It Tells Us | Action Needed |
|--------|------------------|---------------|
| Missing Values | Data completeness | Impute or remove |
| Duplicates | Data integrity | Remove duplicates |
| Outliers | Data validity | Cap or transform |
| Data Types | Correct format | Convert if needed |
```

```python
# ===== SECTION 3: DATA QUALITY EXPLORATION =====

print("🔍 DATA QUALITY ASSESSMENT")
print("=" * 60)

# 📊 Missing Values Analysis
print("1️⃣ MISSING VALUES CHECK:")
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100

missing_df = pd.DataFrame({
    'Column': missing_data.index,
    'Missing_Count': missing_data.values,
    'Missing_Percent': missing_percent.values
}).sort_values('Missing_Count', ascending=False)

if missing_data.sum() == 0:
    print("✅ EXCELLENT: No missing values found!")
else:
    print("⚠️ Missing values detected:")
    for _, row in missing_df[missing_df['Missing_Count'] > 0].iterrows():
        print(f"   {row['Column']}: {row['Missing_Count']} ({row['Missing_Percent']:.1f}%)")

print("\n2️⃣ DUPLICATE RECORDS CHECK:")
duplicates = df.duplicated().sum()
if duplicates == 0:
    print("✅ EXCELLENT: No duplicate records found!")
else:
    print(f"⚠️ Found {duplicates} duplicate records")

print("\n3️⃣ DATA TYPES CHECK:")
print("📋 Current data types:")
for col, dtype in df.dtypes.items():
    print(f"   {col:<20}: {dtype}")

print("\n4️⃣ BASIC STATISTICS:")
display(df.describe())

print("=" * 60)
```

This is just the beginning of the enhanced notebook. Each section would continue with:

1. **Visual data exploration** with enhanced plots
2. **Feature engineering** with clear explanations
3. **Data preprocessing** with progress indicators
4. **Train/test splitting** with validation
5. **Class balancing** with before/after comparisons
6. **Summary section** with key takeaways

Would you like me to continue with the complete enhanced versions of all your notebooks?