# Bank Loan Risk Analysis

This project investigates loan default prediction by leveraging authentic loan data from Lending Club. It employs supervised machine learning methods, specifically emphasizing logistic regression, effective class imbalance management, and data storytelling to replicate the workflow of a standard business analyst or entry-level data scientist.

---

## Main Objectives

- Forecast whether a borrower will default or completely repay their loan  
- Investigate and illustrate borrower behavior trends  
- Address real-world data set challenges: missing values, encoding categorical variables, and handling severe class imbalances  
- Establish a solid foundation using Logistic Regression  
- Assess and contrast performance on imbalanced versus balanced data  
- Demonstrate transparent and professional reporting for stakeholder review  

---

## Overview of the Dataset

- **Data Source**: Lending Club loan data (subset cleaned from Kaggle or public archive)  
- **Original size**: 2,260,701 rows, 151 columns  
- **Processed size**: 1,265,976 rows, 16 columns  
- **Objective variable**: `loan_status`, transformed into binary:
  - Fully Paid = 0  
  - Charged Off = 1

### Key Features (16 columns)

- **Numerical**: `loan_amnt`, `int_rate`, `installment`, `annual_inc`, `dti`, `fico_range_high`, `open_acc`, `revol_util`, `total_acc`  
- **Categorical**: `term`, `emp_length`, `grade`, `home_ownership`, `purpose`  
- **Target**: `loan_default` (binary)

---

## Methodology

### 1. Data Preprocessing

- Eliminated rows with non-final loan statuses (e.g., "Current")  
- Data Conversion:
  - `term`: Changed from "36 months" to 36.0  
  - `emp_length`: Transformed from "10+ years" or "< 1 year" to numeric  
  - `revol_util`: Converted from "81.2%" to a float  
- Discarded any rows containing NaN values post-processing  
- Final dataset size after preprocessing: **1,265,976 rows**

---

### 2. Exploratory Data Analysis (EDA)

#### Boxplot: Numerical Features vs Loan Default  
![Boxplots of Numerical Features](outputs/Boxplots%20(numerical%20vs.%20loan_default).png)

Defaulters exhibited higher interest rates and lower credit scores.  
Distributions of `dti`, `revol_util`, and `annual_inc` displayed heavy right tails (outliers).

#### Countplot: Categorical Features vs Loan Default  
![Countplots of Categorical Features](outputs/Countplots%20(categorical%20vs.%20loan_default).png)

Grades E, F, and G displayed notably higher default rates.  
Loans categorized under `small_business`, `renewable_energy`, and `medical` purposes posed higher risk.

#### Correlation Matrix  
![Correlation Heatmap](outputs/Correlation%20heatmap.png)

`loan_amnt` and `installment` displayed a high correlation coefficient of 0.95.  
The target variable, `loan_default`, demonstrated weak individual correlations with features (< 0.3).

---

### 3. Encoding Categorical Variables

- Utilized `pd.get_dummies()` with `drop_first=True` for:  
  - `term`, `grade`, `home_ownership`, `purpose`  
- Final encoded dataset generated: **37 features**

---

### 4. Model Development: Logistic Regression (StatsModels)

#### Model A: Imbalanced Logistic Regression

- Trained on the entire dataset (80% non-default, 20% default)  
- **Performance metrics**:
  - Recall (default): 5.4%  
  - F1 score (default): 0.098  
  - AUC score: 0.710

_Insight: The model primarily classified loans as "Fully Paid" to optimize accuracy (80%), but inadequately identified defaulted cases._

#### Model B: Balanced Logistic Regression (Downsampled)

- Employed `sklearn.utils.resample()` to address class imbalance  
- Equal samples for default and non-default cases (247k each)  
- **Performance metrics**:
  - Recall (default): 67.1%  
  - F1 score (default): 0.658  
  - AUC score: 0.7105

---

## 5. Performance Evaluation

### A. Confusion Matrix Summary

#### Imbalanced Model (Full Dataset)
- True Positives: 13,391  
- False Negatives: 233,869  
- Recall (Defaults): 5.4%  
- Accuracy: 80.7%

#### Balanced Model (Downsampled)
- True Positives: 165,899  
- False Negatives: 81,361  
- Recall (Defaults): 67.1%  
- Accuracy: 65.2%

**Interpretation**  
- The imbalanced model achieves high accuracy by mostly predicting “Fully Paid,” but fails to catch defaulted loans.  
- The balanced model sacrifices some accuracy to identify significantly more defaults — preferred in risk-sensitive domains like lending.

---

### B. ROC Curve Comparison

#### Imbalanced Model  
![ROC Curve - Imbalanced Model](outputs/ROC%20Curve%20for%20Loan%20Default%20Prediction.png)

#### Balanced Model  
![ROC Curve - Downsampled Model](outputs/ROC%20Curve%20on%20Downsampled%20data.png)

Both models achieved similar AUC:
- Imbalanced AUC: 0.710  
- Balanced AUC: 0.7105  

> The ROC curve illustrates how well each model separates the classes across thresholds.  
> The balanced model performs better in capturing defaults at more sensitive thresholds.

---

## Business Takeaways

- The default prediction task is highly imbalanced — **naive accuracy is misleading**  
- **Downsampling** and **class weighting** should always be tested in risk models  
- Logistic regression offers **interpretable and regulatory-friendly** risk scores  
- The balanced model offers a more actionable solution despite a trade-off in overall accuracy

---
