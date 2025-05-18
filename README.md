
# Hospital Readmission Prediction for Diabetic Patients

---

## Project Overview

This project aims to build a predictive system that identifies whether a diabetic patient is likely to be readmitted to the hospital within 30 days after discharge. The data used comes from a large, real-world medical dataset containing 101,766 patient records, each with 50 features related to demographics, lab results, diagnoses, medications, and prior hospital visits.

### This is a Supervised Machine Learning project because:

- The data is labeled — each patient record includes a readmitted value that tells us the outcome (`0` if the patient was readmitted before 30 days, `1` if after 30 days, `2` if not).
- Our goal is to learn a pattern from these labeled examples and use it to predict the label (readmission status) for new patients.

### Prediction

Predict one of three labels:
- `"<30"` : 0  
- `">30"` : 1  
- `"NO"`  : 2  

---

## 1. Data Preprocessing

### Initial Inspection
- Decoded ID columns using mapping file.
- Detected both invalid and null values in several features.
- For (`diag1`, `diag2`, `diag3`), we decoded them using [ICD-9 codes](https://en.wikipedia.org/wiki/List_of_ICD-9_codes).

### Handling Missing & Invalid Values
- Invalid (non-null but incorrect) values were replaced with `null`.
- Nulls were filled using **mode imputation** (most frequent value).
- All missing values were filled by mode, except:
  - `race` and `payer code` — dropped.
- `weight` feature had mostly nulls — predicted using a Random Forest model based on `age` and `gender`.

### Encoding Categorical Features

- **One-Hot Encoding**: `age`, `weight`, `diag1`, `diag2`, `diag3`
- **Label Encoding**:  
  `gender`, `admission_type_id`, `discharge_disposition_id`, `admission_source_id`,  
  `medical_specialty`, `insulin`, `readmitted`, `metformin`, `max_glu_serum`,  
  `A1Cresult`, `repaglinide`, `nateglinide`, `glimepiride`, `glipizide`, `glyburide`,  
  `pioglitazone`, `rosiglitazone`, `glyburide-metformin`, `change`, `diabetesMed`

> All mappings were saved in a structured JSON file for reproducibility.

### Dropping Irrelevant Features
- Dropped features based on:
  - Extremely low variance
  - Too many unique values
- Combined:
  - `number_outpatient` + `number_inpatient` + `number_emergency` → `Total_visits`
  - All `weight > 125` into a single column `weight_>_125`

### Outliers Handling
- Applied **capping** to numerical features to reduce extreme values' impact.

### Standardization
- Used **z-score normalization** to scale features for consistent training.

---

## 2. Exploratory Data Analysis (EDA)

We categorized features into 5 groups:
- Numerical Features
- Categorical Features
- Medical / Treatment Features
- Diagnosis Features
- Feature Correlation & Importance

### 2.1 Numerical Features
Includes: `time_in_hospital`, `num_lab_procedures`, `num_procedures`, `Total_visits`, etc.

### 2.2 Categorical Features
- Used count plots and grouped bar charts
- Encoded using one-hot or label encoding
- Features like `admission_type_id` showed correlations with readmission

### 2.3 Medication Features
- Values: `up`, `down`, `steady`, `no`
- Analyzed change status and patterns across readmitted vs. non-readmitted patients

### 2.4 Diagnosis Features
- Grouped diagnoses from `diag1`, `diag2`, `diag3`
- Converted to binary indicators for each category
- Compared distributions between readmitted vs. non-readmitted

### 2.5 Feature Correlation & Importance
- Used:
  - Correlation matrices (for numerical, medication, and diagnosis flags)
  - Tree-based models (e.g., Random Forest) for feature importance

---

## Summary of EDA

Structured feature grouping (Num / Cat / Med / Diag) enabled deep and interpretable analysis, helping reveal key patterns critical for accurate prediction.

---

## 3. Model Development

### Preparation
- **Feature Selection**: Used `SelectKBest(f_classif)` to pick top 15 features
- **Handling Imbalance**: Applied `RandomOverSampler` to balance classes
- **Dataset Split**: 80% train / 20% test using **stratified sampling**

### Model Training
- Logistic Regression
- SVM (Linear Kernel)
- Random Forest (150 Trees)
- XGBoost (with class weights `{0:3, 1:2, 2:2}`)

### Model Evaluation
- Metrics:
  - **Classification Report** (Precision, Recall, F1-Score)
  - **ROC-AUC Score** (macro-averaged, multi-class)

### Confusion Matrix Visualization
- Visual plots for all models to show performance by class

---

## Final Summary

- **Random Forest** performed best overall — top accuracy and ROC-AUC
- Captured complex feature interactions effectively
- **Logistic Regression** and **SVM** limited by linear nature
- **XGBoost** was decent but required heavy tuning and still trailed Random Forest
