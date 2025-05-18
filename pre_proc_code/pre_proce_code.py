import pandas as pd

df = pd.read_csv('final2.csv')
print(df.columns.tolist())


# --> for specifc column

pd.set_option('display.max_rows', None) 
print(f"Value counts for column: {'Total_visits'}")
print(df['Total_visits'].value_counts())
print("\n" + "-"*50 + "\n")

# pd.set_option('display.max_rows', None) 
# print(f"Value counts for column: {'diag_2'}")
# print(df['diag_2'].value_counts())
# print("\n" + "-"*50 + "\n")

# pd.set_option('display.max_rows', None) 
# print(f"Value counts for column: {'diag_3'}")
# print(df['diag_3'].value_counts())
# print("\n" + "-"*50 + "\n")

# for col in df.columns:
#     print(f"Value counts for column: {col}")
#     print(df[col].value_counts())
#     print("\n" + "-"*50 + "\n")

    
# for col in ['diag_1_category', 'diag_2_category', 'diag_3_category']:
#     print(f"Value counts for column: {col}")
#     print(df[col].value_counts())
#     print("\n" + "-"*50 + "\n")
# =================================================================

import pandas as pd

data = pd.read_csv('edited_df.csv')
mapping = pd.read_csv('mapping.csv')

# print (data.info())
 

admission_type_mapping = mapping.set_index('admission_type_id')['admission_type_description'].to_dict()
discharge_disposition_mapping = mapping.set_index('discharge_disposition_id')['discharge_disposition_description'].to_dict()
admission_source_mapping = mapping.set_index('admission_source_id')['admission_source'].to_dict()

data['admission_type_id'] = data['admission_type_id'].map(admission_type_mapping)
data['discharge_disposition_id'] = data['discharge_disposition_id'].map(discharge_disposition_mapping)
data['admission_source_id'] = data['admission_source_id'].map(admission_source_mapping)

data.to_csv('decoded_data.csv', index=False)
#  =========================================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('zscale2.csv')  # Change if needed

# -------------------------
# 1. Missing Values
# -------------------------
print("🔍 Missing Values:")
print(df.isnull().sum())
print("-" * 40)

# -------------------------
# 2. Duplicates
# -------------------------
duplicates = df.duplicated().sum()
print(f"🔍 Duplicated Rows: {duplicates}")
print("-" * 40)

# -------------------------
# 3. Data Types
# -------------------------
print("🔍 Data Types:")
print(df.dtypes)
print("-" * 40)

# -------------------------
# 4. Statistical Summary
# -------------------------
print("🔍 Describe Summary:")
print(df.describe())
print("-" * 40)

# -------------------------
# 5. Check Value Ranges (0–1 for normalized columns)
# Update this list to match normalized columns
normalized_cols = ['num_lab_procedures', 'num_procedures']  # <-- Example

# for col in normalized_cols:
#     print(f"🔍 Checking range for '{col}': min={df[col].min()}, max={df[col].max()}")
# print("-" * 40)

# # -------------------------
# # 6. Class Distribution (update 'target_column' if needed)
# # -------------------------
# if 'readmitted' in df.columns:  # example target
#     print("🔍 Class Distribution (readmitted):")
#     print(df['readmitted'].value_counts(normalize=True))
# print("-" * 40)

# # -------------------------
# # 7. Quick Visual Overview (Optional)
# # -------------------------
# sample_df = df.sample(min(500, len(df)), random_state=42)
# try:
#     sns.pairplot(sample_df.select_dtypes(include=['float', 'int']))
#     plt.suptitle("🔍 Pairplot Preview", y=1.02)
#     plt.show()
# except Exception as e:
#     print("Pairplot skipped due to error:", e)
    
# ==========================================

import pandas as pd


data = pd.read_csv('dropped_filled_semi_encoded.csv')

null_counts = data.isnull().sum()

null_counts = null_counts[null_counts > 0]

if null_counts.empty:
    print("No missing values")
else:
    print("Missing values per column:")
    print(null_counts)
    
# ==========================================


import pandas as pd


def categorize_diag(code):
    try:
        if pd.isna(code):
            return 'Unknown'
        code = str(code)
        if code.startswith('V'):
            return 'Supplementary classification'
        if code.startswith('E'):
            return 'External causes'
        code_num = float(code)
        if 390 <= code_num <= 459 or code_num == 785:
            return 'Circulatory'
        elif 460 <= code_num <= 519 or code_num == 786:
            return 'Respiratory'
        elif 520 <= code_num <= 579 or code_num == 787:
            return 'Digestive'
        elif 250 <= code_num < 251:
            return 'Diabetes'
        elif 800 <= code_num <= 999:
            return 'Injury'
        elif 710 <= code_num <= 739:
            return 'Musculoskeletal'
        elif 580 <= code_num <= 629 or code_num == 788:
            return 'Genitourinary'
        elif 140 <= code_num <= 239:
            return 'Neoplasms'
        else:
            return 'Other'
    except:
        return 'Unknown'
    
    
df = pd.read_csv('dropped_filled_fully_encoded.csv')

for col in ['diag_1', 'diag_2', 'diag_3']:
    df[f'{col}_category'] = df[col].apply(categorize_diag)

df.to_csv('final.csv',index=False)

for col in ['diag_1_category', 'diag_2_category', 'diag_3_category']:
        df[col] = df[col].astype(int)
        

df.to_csv('final2.csv',index=False)







# =============================================

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json

df = pd.read_csv('dropped_filled_semi_encoded.csv')

# All columns to label encode
label_encode_cols = [
    
    'gender', 
    'admission_type_id', 
    'discharge_disposition_id', 
    'admission_source_id', 
    'medical_specialty', 
    'readmitted'

]

# Dict to save mapping
label_mappings = {}

#  Encoding
for col in label_encode_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_mappings[col] = {cls: int(val) for cls, val in zip(le.classes_, le.transform(le.classes_))}

# Save mappings
with open('label_mappings.json', 'w') as f:
    json.dump(label_mappings, f, indent=4)


df.to_csv('dropped_filled_fully_encoded.csv',index=False)

# ======================================================



df = pd.read_csv('final2.csv')

# Now apply one-hot encoding
df = pd.get_dummies(df, columns=['diag_1_category', 'diag_2_category', 'diag_3_category'], prefix=['d1', 'd2', 'd3'],).astype(int)

df.to_csv('final2.csv',index=False)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++



data = pd.read_csv('decoded_data.csv')

for column in data.columns:
    mode_value = data[column].mode()[0]
    data[column].fillna(mode_value, inplace=True)

data.to_csv('decoded_output_filled.csv', index=False)

print("Missing values filled using column mode.")



# ========================================================

import pandas as pd


df = pd.read_csv('mido_minmax.csv')



cols_to_drop = [
    
]

df = df.drop(columns=cols_to_drop)

df.to_csv('mido_minmax.csv',index=False)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import pandas as pd

# Load dataset
df = pd.read_csv('final2_further_cleaned.csv')

# Define the non-encoded numeric columns to analyze
numeric_cols = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_emergency', 'number_diagnoses', 'Total_visits'
]

# Function to count outliers using IQR
def count_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return series[(series < lower) | (series > upper)].count()

# Calculate and display outlier counts
outlier_counts = {col: count_outliers(df[col]) for col in numeric_cols}
outlier_df = pd.DataFrame.from_dict(outlier_counts, orient='index', columns=['Outlier Count'])

print(outlier_df)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import pandas as pd

# Load your dataset
df = pd.read_csv('final2_further_cleaned.csv')

# List of numeric columns to cap
columns_to_cap = [
    'num_medications',
    'number_emergency',
    'number_diagnoses'
]

# Cap outliers in each column using IQR method
for col in columns_to_cap:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower, upper=upper)

# Save the fully capped dataset
df.to_csv('final2_all_done.csv', index=False)


# time_in_hospital                0
# num_lab_procedures              0
# num_procedures                  0
# num_medications              2945
# number_emergency             7413
# number_diagnoses              211
# Total_visits                    0


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
