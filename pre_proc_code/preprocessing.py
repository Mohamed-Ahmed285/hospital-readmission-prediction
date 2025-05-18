import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
df = pd.read_csv("diabetic_data.csv")
print(df.head())
print("=======================================")


print(df.describe())
df['weight'] = df['weight'].replace('?', pd.NA)
print("df head=======================================")
print(df.head())
print("=======================================")

print (df.isnull().sum())
print("train weight=======================================")

dftrain=df[df['weight'].notna()].copy()
#copy() returns an original df not a view (separated df)
#training data (the nonmissing weight values only)
print(dftrain['weight'].head(10))
print("df missing weight =======================================")

dfmissingw=df[df['weight'].isna()].copy()
print(dfmissingw['weight'].head(10))
print("df info=======================================")


categorical_features=['gender','age']
#one hot encode
preprocessor = ColumnTransformer(
    transformers=
     [
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
     ]
)
pipeline = Pipeline(steps=[
   ('preprocessor', preprocessor),
   ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
x=dftrain[['gender','age']]
y=dftrain['weight']
pipeline.fit(x, y)
X_missing = dfmissingw[['gender', 'age']]
predicted_weights = pipeline.predict(X_missing)
df.loc[df['weight'].isna(), 'weight'] = predicted_weights
print(df.info())
print("dfhead=======================================")
print(df[['age','gender','weight']].head(10))
print("dftrain head=======================================")
print(dftrain[['age','gender','weight']].head(10))
print("weight nulls sum=======================================")
print(df['weight'].isna().sum())
df.replace('?',pd.NA,inplace=True)
print (df.isnull().sum())
print("===================================================================================================================================")
#============================================================================================================================
mode1=df['diag_1'].mode()
mode2=df['diag_2'].mode()
mode3=df['diag_3'].mode()
df['diag_1']=df['diag_1'].fillna(mode1[0])
df['diag_2']=df['diag_2'].fillna(mode2[0])
df['diag_3']=df['diag_3'].fillna(mode3[0])
df.drop(columns=['race'], inplace=True)
df.drop(columns=['payer_code'], inplace=True)
print (df.isnull().sum())
print(df.columns)
print("===================================================================================================================================")
fdf=pd.read_csv("dropped.csv")
fdf['Total_visits'] = fdf['number_outpatient'] + fdf['number_inpatient'] + fdf['number_emergency']
print (fdf.isnull().sum())
fdf.drop(columns=['number_outpatient'], inplace=True)
fdf.drop(columns=['number_inpatient'], inplace=True)
print(fdf.columns)
fdf.head()
fdf['change'] = fdf['change'].map({'No': 0, 'Ch': 1})
fdf['diabetesMed'] = fdf['diabetesMed'].map({'No': 0, 'Yes': 1})
print(fdf[['change', 'diabetesMed']].head())
print("===================================================================================================================================")
for col in fdf.columns:
    print(f"Value counts for column: {col}")
    print(fdf[col].value_counts())
    print("\n" + "-"*50 + "\n")
pd.set_option('display.max_rows', None)
print(f"Value counts for column: {'diag_1'}")
print(fdf['diag_1'].value_counts())
print("\n" + "-"*50 + "\n")
label_encoded=['metformin','max_glu_serum','A1Cresult','repaglinide','nateglinide','glimepiride',
               'glipizide','glyburide','pioglitazone','rosiglitazone','insulin','glyburide-metformin']
encoder=LabelEncoder()
for col in label_encoded:
    fdf[col]=encoder.fit_transform(fdf[col])
    print(col + " classes: ")
    print(encoder.classes_)
fdf.info()
print("===================================================================================================================================")
finaldf=pd.read_csv("final2.csv")
numerical_cols=['time_in_hospital','num_lab_procedures','num_procedures',
                'num_medications','number_emergency','number_diagnoses','Total_visits'
               ]
for col in numerical_cols:
  Q1=finaldf[col].quantile(0.25)
  Q3=finaldf[col].quantile(0.75)
  IQR=Q3-Q1
  lower_bound=Q1-1.5*IQR
  upper_bound=Q3+1.5*IQR
  outliers = (finaldf[col] < lower_bound) | (finaldf[col] > upper_bound)
  print(f"{col}: {outliers.sum()} outliers")
print("OUTLIERS AFTER CAPPING")
capped_df = finaldf.copy()
for col in numerical_cols:
  Q1=finaldf[col].quantile(0.25)
  Q3=finaldf[col].quantile(0.75)
  IQR=Q3-Q1
  lower_bound=Q1-1.5*IQR
  upper_bound=Q3+1.5*IQR
  capped_df[col] = capped_df[col].clip(lower=lower_bound, upper=upper_bound)
  finaldf[numerical_cols] = capped_df[numerical_cols]
  outliers = (finaldf[col] < lower_bound) | (finaldf[col] > upper_bound)
  print(f"{col}: {outliers.sum()} outliers")
finaldf.columns =  finaldf.columns.str.replace(r'\s+', '_', regex=True)
finaldf.columns =  finaldf.columns.str.replace('>', 'morethan_')
finaldf.columns =  finaldf.columns.str.replace(')', ']')
finaldf.to_csv('nooutrepcol.csv',index=False)



# print("Zscore normalization====================================================================================================================================")
# scaler = StandardScaler()
#
# fdf_numerical_scaled = pd.DataFrame(scaler.fit_transform(finaldf[numerical_cols]), columns=numerical_cols)
# finaldf2=finaldf.copy()
# finaldf2[numerical_cols] = fdf_numerical_scaled
# finaldf.drop(columns=['number_emergency'], inplace=True)
# finaldf2.drop(columns=['number_emergency'], inplace=True)
# finaldf2.to_csv('my_dataframe.csv', index=False)
# print("minmax normalization====================================================================================================================================")
# numerical_cols2=['time_in_hospital','num_lab_procedures','num_procedures',
#                 'num_medications','number_diagnoses','Total_visits'
#                ]
# df_minmax_normalized = (finaldf[numerical_cols2] - finaldf[numerical_cols2].min()) / (finaldf[numerical_cols2].max() - finaldf[numerical_cols2].min())
# finaldf[numerical_cols2] = df_minmax_normalized
# finaldf.to_csv('final2_normalized_minmaxpycharm.csv', index=False)
# print("diff outliers handling norm========================")
# print("Zscore normalization====================================================================================================================================")
# numerical_cols2=['time_in_hospital','num_lab_procedures','num_procedures','number_diagnoses','Total_visits']
# original_outfinal = pd.read_csv("final2_all_capped_rounded.csv")
# #
# #
# # scaler = StandardScaler()
# # outfinal_numerical_scaled = pd.DataFrame(scaler.fit_transform(original_outfinal[numerical_cols2]), columns=numerical_cols2)
# # outfinal = original_outfinal.copy()
# # outfinal[numerical_cols2] = outfinal_numerical_scaled
# # outfinal.drop(columns=['number_emergency','num_medications'], inplace=True)
# # outfinal.to_csv('outfinal_scaled.csv', index=False)
# # print("minmax normalization====================================================================================================================================")
# # df_minmax_normalized = (original_outfinal[numerical_cols2] - original_outfinal[numerical_cols2].min()) / (original_outfinal[numerical_cols2].max() - original_outfinal[numerical_cols2].min())
# # original_outfinal[numerical_cols2] = df_minmax_normalized
# # original_outfinal.drop(columns=['number_emergency','num_medications'], inplace=True)
# # original_outfinal.to_csv('outfinal_normalized_minmax.csv', index=False)
# original = pd.read_csv("final2_all_capped_rounded.csv")
#
# # Z-score normalization
#
# zscore_df = original_outfinal.copy()
# scaler = StandardScaler()
# zscore_df[numerical_cols2] = scaler.fit_transform(zscore_df[numerical_cols2])
# zscore_df.drop(columns=['number_emergency', 'num_medications'], inplace=True)
# zscore_df.to_csv('outfinal_scaled.csv', index=False)
# print("minmax normalization====================================================================================================================================")
# # Min-max normalization
# minmax_df = original_outfinal.copy()
# minmax_df[numerical_cols2] = (minmax_df[numerical_cols2] - minmax_df[numerical_cols2].min()) / (minmax_df[numerical_cols2].max() - minmax_df[numerical_cols2].min())
# minmax_df.drop(columns=['number_emergency', 'num_medications'], inplace=True)
# minmax_df.to_csv('outfinal_normalized_minmax.csv', index=False)