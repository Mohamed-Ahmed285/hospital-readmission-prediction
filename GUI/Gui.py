from tkinter import *
import customtkinter as ctk
from PIL import Image, ImageTk
import pandas as pd
from customtkinter import CTkFont
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tkinter import messagebox, ttk
import joblib
import json
import os
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek
from sklearn.utils import compute_sample_weight

# Load label mappings
with open("label_mappings.json", "r") as f:
    label_mapping = json.load(f)

CSV_PATH = "cleaned_data.csv"
MODEL_PATHS = {
    "Logistic Regression": "logistic_model.pkl",
    "SVM": "svm_model.pkl",
    "Random Forest": "rf_model.pkl",
    "XGBoost": "xgb_model.pkl"
}

# Load data and train/test split
df = pd.read_csv(CSV_PATH)
y = df['readmitted']
x = df.drop(['readmitted'], axis=1)
selector = SelectKBest(score_func=f_classif, k=15)
selector.fit(x, y)
mask = selector.get_support()
selected_features = x.columns[mask]
X_selected = df[selected_features]
ros = RandomOverSampler(random_state=42)
X_balanced, y_balanced = ros.fit_resample(X_selected, y)
x_train, x_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, stratify=y_balanced, random_state=42
)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

if not all(os.path.exists(path) for path in MODEL_PATHS.values()):
    log_model = LogisticRegression(max_iter=500, random_state=0)
    log_model.fit(x_train, y_train)
    joblib.dump((log_model, sc, selected_features.tolist()), MODEL_PATHS["Logistic Regression"])
    print("Logistic Regression accuracy", accuracy_score(y_test, log_model.predict(x_test)))
    svm_model = SVC(kernel='linear', max_iter=500, random_state=0, class_weight='balanced', probability=True)
    svm_model.fit(x_train, y_train)
    joblib.dump((svm_model, sc, selected_features.tolist()), MODEL_PATHS["SVM"])
    print("Svm accuracy", accuracy_score(y_test, svm_model.predict(x_test)))
    rf_model = RandomForestClassifier(n_estimators=150, random_state=42)
    rf_model.fit(x_train, y_train)
    joblib.dump((rf_model, sc, selected_features.tolist()), MODEL_PATHS["Random Forest"])
    print("Random accuracy", accuracy_score(y_test, rf_model.predict(x_test)))
    weights = {0: 3, 1: 2, 2: 2}
    sample_weight = y_train.map(weights)
    xgb_model = XGBClassifier(random_state=42, n_estimators=150, max_depth=6)
    xgb_model.fit(x_train, y_train, sample_weight=sample_weight)
    y_proba = xgb_model.predict_proba(x_test)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class='ovr', average='macro')
    joblib.dump((xgb_model, sc, selected_features.tolist()), MODEL_PATHS["XGBoost"])
    print("xgb accuracy", accuracy_score(y_test, xgb_model.predict(x_test)))

    print("ROC-AUC (XGB): ", round(roc_auc, 2))

    y_proba_log = log_model.predict_proba(x_test)
    roc_auc_log = roc_auc_score(y_test, y_proba_log, multi_class='ovr')
    print("ROC-AUC (Logistic Regression):", round(roc_auc_log, 2))

    y_proba_rf = rf_model.predict_proba(x_test)
    roc_auc_rf = roc_auc_score(y_test, y_proba_rf, multi_class='ovr')
    print("ROC-AUC (Random Forest):", round(roc_auc_rf, 2))

models = {}
for model_name, path in MODEL_PATHS.items():
    model, scaler, feature_names = joblib.load(path)
    models[model_name] = {"model": model, "scaler": scaler, "feature_names": feature_names}

d1_values = sorted([col[3:] for col in df.columns if col.startswith("d1_")])
d2_values = sorted([col[3:] for col in df.columns if col.startswith("d2_")])
d3_values = sorted([col[3:] for col in df.columns if col.startswith("d3_")])
def show_selected_features(model_name):
    if model_name in models:
        feature_names = models[model_name]["feature_names"]
        print(f"Selected features for {model_name}:")
        for i, feature in enumerate(feature_names, start=1):
            print(f"{i}. {feature}")
    else:
        print(f"No model found with name '{model_name}'.")

show_selected_features("Logistic Regression")
def preprocess_input(data_dict, feature_names):
    encoded_dict = {}
    yes_no_mapping = {"Yes": 1, "No": 0}
    age_ranges = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100"]
    weight_ranges = ["0-25", "25-50", "50-75", "75-100", "100-125", "125-150", "150-175", "175-200", "200+"]

    def one_hot_encode_range(value, ranges, prefix):
        one_hot = {}
        try:
            value = float(value)
        except:
            raise ValueError(f"Invalid numeric value for {prefix}: '{value}'")
        found = False
        for r in ranges:
            col_name = f"{prefix}_[{r})"
            if "+" in r:
                low = float(r.replace("+", ""))
                match = value >= low
            else:
                low, high = map(float, r.split("-"))
                match = low <= value < high
            one_hot[col_name] = 1 if match else 0
            if match:
                found = True
        if not found:
            raise ValueError(f"Value {value} for '{prefix}' does not match any known range.")
        return one_hot

    for key, value in data_dict.items():
        key_clean = key.lower().replace(" ", "_")
        if key_clean == "age":
            encoded_dict.update(one_hot_encode_range(value, age_ranges, "age"))
        elif key_clean == "weight":
            encoded_dict.update(one_hot_encode_range(value, weight_ranges, "weight"))
        elif key_clean in ["change", "diabetesmed"]:
            encoded_dict[key_clean] = yes_no_mapping.get(value, 0)
        elif key_clean in ["diagnosis_1", "diagnosis_2", "diagnosis_3"]:
            prefix = key_clean.replace("diagnosis_", "d")
            for col in feature_names:
                if col.startswith(f"{prefix}_"):
                    encoded_dict[col] = 1 if col == f"{prefix}_{value}" else 0
        else:
            mapping = label_mapping.get(key_clean)
            if mapping:
                if value not in mapping:
                    raise ValueError(f"القيمة '{value}' غير معرّفة لـ '{key}' في ملف الـ mapping.")
                encoded_dict[key_clean] = mapping[value]
            else:
                if key_clean in feature_names:
                    try:
                        encoded_dict[key_clean] = float(value)
                    except ValueError:
                        raise ValueError(f"لا يمكن تحويل القيمة '{value}' في '{key}' إلى عدد.")
    df = pd.DataFrame([encoded_dict])
    df = df.reindex(columns=feature_names, fill_value=0)
    return df

def open_main_window():
    welcome_window.destroy()
    main_window = ctk.CTk()
    main_window.title("Hospital Readmission Predictor")
    main_window.geometry("1000x800")

    ctk.CTkLabel(main_window, text="Patient Info", font=("Arial", 22)).pack(pady=10)

    fields = {}
    selected_model = ctk.StringVar(value="Random Forest")

    # Scrollable Frame
    scrollable_frame = ctk.CTkScrollableFrame(main_window, width=1100, height=650)
    scrollable_frame.pack(padx=20, pady=10, fill="both", expand=True)

    # Model dropdown
    ctk.CTkLabel(scrollable_frame, text="Select Model", font=("Arial", 18)).grid(row=0, column=0, padx=10, pady=10, sticky="w")
    model_dropdown = ctk.CTkOptionMenu(scrollable_frame, values=list(MODEL_PATHS.keys()), variable=selected_model, width=200)
    model_dropdown.grid(row=0, column=1, padx=10, pady=10, sticky="w")

    current_row = 1  # Start row count after model selection

    def create_dropdown(label_text, options, row, col):
        ctk.CTkLabel(scrollable_frame, text=label_text, font=("Arial", 12)).grid(row=row, column=col*2, padx=10, pady=5, sticky="w")
        var = ctk.StringVar(value=options[0])
        dropdown = ctk.CTkOptionMenu(scrollable_frame, values=options, variable=var, width=200)
        dropdown.grid(row=row, column=col*2+1, padx=10, pady=5, sticky="w")
        fields[label_text.lower()] = var

    def create_entry(label_text, row, col):
        ctk.CTkLabel(scrollable_frame, text=label_text, font=("Arial", 12)).grid(row=row, column=col*2, padx=10, pady=5, sticky="w")
        var = ctk.StringVar()
        entry = ctk.CTkEntry(scrollable_frame, textvariable=var, width=200)
        entry.grid(row=row, column=col*2+1, padx=10, pady=5, sticky="w")
        fields[label_text.lower()] = var

    row_count = [current_row, current_row]

    # Diagnoses
    for i, (diag_label, diag_options) in enumerate([("Diagnosis 1", d1_values), ("Diagnosis 2", d2_values), ("Diagnosis 3", d3_values)]):
        col = 0 if i % 2 == 0 else 1
        create_dropdown(diag_label, diag_options, row_count[col], col)
        row_count[col] += 1

    # Other fields
    all_fields = ["age", "weight", "change", "diabetesmed"] + [
        field for field in label_mapping.keys() if field not in ["readmitted", "change", "diabetesmed"]
    ]

    for i, field in enumerate(all_fields):
        label = field.replace("_", " ").capitalize()
        col = 0 if row_count[0] <= row_count[1] else 1
        row = row_count[col]

        if field in ["age", "weight"]:
            create_entry(label, row, col)
        elif field in ["change", "diabetesmed"]:
            ctk.CTkLabel(scrollable_frame, text=label, font=("Arial", 12)).grid(row=row, column=col*2, padx=10, pady=5, sticky="w")
            var = ctk.IntVar(value=0)
            ctk.CTkRadioButton(scrollable_frame, text="Yes", variable=var, value=1).grid(row=row, column=col*2+1, padx=5, sticky="w")
            ctk.CTkRadioButton(scrollable_frame, text="No", variable=var, value=0).grid(row=row, column=col*2+1, padx=80, sticky="w")
            fields[field] = var
        else:
            create_dropdown(label, list(label_mapping[field].keys()), row, col)

        row_count[col] += 1

    # Numeric fields
    numeric_fields = [
        "Total_visits", "Num_lab_procedure", "Num_procedure",
        "Number_diagnoses", "Time_in_hospital"
    ]

    for field in numeric_fields:
        label = field.replace("_", " ").capitalize()
        col = 0 if row_count[0] <= row_count[1] else 1
        create_entry(label, row_count[col], col)
        row_count[col] += 1

    # Predict Button
    final_row = max(row_count)

    def predict(fields, model_name):
        try:
            # Get input data from UI
            input_data = {k: v.get() for k, v in fields.items()}

            # Get model and its components
            filtered_input = {k: input_data[k] for k in selected_features if k in input_data}
            current_model = models[model_name]["model"]
            scaler = models[model_name]["scaler"]
            feature_names = models[model_name]["feature_names"]

            # Preprocess and scale input
            processed = preprocess_input(input_data, feature_names)
            processed_scaled = sc.transform(processed)

            # Make prediction
            pred = current_model.predict(processed_scaled)[0]
            prob = current_model.predict_proba(processed_scaled)[0][pred]

            # Display result
            result = "Readmitted (<30d)" if pred == 0 else "Not Readmitted"
            messagebox.showinfo("Prediction Result", f"Model: {model_name}\nResult: {result}\nConfidence: {prob:.1%}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    ctk.CTkButton(
        scrollable_frame,
        text="Predict",
        font=("Arial", 14),
        command=lambda: predict(fields, selected_model.get()),
        fg_color="darkblue",
        corner_radius=12,
        width=200,
        height=40

    ).grid(row=final_row+1, column=1, columnspan=2, pady=20,)

    main_window.mainloop()

# ---------------- نافذة البداية ----------------
welcome_window = ctk.CTk()
welcome_window.title("Welcome")
welcome_window.geometry("400x400")
ctk.CTkLabel(welcome_window, text="Hospital Readmission Predictor", font=("Arial", 22)).pack(pady=100)

# try:
#     img = Image.open("background.jpg")
#     img = img.resize((1570, 810))
#     photo = ImageTk.PhotoImage(img)
#     Label(welcome_window, image=photo).place(x=0, y=0, relwidth=1, relheight=1)
# except FileNotFoundError:
#     pass

button = ctk.CTkButton(
    master=welcome_window,
    text="Get Started",
    font=("Arial", 16),
    command=open_main_window,
    corner_radius=30,       # <-- Rounded corners!
    fg_color="blue",        # Button background
    width=160,              # Width in pixels (not characters)
    height=40               # Height in pixels
)
button.place(relx=0.5, rely=0.6, anchor=CENTER)
welcome_window.mainloop()


