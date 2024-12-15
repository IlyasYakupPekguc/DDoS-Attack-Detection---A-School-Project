# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_curve, f1_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib
import time

# Data Preprocessing Function
def preprocess_data(file_path):
    print("Loading dataset...")
    df = pd.read_csv(file_path)

    print("Initial dataset info:")
    print(df.info())

    print("Missing values before cleaning:")
    print(df.isnull().sum())

    # Fill missing values
    imputer = SimpleImputer(strategy='mean')
    for col in ['pktcount', 'bytecount', 'rx_kbps', 'tot_kbps']:
        df[col] = imputer.fit_transform(df[[col]])
    df['Protocol'] = df['Protocol'].fillna(df['Protocol'].mode()[0])

    # Remove outliers using IQR for pktcount
    Q1, Q3 = df['pktcount'].quantile(0.25), df['pktcount'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['pktcount'] >= Q1 - 1.5 * IQR) & (df['pktcount'] <= Q3 + 1.5 * IQR)]

    # Drop unnecessary columns
    df.drop(columns=['src', 'dst', 'time'], inplace=True, errors='ignore')

    # Encode categorical variables
    label_encoder = LabelEncoder()
    df['Protocol'] = label_encoder.fit_transform(df['Protocol'])

    # Scale numerical features
    scaler = StandardScaler()
    numeric_features = ['pktcount', 'bytecount']
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    print("Data preprocessing completed.")
    return df

# Precision-Recall Curve Function
def plot_precision_recall_curve(y_test, y_scores, model_name):
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    plt.plot(recall, precision, label=model_name)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.show()

# Load and Preprocess Data
file_path = "dataset_sdn.csv"
df = preprocess_data(file_path)

# Feature and Target Split
features = [
    'pktcount', 'bytecount', 'dur', 'tot_dur', 'flows',
    'packetins', 'pktperflow', 'byteperflow', 'pktrate',
    'Pairflow', 'port_no', 'tx_bytes', 'rx_bytes',
    'tx_kbps', 'rx_kbps', 'tot_kbps'
]
X = df[features]
y = df['label']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models and GridSearchCV
models = {
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
    },
    "XGBoost": {
        "model": XGBClassifier(eval_metric='logloss', random_state=42),
        "params": {'n_estimators': [100, 200], 'max_depth': [3, 6]}
    },
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "params": {'C': [0.1, 1, 10]}
    }
}

# Train and Evaluate Models
best_models = {}
for name, config in models.items():
    print(f"\nTraining {name} with GridSearchCV...")
    grid = GridSearchCV(config['model'], config['params'], cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    best_models[name] = grid.best_estimator_

    # Predictions
    y_pred = best_models[name].predict(X_test_scaled)
    y_scores = best_models[name].predict_proba(X_test_scaled)[:, 1] if hasattr(best_models[name], 'predict_proba') else None

    # Evaluation Metrics
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    plt.title(f"{name} - Confusion Matrix")
    plt.show()

    if y_scores is not None:
        plot_precision_recall_curve(y_test, y_scores, name)

# Save Best Models
for name, model in best_models.items():
    filename = f"{name.lower().replace(' ', '_')}_model.pkl"
    joblib.dump(model, filename)
    print(f"{name} saved as {filename}.")

print("\nTraining Complete!")