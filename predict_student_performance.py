"""
predict_student_performance.py

End-to-end script to predict student academic performance using xAPI-Edu-Data.csv.

Features:
- Loads CSV (expects path /mnt/data/xAPI-Edu-Data.csv)
- Basic EDA prints
- Preprocessing: categorical encoding, scaling numeric features
- Trains a RandomForestClassifier to predict "Class" (or a fallback target if not found)
- Evaluates model (accuracy, precision, recall, f1, confusion matrix)
- Saves trained model to /mnt/data/student_perf_model.pkl

Usage:
    python predict_student_performance.py

"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = Path("/mnt/data/xAPI-Edu-Data.csv")
MODEL_PATH = Path("/mnt/data/student_perf_model.pkl")
REPORT_PATH = Path("/mnt/data/model_report.json")

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Please place the dataset at this path.")
    df = pd.read_csv(DATA_PATH)
    print("Loaded dataset:", DATA_PATH)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    # --- Choose target ---
    # Common targets in xAPI-Edu-Data are 'Class' or 'grade'. We'll prefer 'Class' then 'grade'. 
    if 'Class' in df.columns:
        target_col = 'Class'
    elif 'grade' in df.columns:
        target_col = 'grade'
    else:
        # If neither exists, try to infer a sensible numeric target: use 'raisedhands' as regression target (but we expect classification).
        raise ValueError("No column named 'Class' or 'grade' found. Please adjust target in the script.")

    print("Using target column:", target_col)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Quick preprocess: identify numeric and categorical features
    numeric_features = X.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object','category','bool']).columns.tolist()

    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    # Fill missing values if any (simple strategy)
    X[numeric_features] = X[numeric_features].fillna(X[numeric_features].median())
    X[categorical_features] = X[categorical_features].fillna("missing")

    # Build preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='drop')

    # Classifier pipeline
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Quick hyperparameter search (small grid to keep runtime modest)
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
    }

    grid = GridSearchCV(clf, param_grid, cv=3, n_jobs=-1, scoring='accuracy', verbose=1)
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    best_model = grid.best_estimator_

    # Evaluate
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Test accuracy:", acc)
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model and a small report
    joblib.dump(best_model, MODEL_PATH)
    report = {
        'best_params': grid.best_params_,
        'test_accuracy': float(acc),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    with REPORT_PATH.open('w') as f:
        json.dump(report, f, indent=2)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Report saved to {REPORT_PATH}")

if __name__ == "__main__":
    main()
