# Machine Learning Assignment 2 - Model Training Script
# Student: Prabhat Sahu
# M.Tech AIML - BITS Pilani

import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

def main():
    print("Loading Breast Cancer Wisconsin dataset from sklearn...")
    cancer = load_breast_cancer(as_frame=True)
    df = cancer.frame
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features count: {len(cancer.feature_names)}")
    
    # Save full dataset copy for reference
    data_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(data_dir)
    
    # Features and target
    X = df.drop(columns=['target'])
    y = df['target']
    
    # 80-20 train test split with fixed random seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save test dataset to root directory as required by assignment
    test_df = X_test.copy()
    test_df['target'] = y_test
    test_csv_path = os.path.join(project_dir, 'test_data.csv')
    test_df.to_csv(test_csv_path, index=False)
    print(f"Saved test dataset to {test_csv_path} ({len(test_df)} rows)")
    
    # Feature scaling for distance and gradient based models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    scaler_path = os.path.join(data_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print("Saved feature scaler.")
    
    # Define the 5 classification models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Support Vector Machine': SVC(probability=True, random_state=42)
    }
    
    # Model filenames mapping
    file_names = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'KNN': 'knn.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
        'Support Vector Machine': 'svm.pkl'
    }
    
    metrics_summary = {}
    
    print("\nTraining models and evaluating metrics on test set:")
    for name, model in models.items():
        # Train model
        if name == 'Decision Tree':
            # Decision Tree does not strictly require scaled features
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            
        # Calculate all 6 required evaluation metrics
        acc = float(accuracy_score(y_test, y_pred))
        auc = float(roc_auc_score(y_test, y_prob))
        prec = float(precision_score(y_test, y_pred))
        rec = float(recall_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))
        mcc = float(matthews_corrcoef(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred).tolist()
        cr = classification_report(y_test, y_pred, target_names=['Malignant (0)', 'Benign (1)'], output_dict=True)
        
        # Save trained model artifact
        model_save_path = os.path.join(data_dir, file_names[name])
        joblib.dump(model, model_save_path)
        
        metrics_summary[name] = {
            'Accuracy': round(acc, 4),
            'AUC': round(auc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1': round(f1, 4),
            'MCC': round(mcc, 4),
            'confusion_matrix': cm,
            'classification_report': cr
        }
        
        print(f"-> {name}: Accuracy={acc:.4f}, AUC={auc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, MCC={mcc:.4f}")
        
    # Save metrics json for quick loading in app
    metrics_json_path = os.path.join(data_dir, 'metrics.json')
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
        
    print(f"\nAll 5 models trained and saved successfully in {data_dir}")

if __name__ == '__main__':
    main()
