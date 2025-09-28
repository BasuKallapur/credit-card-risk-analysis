# Credit Card Fraud Detection System - Architecture Diagram

## Complete System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Transaction   │    │   Historical    │    │   Model Files   │
│   Data Input     │    │   Dataset       │    │   (PKL Files)   │
│   (Real-time)    │    │   (CSV File)    │    │                 │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Feature         │    │ Data Loader      │    │ Model Loader    │
│ Extraction      │    │ (Pandas)        │    │ (Joblib)        │
│                 │    │                 │    │                 │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Preprocessing Layer                      │
│  • Data Validation & Cleaning                                  │
│  • Feature Scaling (StandardScaler)                            │
│  • Train/Validation/Test Split (Stratified)                   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Machine Learning Training Layer               │
│  • Model Comparison (4 Algorithms)                              │
│  • Hyperparameter Tuning (GridSearchCV)                         │
│  • Cross-Validation & Performance Evaluation                    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Model Selection & Persistence Layer           │
│  • Best Model Selection (Random Forest)                          │
│  • Model Serialization (fraud_model.pkl)                         │
│  • Scaler Persistence (scaler.pkl)                              │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Prediction & Inference Layer                  │
│  • Real-time Feature Scaling                                   │
│  • Fraud Probability Calculation                               │
│  • Risk Factor Analysis                                         │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    User Interface Layer                          │
│  • Streamlit Web Dashboard                                      │
│  • CLI Prediction Tool                                          │
│  • Interactive Visualizations                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Model Comparison Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Random Forest   │    │ Logistic        │    │ Decision Tree   │    │ SGD Classifier  │
│ (Best Model)    │    │ Regression      │    │                 │    │                 │
│                 │    │                 │    │                 │    │                 │
│ • 99.98% Acc    │    │ • 93.49% Acc    │    │ • 99.35% Acc    │    │ • 92.71% Acc    │
│ • 0.999 AUC     │    │ • 0.979 AUC     │    │ • 0.999 AUC     │    │ • 0.979 AUC     │
│ • Feature Imp   │    │ • Linear        │    │ • Interpretable │    │ • Fast Training │
│ • Ensemble      │    │ • Baseline      │    │ • Tree-based    │    │ • Linear SVM    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │                      │
          ▼                      ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    Model Selection & Evaluation Layer                           │
│  • GridSearchCV Hyperparameter Tuning                                           │
│  • 3-Fold Cross-Validation                                                     │
│  • Performance Metrics (Accuracy, Precision, Recall, F1, AUC)                   │
│  • ROC Curves & Confusion Matrices                                             │
└─────────────────────┬───────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    Critical Data Leakage Discovery                              │
│  • Feature: ratio_to_median_purchase_price (46% correlation with fraud)        │
│  • Impact: Accuracy drops from 99.98% to 48.16% when removed                  │
│  • Significance: Single feature "gives away" the answer                         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Component File Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ credit_card_    │    │ fraud_predictor │    │ app.py          │    │ predict_fraud   │
│ fraud_detection │    │ .py             │    │ (Streamlit)     │    │ .py             │
│ .py             │    │                 │    │                 │    │                 │
│                 │    │                 │    │                 │    │                 │
│ • Main ML       │    │ • Model         │    │ • Web           │    │ • CLI           │
│   Pipeline      │    │   Training      │    │   Dashboard     │    │   Interface     │
│ • 4 Model       │    │ • Model         │    │ • Interactive   │    │ • Command Line  │
│   Comparison    │    │   Persistence   │    │   Predictions   │    │ • Batch Testing │
│ • Visualization │    │ • Prediction     │    │ • Model         │    │ • User Input    │
│   Generation    │    │   API           │    │   Performance   │    │ • Risk Analysis │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │                      │
          ▼                      ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    Shared Data & Model Files                                     │
│  • card_transdata copy.csv (1M+ transactions)                                   │
│  • fraud_model.pkl (Trained Random Forest)                                      │
│  • scaler.pkl (StandardScaler)                                                   │
│  • model_metrics.csv (Performance metrics)                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Technology Stack Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer     │    │  ML Framework   │    │ Visualization  │    │  Web Framework  │
│                 │    │                 │    │                 │    │                 │
│ • Pandas        │    │ • Scikit-learn  │    │ • Matplotlib    │    │ • Streamlit     │
│ • NumPy         │    │ • Joblib        │    │ • Seaborn       │    │ • Plotly        │
│ • CSV Files     │    │ • GridSearchCV  │    │ • Static Images │    │ • Interactive   │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │                      │
          ▼                      ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    Core Python ML Ecosystem                                     │
│  • Cross-platform compatibility                                               │
│  • Rich library ecosystem                                                      │
│  • Production-ready tools                                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Interview Explanation Guide

### **1. System Overview (30 seconds)**
*"This is a complete end-to-end machine learning system for credit card fraud detection. It processes transaction data through multiple layers: data ingestion, preprocessing, model training, persistence, and real-time prediction interfaces."*

### **2. Architecture Walkthrough (2 minutes)**
*"The architecture follows a layered approach. Data flows from CSV files through preprocessing where I apply StandardScaler and stratified splitting. Then through ML training where I compare 4 algorithms using GridSearchCV. The best model gets persisted, and finally served through multiple interfaces - web dashboard, CLI tool, and API functions."*

### **3. Key Technical Decisions (1 minute)**
*"I chose Random Forest for its balance of performance and interpretability. I used stratified sampling to handle the 8.7% fraud rate. Most importantly, I discovered critical data leakage where one feature had 46% correlation with fraud - this dropped accuracy from 99.98% to 48% when removed."*

### **4. Scalability & Production (1 minute)**
*"The current architecture is single-machine but designed for scaling. Each layer can be independently replaced - the ML layer could use distributed training, the interface layer could use microservices, and we could add streaming for real-time processing."*
