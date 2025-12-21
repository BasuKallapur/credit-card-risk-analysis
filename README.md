# Credit Card Fraud Detection

Demo Video: [Watch Demo](https://drive.google.com/file/d/1lZ4IcC6M86f20kwWBCAvDLqeP0-Wv0Xa/view?usp=drive_link)

This project implements machine learning models for credit card fraud detection, focusing on building accurate, practical, and user-friendly solutions. The application is fully containerized with Docker for easy deployment.

## Project Overview

<img width="700" height="800" alt="image" src="https://github.com/user-attachments/assets/5ad4a56a-9bea-476c-8e14-b44492394e34" />

The project consists of:

1. **Analysis Script** (`credit_card_fraud_detection.py`): Compares four models (Logistic Regression, Decision Tree, SGD, Random Forest)
2. **Model Training** (`fraud_predictor.py`): Trains and saves the best model (Random Forest)
3. **Prediction Interface** (`predict_fraud.py`): User-friendly interface for fraud prediction

# Project Structure

```
credit-card-risk-analysis/
├── .dockerignore              # Docker ignore file
├── .gitignore
├── .python-version
├── Dockerfile                 # Docker configuration
├── app.py                     # Streamlit entry point
├── fraud_model.pkl            # Trained model
├── scaler.pkl                 # Scaler for preprocessing
├── fraud_predictor.py         # Model loading + prediction logic
├── requirements.txt           # Dependencies
├── runtime.txt                # Python version specification
├── card_transdata copy.csv    # Dataset
└── ...                        # Other files (pngs, analysis scripts)
```

---

## Model Comparison

The `credit_card_fraud_detection.py` script performs a comprehensive analysis of four different machine learning models:

1. **Logistic Regression**

   - Configuration: Strong regularization (C=0.01, 0.1), balanced class weights
   - Evaluation: Accuracy, precision, recall, F1-score, ROC curve, confusion matrix

2. **Decision Tree**

   - Configuration: Limited max_depth (3-4), high min_samples_split (50-100), high min_samples_leaf (20-50)
   - Evaluation: Accuracy, precision, recall, F1-score, ROC curve, confusion matrix

3. **SGD Classifier** (Linear SVM)

   - Configuration: L2 penalty, hinge loss, strong regularization
   - Evaluation: Accuracy, precision, recall, F1-score, ROC curve, confusion matrix

4. **Random Forest** (Best performing model)
   - Configuration: Limited tree depth, high minimum samples for splits/leaves, balanced class weights
   - Evaluation: Accuracy, precision, recall, F1-score, ROC curve, confusion matrix
   - Feature importance analysis to identify key predictors

The script generates various visualization files including:

- Confusion matrices for each model
- ROC curves showing model performance
- Feature importance chart (for Random Forest)
- Precision-recall curves
- Model comparison summary

---

## Dataset

The dataset contains the following features:

- `distance_from_home` - Distance from home where the transaction happened
- `distance_from_last_transaction` - Distance from last transaction
- `ratio_to_median_purchase_price` - Ratio of purchased price to median purchase price
- `repeat_retailer` - Is the transaction from same retailer (1 for yes, 0 for no)
- `used_chip` - Is the transaction through chip (1 for yes, 0 for no)
- `used_pin_number` - Is the transaction using PIN number (1 for yes, 0 for no)
- `online_order` - Is the transaction an online order (1 for yes, 0 for no)
- `fraud` - Is the transaction fraudulent (target variable)

## Setup

### Option 1: Using Docker (Recommended)

1. Build the Docker image:

   ```bash
   docker build -t fraud-detection .
   ```

2. Run the container:

   ```bash
   docker run -p 8501:8501 fraud-detection
   ```

3. Open your browser to [http://localhost:8501](http://localhost:8501)

### Option 2: Local Installation

1. Ensure you have Python installed (3.7+ recommended)
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Option 1: Run the Full Analysis

For a comprehensive analysis of different models:

```bash
python credit_card_fraud_detection.py
```

This generates visualizations including:

- Confusion matrices
- ROC curves
- Feature importance charts
- Model comparison charts

### Option 2: Train and Save the Model

To train and save the optimal Random Forest model:

```bash
python fraud_predictor.py
```

This creates:

- `fraud_model.pkl` - The trained model
- `scaler.pkl` - The standardization scaler

### Option 3: Predict Fraud on New Transactions

For an interactive prediction interface:

```bash
python predict_fraud.py
```

This allows you to:

- Use the sidebar to navigate between different pages
- On the Fraud Prediction page, enter transaction details to get a prediction
- On the Model Performance page, analyze model performance

## Implementation Details

The Random Forest model is configured with:

- Reduced tree depth (max_depth=5)
- Higher min_samples_split (50) and min_samples_leaf (20)
- Balanced class weights
- Square root feature selection

These parameters were chosen to minimize overfitting and ensure robust model performance.

## Module Structure

- `fraud_predictor.py` - Core model training and prediction functionality
- `predict_fraud.py` - User interface for making predictions
- `credit_card_fraud_detection.py` - Comprehensive analysis and model comparison

## Learning Outcomes

This project demonstrates:

1. How to implement machine learning for fraud detection
2. The importance of thorough feature analysis
3. Model comparison and evaluation techniques
4. Building a practical, user-friendly prediction interface
5. Containerizing ML applications with Docker

## Streamlit Dashboard (Interactive App)

This project includes an interactive dashboard built with Streamlit for real-time fraud prediction and model exploration.

### How to Launch the Dashboard

#### Using Docker (Recommended)

1. Build and run the Docker container:
   ```bash
   docker build -t fraud-detection .
   docker run -p 8501:8501 fraud-detection
   ```

#### Local Installation

1. Ensure you have trained the model and scaler files by running:
   ```bash
   python fraud_predictor.py
   ```
2. Start the dashboard:

   ```bash
   streamlit run app.py
   ```

3. Open your browser to [http://localhost:8501](http://localhost:8501) to use the app.

For detailed usage instructions, see [docs/streamlit_app_README.md](docs/streamlit_app_README.md).

### Dashboard Features

- Predict if a transaction is fraudulent
- Visualize model performance and feature importance
- Compare different models interactively
- Explore comprehensive fraud detection analytics

The dashboard uses metrics from `data/model_metrics.csv` for model performance visualizations.

---

## Presentation Guide

If you are preparing a demo or presentation, see [docs/guide.md](docs/guide.md) for tips on how to present your results and key findings.
