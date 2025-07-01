import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from fraud_predictor import predict_transaction
import plotly.graph_objects as go
import plotly.express as px
import json

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('fraud_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        st.error("Model files not found. Please run 'python fraud_predictor.py' first to create the model.")
        return None, None

# Function to load dataset for exploration
@st.cache_data
def load_data():
    return None, None

def create_correlation_heatmap(df):
    pass

def create_fraud_distribution(df):
    pass

def create_feature_importance_chart():
    try:
        model, _ = load_model_and_scaler()
        if model:
            feature_names = [
                'distance_from_home', 'distance_from_last_transaction',
                'ratio_to_median_purchase_price', 'repeat_retailer', 
                'used_chip', 'used_pin_number', 'online_order'
            ]
            importance = model.feature_importances_
            
            # Create a DataFrame for visualization
            feat_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values(by='Importance', ascending=False)
            
            fig = px.bar(
                feat_importance, 
                x='Importance', 
                y='Feature',
                title='Feature Importance',
                orientation='h',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            return fig
        return None
    except:
        return None

def main():
    # Load model and data
    model, scaler = load_model_and_scaler()
    data, static_stats = load_data()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page",
        ["Home", "Fraud Prediction", "Model Performance", "About"]
    )
    
    # Home page
    if page == "Home":
        st.title("Credit Card Fraud Detection Dashboard")
        st.markdown("""
        ### Welcome to the Credit Card Fraud Detection Dashboard
        
        This interactive dashboard provides tools to predict credit card fraud.
        
        **Key Features**:
        - ✅ Predict if a transaction is fraudulent
        - 🔍 Understand model features and importance
        
        **How to use**:
        - Use the sidebar to navigate between different pages
        - On the Fraud Prediction page, enter transaction details to get a prediction
        - On the Model Performance page, analyze model performance
        
        **Model Performance**:
        - This dashboard uses a Random Forest model
        - Feature engineering addresses data leakage issues
        - High accuracy in distinguishing fraudulent transactions
        """)
        st.image("https://img.freepik.com/free-vector/realistic-credit-card-design_23-2149126090.jpg", width=400)
    
    # Fraud prediction page
    elif page == "Fraud Prediction":
        st.title("Fraud Prediction")
        st.write("Enter transaction details to predict if it's fraudulent")
        
        with st.expander("ℹ️ How to use this tool", expanded=False):
            st.markdown("""
            **This tool helps you predict if a credit card transaction is fraudulent based on its characteristics.**

            1. Enter your transaction details in the input fields below, or
            2. [View and copy a sample row from the dataset here.](https://docs.google.com/spreadsheets/d/1L-hIk2OE2zhBp6Ou76TMmbeE5ihdbfvp1Uw6MyvbbN4/edit?usp=sharing)
            3. Paste the row into the quick data entry field.
            4. Click the "Predict" button to see the results.
            5. Review the fraud probability and risk factors.

            The prediction is based on a Random Forest model trained on historical transaction data.
            """)
        
        # Add quick data paste option
        st.subheader("Quick Data Entry")
        st.write("Paste a row of transaction data to automatically fill the fields below:")
        
        paste_data = st.text_area(
            "Paste data row here (space or tab separated values)",
            help="Format: distance_from_home distance_from_last_transaction ratio_to_median_purchase_price repeat_retailer used_chip used_pin_number online_order"
        )
        
        # Initialize default values
        distance_home = 50.0
        distance_last = 20.0
        ratio_median = 1.5
        repeat_retailer = 1
        used_chip = 1
        used_pin = 0
        online_order = 0
        
        # Process pasted data if available
        if paste_data:
            try:
                # Split by any whitespace (space, tab)
                values = paste_data.strip().split()
                
                # Convert values to appropriate types
                if len(values) >= 7:
                    distance_home = float(values[0])
                    distance_last = float(values[1])
                    ratio_median = float(values[2])
                    # Cap ratio_median to max allowed value to avoid StreamlitValueAboveMaxError
                    if ratio_median > 10.0:
                        st.warning(f"⚠️ Ratio to median purchase price value ({ratio_median}) exceeds maximum (10.0). Value will be capped at 10.0.")
                        ratio_median = 10.0
                    repeat_retailer = int(float(values[3]))
                    used_chip = int(float(values[4]))
                    used_pin = int(float(values[5]))
                    online_order = int(float(values[6]))
                    
                    st.success("✅ Data parsed successfully! Fields below have been updated.")
                else:
                    st.warning("⚠️ Not enough values provided. Please ensure you paste all 7 values.")
            except Exception as e:
                st.error(f"❌ Error parsing data: {str(e)}. Please check the format and try again.")
        
        # Create two columns for inputs
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Transaction Location & Amount")
            
            distance_home = st.number_input(
                "Distance from home (miles) 🏠", 
                min_value=0.0, 
                max_value=5000.0, 
                value=distance_home,
                step=1.0,
                help="The distance from the customer's home location where the transaction occurred"
            )
            
            distance_last = st.number_input(
                "Distance from last transaction (miles) 📍", 
                min_value=0.0, 
                max_value=5000.0, 
                value=distance_last,
                step=1.0,
                help="The distance from the customer's previous transaction location"
            )
            
            # Ensure ratio_median is within the allowed range for the number_input
            ratio_median = min(ratio_median, 10.0)
            ratio_median = st.number_input(
                "Ratio to median purchase price 💰", 
                min_value=0.01, 
                max_value=10.0, 
                value=ratio_median,
                step=0.01,
                help="How this transaction compares to the customer's median purchase (1.0 = same as median, 2.0 = twice the median)"
            )
        
        with col2:
            st.subheader("Transaction Characteristics")
            
            repeat_retailer_options = {"Yes": 1, "No": 0}
            repeat_retailer_display = "Yes" if repeat_retailer == 1 else "No"
            repeat_retailer_selection = st.selectbox(
                "Repeat retailer? 🔄",
                options=list(repeat_retailer_options.keys()),
                index=list(repeat_retailer_options.keys()).index(repeat_retailer_display),
                help="Is this a retailer the customer has purchased from before?"
            )
            repeat_retailer = repeat_retailer_options[repeat_retailer_selection]
            
            used_chip_options = {"Yes": 1, "No": 0}
            used_chip_display = "Yes" if used_chip == 1 else "No"
            used_chip_selection = st.selectbox(
                "Used chip? 💳", 
                options=list(used_chip_options.keys()),
                index=list(used_chip_options.keys()).index(used_chip_display),
                help="Was the card's chip used for this transaction?"
            )
            used_chip = used_chip_options[used_chip_selection]
            
            used_pin_options = {"Yes": 1, "No": 0}
            used_pin_display = "Yes" if used_pin == 1 else "No"
            used_pin_selection = st.selectbox(
                "Used PIN number? 🔢", 
                options=list(used_pin_options.keys()),
                index=list(used_pin_options.keys()).index(used_pin_display),
                help="Was a PIN number used for this transaction?"
            )
            used_pin = used_pin_options[used_pin_selection]
            
            online_order_options = {"Yes": 1, "No": 0}
            online_order_display = "Yes" if online_order == 1 else "No"
            online_order_selection = st.selectbox(
                "Online order? 🌐", 
                options=list(online_order_options.keys()),
                index=list(online_order_options.keys()).index(online_order_display),
                help="Was this transaction conducted online?"
            )
            online_order = online_order_options[online_order_selection]
        
        # Add a divider and prediction button
        st.markdown("---")
        predict_col1, predict_col2, predict_col3 = st.columns([1, 1, 1])
        with predict_col2:
            predict_button = st.button("🔍 Predict Fraud Risk", use_container_width=True)
        
        if predict_button:
            if model is not None and scaler is not None:
                # Create a spinner to show processing
                with st.spinner("Analyzing transaction..."):
                    prediction, probability = predict_transaction(
                        distance_from_home=distance_home,
                        distance_from_last_transaction=distance_last,
                        ratio_to_median_purchase_price=ratio_median,
                        repeat_retailer=repeat_retailer,
                        used_chip=used_chip,
                        used_pin_number=used_pin,
                        online_order=online_order,
                        model=model,
                        scaler=scaler
                    )
                
                # Display result with Streamlit components
                st.markdown("## Prediction Result")
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    if prediction == 1:
                        st.error("⚠️ **FRAUD ALERT!**")
                        st.markdown("""
                        <div style="padding: 10px; border-radius: 10px; background-color: rgba(255, 0, 0, 0.1);">
                            <h3 style="color: darkred;">This transaction is likely FRAUDULENT</h3>
                            <p>Our model has flagged this transaction as potentially fraudulent based on the provided details.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.success("✅ **LEGITIMATE TRANSACTION**")
                        st.markdown("""
                        <div style="padding: 10px; border-radius: 10px; background-color: rgba(0, 255, 0, 0.1);">
                            <h3 style="color: darkgreen;">This transaction appears LEGITIMATE</h3>
                            <p>Our model has not detected signs of fraud in this transaction based on the provided details.</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Display fraud probability gauge
                with result_col2:
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = probability * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Fraud Probability (%)"},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            },
                            'bar': {'color': "darkred" if probability > 0.5 else "darkgreen"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "gold"},
                                {'range': [70, 100], 'color': "lightsalmon"}
                            ]
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Insight section
                st.subheader("Risk Analysis")
                factors = []
                
                if ratio_median > 2:
                    factors.append({
                        "factor": f"Purchase amount is {ratio_median:.1f}x the median", 
                        "risk": "high",
                        "explanation": "Transactions significantly above a customer's median purchase amount are more likely to be fraudulent."
                    })
                
                if online_order == 1:
                    factors.append({
                        "factor": "Online order", 
                        "risk": "medium",
                        "explanation": "Online transactions generally have higher fraud rates than in-person transactions."
                    })
                    
                if distance_home > 100:
                    factors.append({
                        "factor": f"Transaction occurred {distance_home:.1f} miles from home", 
                        "risk": "medium",
                        "explanation": "Transactions far from a customer's home location may indicate card theft or unauthorized use."
                    })
                    
                if used_chip == 0 and used_pin == 0:
                    factors.append({
                        "factor": "No chip or PIN security features used", 
                        "risk": "high",
                        "explanation": "Transactions without chip or PIN verification have fewer security protections."
                    })
                
                if not factors:
                    st.info("✓ No significant risk factors identified")
                else:
                    for factor in factors:
                        if factor["risk"] == "high":
                            st.error(f"**{factor['factor']}** (High Risk): {factor['explanation']}")
                        elif factor["risk"] == "medium":
                            st.warning(f"**{factor['factor']}** (Medium Risk): {factor['explanation']}")
                        else:
                            st.info(f"**{factor['factor']}** (Low Risk): {factor['explanation']}")
                
                # Feature importance for this prediction
                if model is not None:
                    st.subheader("Feature Contribution Analysis")
                    features = [
                        distance_home, distance_last, ratio_median,
                        repeat_retailer, used_chip, used_pin, online_order
                    ]
                    feature_names = [
                        'Distance from home', 'Distance from last transaction',
                        'Ratio to median price', 'Repeat retailer', 
                        'Used chip', 'Used PIN', 'Online order'
                    ]
                    
                    # Convert to proper format for visualization
                    feat_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Value': features
                    })
                    
                    # Match the values with importance (simplified approach)
                    importance = model.feature_importances_
                    
                    fig = px.bar(
                        feat_df,
                        x='Value', 
                        y='Feature',
                        orientation='h',
                        title='Input Features for This Prediction',
                        color=feat_df.index,
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add a comparison to average values if data is available
                    if data is not None:
                        st.subheader("How This Transaction Compares")
                        
                        comparison_data = pd.DataFrame({
                            'Feature': feature_names,
                            'Your Transaction': features,
                            'Average Legitimate': [
                                data[data['fraud'] == 0]['distance_from_home'].mean(),
                                data[data['fraud'] == 0]['distance_from_last_transaction'].mean(),
                                data[data['fraud'] == 0]['ratio_to_median_purchase_price'].mean(),
                                data[data['fraud'] == 0]['repeat_retailer'].mean(),
                                data[data['fraud'] == 0]['used_chip'].mean(),
                                data[data['fraud'] == 0]['used_pin_number'].mean(),
                                data[data['fraud'] == 0]['online_order'].mean()
                            ],
                            'Average Fraudulent': [
                                data[data['fraud'] == 1]['distance_from_home'].mean(),
                                data[data['fraud'] == 1]['distance_from_last_transaction'].mean(),
                                data[data['fraud'] == 1]['ratio_to_median_purchase_price'].mean(),
                                data[data['fraud'] == 1]['repeat_retailer'].mean(),
                                data[data['fraud'] == 1]['used_chip'].mean(),
                                data[data['fraud'] == 1]['used_pin_number'].mean(),
                                data[data['fraud'] == 1]['online_order'].mean()
                            ]
                        })
                        
                        st.dataframe(comparison_data, use_container_width=True)
    
    # Model Performance page
    elif page == "Model Performance":
        st.title("Model Performance Analysis")
        st.markdown("""
        This page presents the performance metrics and visualizations for the four machine learning models
        that were evaluated for credit card fraud detection.
        """)

        # Load metrics from CSV
        try:
            metrics_df = pd.read_csv("data/model_metrics.csv")
        except Exception as e:
            st.error(f"Could not load model metrics: {e}")
            return

        # Tabs for the different models
        model_tabs = st.tabs([
            "Random Forest (Best)",
            "Logistic Regression",
            "Decision Tree",
            "SGD Classifier"
        ])

        # Helper to get metrics for a model
        def get_metrics_for(model_name):
            row = metrics_df[metrics_df['Model'] == model_name]
            if row.empty:
                return None
            return row.iloc[0]

        # Random Forest tab
        with model_tabs[0]:
            st.header("Random Forest Classifier")
            st.markdown("""
            The Random Forest model was selected as the best performing model for this project.
            It provided the highest accuracy and balanced precision/recall metrics.
            
            **Configuration:**
            - n_estimators: 50
            - max_depth: 5
            - min_samples_split: 50
            - min_samples_leaf: 20
            - max_features: 'sqrt'
            - class_weight: 'balanced'
            """)
            col1, col2 = st.columns(2)
            with col1:
                try:
                    st.image("assets/confusion_matrix_Random Forest.png", caption="Confusion Matrix", use_container_width=True)
                except:
                    st.warning("Confusion matrix image not found.")
            with col2:
                try:
                    st.image("assets/roc_curve_Random Forest.png", caption="ROC Curve", use_container_width=True)
                except:
                    st.warning("ROC curve image not found.")
            st.subheader("Feature Importance")
            feature_importance_fig = create_feature_importance_chart()
            if feature_importance_fig:
                st.plotly_chart(feature_importance_fig, use_container_width=True)
            st.subheader("Performance Metrics")
            rf_metrics = get_metrics_for("Random Forest")
            if rf_metrics is not None:
                metrics_rf = {
                    "Accuracy": rf_metrics["Accuracy"],
                    "Precision": rf_metrics["Precision"],
                    "Recall": rf_metrics["Recall"],
                    "F1 Score": rf_metrics["F1 Score"],
                    "AUC": rf_metrics["AUC"]
                }
                metrics_df_rf = pd.DataFrame({
                    "Metric": list(metrics_rf.keys()),
                    "Value": list(metrics_rf.values())
                })
                fig = px.bar(
                    metrics_df_rf,
                    x="Metric",
                    y="Value",
                    color="Value",
                    color_continuous_scale="Viridis",
                    title="Key Performance Metrics",
                    labels={"Value": "Score (0-1)"}
                )
                fig.update_layout(yaxis_range=[0.0, 1.0])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Metrics not found for Random Forest.")
            st.subheader("Data Leakage Analysis")
            st.markdown("""
            Despite excellent metrics, we discovered a significant data leakage issue:
            - The `ratio_to_median_purchase_price` feature had a 46% correlation with fraud
            - When we removed just this one feature, accuracy dropped from 99.98% to 48.16%
            - This indicated that a single feature was essentially "giving away" the answer
            This finding is crucial for real-world applications, as:
            1. Such a powerful single indicator might not be available in real-time
            2. Fraudsters could potentially learn to circumvent this single detection mechanism
            3. The model wasn't learning complex patterns but relying heavily on one feature
            """)

        # Logistic Regression tab
        with model_tabs[1]:
            st.header("Logistic Regression")
            st.markdown("""
            Logistic Regression provided a strong baseline model with good interpretability.
            **Configuration:**
            - C: 0.01 (strong regularization)
            - class_weight: 'balanced'
            - solver: 'liblinear'
            - max_iter: 1000
            """)
            col1, col2 = st.columns(2)
            with col1:
                try:
                    st.image("assets/confusion_matrix_Logistic Regression.png", caption="Confusion Matrix", use_container_width=True)
                except:
                    st.warning("Confusion matrix image not found.")
            with col2:
                try:
                    st.image("assets/roc_curve_Logistic Regression.png", caption="ROC Curve", use_container_width=True)
                except:
                    st.warning("ROC curve image not found.")
            st.subheader("Performance Metrics")
            lr_metrics = get_metrics_for("Logistic Regression")
            if lr_metrics is not None:
                metrics_lr = {
                    "Accuracy": lr_metrics["Accuracy"],
                    "Precision": lr_metrics["Precision"],
                    "Recall": lr_metrics["Recall"],
                    "F1 Score": lr_metrics["F1 Score"],
                    "AUC": lr_metrics["AUC"]
                }
                metrics_df_lr = pd.DataFrame({
                    "Metric": list(metrics_lr.keys()),
                    "Value": list(metrics_lr.values())
                })
                fig = px.bar(
                    metrics_df_lr,
                    x="Metric",
                    y="Value",
                    color="Value",
                    color_continuous_scale="Viridis",
                    title="Key Performance Metrics",
                    labels={"Value": "Score (0-1)"}
                )
                fig.update_layout(yaxis_range=[0.0, 1.0])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Metrics not found for Logistic Regression.")

        # Decision Tree tab
        with model_tabs[2]:
            st.header("Decision Tree")
            st.markdown("""
            The Decision Tree model offered good explainability with competitive performance.
            **Configuration:**
            - max_depth: 4
            - min_samples_split: 50
            - min_samples_leaf: 20
            - class_weight: 'balanced'
            """)
            col1, col2 = st.columns(2)
            with col1:
                try:
                    st.image("assets/confusion_matrix_Decision Tree.png", caption="Confusion Matrix", use_container_width=True)
                except:
                    st.warning("Confusion matrix image not found.")
            with col2:
                try:
                    st.image("assets/roc_curve_Decision Tree.png", caption="ROC Curve", use_container_width=True)
                except:
                    st.warning("ROC curve image not found.")
            st.subheader("Performance Metrics")
            dt_metrics = get_metrics_for("Decision Tree")
            if dt_metrics is not None:
                metrics_dt = {
                    "Accuracy": dt_metrics["Accuracy"],
                    "Precision": dt_metrics["Precision"],
                    "Recall": dt_metrics["Recall"],
                    "F1 Score": dt_metrics["F1 Score"],
                    "AUC": dt_metrics["AUC"]
                }
                metrics_df_dt = pd.DataFrame({
                    "Metric": list(metrics_dt.keys()),
                    "Value": list(metrics_dt.values())
                })
                fig = px.bar(
                    metrics_df_dt,
                    x="Metric",
                    y="Value",
                    color="Value",
                    color_continuous_scale="Viridis",
                    title="Key Performance Metrics",
                    labels={"Value": "Score (0-1)"}
                )
                fig.update_layout(yaxis_range=[0.0, 1.0])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Metrics not found for Decision Tree.")

        # SGD Classifier tab
        with model_tabs[3]:
            st.header("SGD Classifier (Linear SVM)")
            st.markdown("""
            The SGD Classifier (configured as a Linear SVM) offered fast training time with good scalability.
            **Configuration:**
            - loss: 'hinge' (SVM)
            - penalty: 'l2'
            - alpha: 0.01 (regularization strength)
            - class_weight: 'balanced'
            - max_iter: 1000
            """)
            col1, col2 = st.columns(2)
            with col1:
                try:
                    st.image("assets/confusion_matrix_SGD Classifier.png", caption="Confusion Matrix", use_container_width=True)
                except:
                    st.warning("Confusion matrix image not found.")
            with col2:
                try:
                    st.image("assets/roc_curve_SGD Classifier.png", caption="ROC Curve", use_container_width=True)
                except:
                    st.warning("ROC curve image not found.")
            st.subheader("Performance Metrics")
            sgd_metrics = get_metrics_for("SGD Classifier")
            if sgd_metrics is not None:
                metrics_sgd = {
                    "Accuracy": sgd_metrics["Accuracy"],
                    "Precision": sgd_metrics["Precision"],
                    "Recall": sgd_metrics["Recall"],
                    "F1 Score": sgd_metrics["F1 Score"],
                    "AUC": sgd_metrics["AUC"]
                }
                metrics_df_sgd = pd.DataFrame({
                    "Metric": list(metrics_sgd.keys()),
                    "Value": list(metrics_sgd.values())
                })
                fig = px.bar(
                    metrics_df_sgd,
                    x="Metric",
                    y="Value",
                    color="Value",
                    color_continuous_scale="Viridis",
                    title="Key Performance Metrics",
                    labels={"Value": "Score (0-1)"}
                )
                fig.update_layout(yaxis_range=[0.0, 1.0])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Metrics not found for SGD Classifier.")

        # Model comparison section
        st.header("Model Comparison")
        # Use the loaded metrics_df for the comparison table and radar chart
        comparison_df = metrics_df.copy()
        categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
        fig = go.Figure()
        for i, model in enumerate(comparison_df["Model"]):
            fig.add_trace(go.Scatterpolar(
                r=[comparison_df.iloc[i][cat] for cat in categories],
                theta=categories,
                fill='toself',
                name=model
            ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0.0, 1]
                )),
            title="Model Performance Comparison",
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Performance Metrics Comparison")
        st.dataframe(comparison_df, use_container_width=True)
        st.subheader("Precision-Recall Curves")
        try:
            st.image("assets/precision_recall_curve.png", caption="Precision-Recall Curves for All Models", use_container_width=True)
        except:
            st.warning("Precision-recall curve image not found.")
        st.subheader("Model Selection Rationale")
        st.markdown("""
        **Random Forest was selected as the final model for the following reasons:**
        1. Highest overall performance across all metrics
        2. Better generalization to unseen data
        3. Provides feature importance for explainability
        4. More robust to the data leakage issue
        5. Good balance between precision and recall
        While all models showed exceptional performance due to the strong signal in the 
        `ratio_to_median_purchase_price` feature, Random Forest provided the most balanced approach
        and better handling of edge cases.
        """)
    
    # About page
    elif page == "About":
        st.title("About This Project")
        st.markdown("""
        ## Credit Card Fraud Detection Project
        
        This project implements machine learning models for credit card fraud detection, 
        with a focus on identifying and addressing data leakage issues while providing a practical application.
        
        ### Key Components:
        
        1. **Analysis Script** (`credit_card_fraud_detection.py`): Compares four models (Logistic Regression, Decision Tree, SGD, Random Forest)
        2. **Model Training** (`fraud_predictor.py`): Trains and saves the best model (Random Forest)
        3. **Prediction Interface** (`predict_fraud.py`): User-friendly interface for fraud prediction
        4. **Streamlit Dashboard** (`app.py`): Interactive visualization and prediction interface
        
        ### Key Findings: Data Leakage Issue
        
        During our analysis, we discovered an important issue that's common in machine learning:
        
        - All models achieved suspiciously high accuracy (99.98% for Random Forest)
        - The `ratio_to_median_purchase_price` feature had a 46% correlation with fraud
        - When we removed just this one feature, accuracy dropped dramatically from 99.98% to 48.16%
        
        ### Dataset Features:
        
        - `distance_from_home` - Distance from home where the transaction happened
        - `distance_from_last_transaction` - Distance from last transaction
        - `ratio_to_median_purchase_price` - Ratio of purchased price to median purchase price
        - `repeat_retailer` - Is the transaction from same retailer (1 for yes, 0 for no)
        - `used_chip` - Is the transaction through chip (1 for yes, 0 for no)
        - `used_pin_number` - Is the transaction using PIN number (1 for yes, 0 for no)
        - `online_order` - Is the transaction an online order (1 for yes, 0 for no)
        - `fraud` - Is the transaction fraudulent (target variable)
        """)
        
        # Display team or acknowledgments
        st.subheader("Acknowledgments")
        st.write("This project was created as a demonstration of machine learning applications in fraud detection.")

if __name__ == "__main__":
    main() 