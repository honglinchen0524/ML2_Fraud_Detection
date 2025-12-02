import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, roc_auc_score

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Home", "Results"])

# Load data
@st.cache_data
def load_data():
    y_test = pd.read_csv('data/y_test.csv')['y'].values
    results = pd.read_csv('results/model_result.csv')
    probs = {
        'MLP_34': pd.read_csv('results/mlp_34.csv')['prob'].values,
        'NB_34': pd.read_csv('results/nb_34.csv')['prob'].values,
        'MLP_10': pd.read_csv('results/mlp_10.csv')['prob'].values,
        'NB_10': pd.read_csv('results/nb_10.csv')['prob'].values,
    }
    probs['Ensemble_34'] = (probs['MLP_34'] + probs['NB_34']) / 2
    probs['Ensemble_10'] = (probs['MLP_10'] + probs['NB_10']) / 2
    return y_test, results, probs

# Home
if page == "Home":
    st.title("Credit Card Fraud Detection")
    st.subheader("Hybrid Ensemble using MLP and Naive Bayes")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Multi-Layer Perceptron Model")
        st.markdown("""
        - **Architecture:** 30 -> 16 -> 1
        - **Hidden activation:** ReLU
        - **Output activation:** Sigmoid
        - **Loss:** Binary Cross-Entropy
        - **Optimizer:** Gradient Descent η = 0.01
        - **Regularization:** Early stopping patience = 50
        """)
    
    with col2:
        st.markdown("### Naive Bayes Model")
        st.markdown("""
        - **Variant:** Gaussian Naive Bayes
        - **Assumption:** Features conditionally independent given class
        - **Estimation:** Maximum likelihood for P(X|Y)
        - **Prediction:** Bayes' theorem for P(Y|X)
        """)
    
    st.markdown("---")
    
    st.markdown("### Ensemble Method: Soft Voting")
    st.markdown("""
    Our ensemble combines both models by averaging their predicted probabilities:
    
    $$P_{ensemble} = \\frac{P_{MLP} + P_{NB}}{2}$$
    
    Final prediction is **Fraud** if $P_{ensemble} \\geq 0.5$, else **Legitimate**.
    """)
    
    st.markdown("---")
    
    st.markdown("### Data Preprocessing")
    st.markdown("""
    Following [Awoyemi et al. (2017)](https://ieeexplore.ieee.org/document/8093247):
    - **Dataset:** ULB Credit Card Fraud (284,807 transactions, 0.17% fraud)
    - **Split:** 70% train, 30% test (stratified)
    - **Sampling:** Hybrid MOTE oversampling + random undersampling
    - **Distributions tested:** 34:66 and 10:90 (fraud:legitimate)
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Team Members")
        st.markdown("""
        - Honglin Chen
        - Kerissa Duliga
        - Vaibhav Singh
        """)
    
    with col2:
        st.markdown("### Source Code")
        st.markdown("[GitHub Repository](https://github.com/honglinchen0524/ML2_Fraud_Detection/tree/main)")

# Results
elif page == "Results":
    st.title("Model Results")
    
    y_test, results, probs = load_data()
    
    # Filters
    st.sidebar.markdown("### Filters")
    dist_filter = st.sidebar.selectbox("Distribution", ["All", "34:66", "10:90"])
    model_filter = st.sidebar.multiselect("Models", ["MLP", "Naive Bayes", "Ensemble"], default=["MLP", "Naive Bayes", "Ensemble"])
    
    # Filter results table
    df_display = results.copy()
    if dist_filter != "All":
        df_display = df_display[df_display['Distribution'] == dist_filter]
    if model_filter:
        df_display = df_display[df_display['Model'].isin(model_filter)]
    
    # Metrics Table
    st.markdown("### Metrics Comparison")
    st.dataframe(
        df_display[['Distribution', 'Model', 'Sensitivity', 'Precision', 'MCC', 'AUC-ROC']].round(4),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    
    # ROC Curves
    st.markdown("### ROC Curves")
    
    colors = {'MLP': 'blue', 'Naive Bayes': 'green', 'Ensemble': 'red'}
    model_keys = {'MLP': 'MLP', 'Naive Bayes': 'NB', 'Ensemble': 'Ensemble'}
    
    col1, col2 = st.columns(2)
    
    # ROC 34:66
    with col1:
        if dist_filter in ["All", "34:66"]:
            fig = go.Figure()
            for model in model_filter:
                key = f"{model_keys[model]}_34"
                fpr, tpr, _ = roc_curve(y_test, probs[key])
                auc = roc_auc_score(y_test, probs[key])
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{model} (AUC={auc:.3f})', line=dict(color=colors[model])))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(color='gray', dash='dash')))
            fig.update_layout(title='ROC Curve (34:66)', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # ROC 10:90
    with col2:
        if dist_filter in ["All", "10:90"]:
            fig = go.Figure()
            for model in model_filter:
                key = f"{model_keys[model]}_10"
                fpr, tpr, _ = roc_curve(y_test, probs[key])
                auc = roc_auc_score(y_test, probs[key])
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{model} (AUC={auc:.3f})', line=dict(color=colors[model])))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(color='gray', dash='dash')))
            fig.update_layout(title='ROC Curve (10:90)', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', height=400)
            st.plotly_chart(fig, use_container_width=True)
