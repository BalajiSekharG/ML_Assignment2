import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import sys
import os

# Add the model directory to Python path
model_path = os.path.join(os.path.dirname(__file__), 'model')
if model_path not in sys.path:
    sys.path.insert(0, model_path)

try:
    from ml_models import MLClassifier
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error(f"Current working directory: {os.getcwd()}")
    st.error(f"Python path: {sys.path}")
    st.error(f"Model path: {model_path}")
    st.stop()

import io
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="ML Classification Models Comparison",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .model-header {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🤖 ML Classification Models Comparison</h1>', unsafe_allow_html=True)
st.markdown("""
This application demonstrates the performance of 6 different classification models on a dataset.
Upload your CSV file to see how different models perform!
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page:", ["Home", "Model Training", "Model Comparison", "Prediction"])

# Initialize session state
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'dataset_info' not in st.session_state:
    st.session_state.dataset_info = None

def load_sample_dataset():
    """Load a sample dataset for demonstration"""
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=15, n_informative=10, 
                               n_redundant=5, n_classes=3, random_state=42)
    
    # Create feature names
    feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df

if page == "Home":
    st.header("Welcome to ML Classification Models Comparison!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📊 Features:
        - **6 Classification Models**: Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost
        - **Comprehensive Metrics**: Accuracy, AUC, Precision, Recall, F1 Score, MCC
        - **Interactive Visualizations**: Confusion matrices, performance comparisons
        - **Real-time Predictions**: Upload your own data and get predictions
        
        ### 🎯 Models Implemented:
        1. **Logistic Regression**: Linear baseline model
        2. **Decision Tree**: Non-linear tree-based model
        3. **K-Nearest Neighbor**: Instance-based learning
        4. **Naive Bayes**: Probabilistic classifier
        5. **Random Forest**: Ensemble of decision trees
        6. **XGBoost**: Gradient boosting ensemble
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Evaluation Metrics:
        - Accuracy
        - AUC Score
        - Precision
        - Recall
        - F1 Score
        - Matthews Correlation Coefficient
        """)
    
    st.markdown("---")
    
    # Sample dataset option
    if st.button("Load Sample Dataset (Wine Quality)", type="primary"):
        with st.spinner("Loading sample dataset..."):
            sample_df = load_sample_dataset()
            st.session_state.sample_data = sample_df
            st.success("Sample dataset loaded! Go to Model Training to proceed.")
            
            # Display sample data info
            st.subheader("Sample Dataset Preview:")
            st.dataframe(sample_df.head())
            st.write(f"Dataset Shape: {sample_df.shape}")
            st.write(f"Target Distribution:")
            st.write(sample_df['target'].value_counts())

elif page == "Model Training":
    st.header("Model Training")
    
    # File upload section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
                
                # Data validation and cleaning
                st.subheader("Data Validation:")
                
                # Check for missing values
                missing_values = df.isnull().sum()
                if missing_values.sum() > 0:
                    st.warning(f"Found missing values in dataset:")
                    st.write(missing_values[missing_values > 0])
                    
                    # Option to handle missing values
                    handle_missing = st.selectbox("How to handle missing values?", 
                                                ["Drop rows with missing values", "Fill with mean/median/mode"])
                    
                    if handle_missing == "Drop rows with missing values":
                        df_clean = df.dropna()
                        st.info(f"Dropped {len(df) - len(df_clean)} rows with missing values")
                        df = df_clean
                    else:
                        # Fill missing values
                        for col in df.columns:
                            if df[col].dtype in ['int64', 'float64']:
                                df[col].fillna(df[col].mean(), inplace=True)
                            else:
                                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
                        st.info("Filled missing values with mean/median/mode")
                
                # Display dataset info
                st.subheader("Dataset Information:")
                st.write(f"Shape: {df.shape}")
                st.dataframe(df.head())
                
                # Target column selection
                target_column = st.selectbox("Select target column:", df.columns)
                
                # Validate target column
                if df[target_column].isnull().any():
                    st.error(f"Target column '{target_column}' contains NaN values. Please clean the data first.")
                    st.stop()
                
                # Check if dataset meets requirements
                if df.shape[0] < 500:
                    st.warning(f"Dataset has only {df.shape[0]} instances (minimum 500 required)")
                
                if df.shape[1] < 13:  # target column + 12 features
                    st.warning(f"Dataset has only {df.shape[1]-1} features (minimum 12 required)")
                
                if st.button("Train All Models", type="primary"):
                    with st.spinner("Training all models... This may take a few minutes..."):
                        # Initialize classifier
                        classifier = MLClassifier()
                        
                        # Load and preprocess data
                        instances, features = classifier.load_and_preprocess_data(df, target_column)
                        st.session_state.dataset_info = {
                            'instances': instances,
                            'features': features,
                            'target_column': target_column
                        }
                        
                        # Train and evaluate models
                        results = classifier.train_and_evaluate_all()
                        st.session_state.results = results
                        st.session_state.classifier = classifier
                        
                        st.success("All models trained successfully!")
                        st.info("Go to Model Comparison page to see the results!")
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with col2:
        st.markdown("""
        ### 📋 Dataset Requirements:
        - **Format**: CSV file
        - **Features**: Minimum 12 columns
        - **Instances**: Minimum 500 rows
        - **Target**: One categorical/numerical target column
        
        ### 💡 Tips:
        - Ensure your target column is clearly identifiable
        - Handle missing values before uploading
        - Remove unnecessary columns
        """)

elif page == "Model Comparison":
    st.header("Model Performance Comparison")
    
    if st.session_state.results is None:
        st.warning("No results available! Please train models first.")
    else:
        results = st.session_state.results
        classifier = st.session_state.classifier
        
        # Create comparison table
        comparison_df = classifier.get_results_table()
        st.subheader("📊 Performance Comparison Table")
        st.dataframe(comparison_df.style.background_gradient(cmap='Blues'), use_container_width=True)
        
        # Model selection for detailed view
        selected_model = st.selectbox("Select a model for detailed analysis:", list(results.keys()))
        
        if selected_model:
            model_result = results[selected_model]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"📈 {selected_model} - Metrics")
                metrics = model_result['metrics']
                
                for metric, value in metrics.items():
                    st.metric(metric, f"{value:.4f}")
            
            with col2:
                st.subheader("🎯 Confusion Matrix")
                cm = model_result['confusion_matrix']
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title(f'Confusion Matrix - {selected_model}')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)
                plt.close()
            
            # Classification Report
            st.subheader("📋 Classification Report")
            st.text(model_result['classification_report'])
        
        # Visual comparison
        st.subheader("📊 Visual Performance Comparison")
        
        # Create bar charts for metrics comparison
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'MCC']
        
        for metric in metrics_to_plot:
            fig = px.bar(
                x=list(results.keys()),
                y=[result['metrics'][metric] for result in results.values()],
                title=f'{metric} Comparison Across Models',
                labels={'x': 'Models', 'y': metric}
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Model observations
        st.subheader("🔍 Model Performance Observations")
        observations = classifier.get_model_observations()
        
        for model_name, observation in observations.items():
            with st.expander(f"{model_name}"):
                st.write(observation)

elif page == "Prediction":
    st.header("Make Predictions")
    
    if st.session_state.classifier is None:
        st.warning("No trained models available! Please train models first.")
    else:
        classifier = st.session_state.classifier
        results = st.session_state.results
        
        # Model selection
        selected_model = st.selectbox("Select model for prediction:", list(results.keys()))
        
        # Data upload for prediction
        st.subheader("Upload data for prediction:")
        prediction_file = st.file_uploader("Upload CSV file for prediction", type=['csv'], key='prediction')
        
        if prediction_file is not None:
            try:
                pred_df = pd.read_csv(prediction_file)
                st.success("Prediction data loaded!")
                
                # Data validation for prediction
                missing_values = pred_df.isnull().sum()
                if missing_values.sum() > 0:
                    st.warning(f"Found missing values in prediction data:")
                    st.write(missing_values[missing_values > 0])
                    
                    # Option to handle missing values
                    handle_missing = st.selectbox("How to handle missing values in prediction data?", 
                                                ["Drop rows with missing values", "Fill with mean/median/mode"],
                                                key='pred_missing')
                    
                    if handle_missing == "Drop rows with missing values":
                        pred_df_clean = pred_df.dropna()
                        st.info(f"Dropped {len(pred_df) - len(pred_df_clean)} rows with missing values")
                        pred_df = pred_df_clean
                    else:
                        # Fill missing values
                        for col in pred_df.columns:
                            if pred_df[col].dtype in ['int64', 'float64']:
                                pred_df[col].fillna(pred_df[col].mean(), inplace=True)
                            else:
                                pred_df[col].fillna(pred_df[col].mode()[0] if not pred_df[col].mode().empty else 'Unknown', inplace=True)
                        st.info("Filled missing values with mean/median/mode")
                
                st.dataframe(pred_df.head())
                
                if st.button("Make Predictions", type="primary"):
                    with st.spinner("Making predictions..."):
                        # Remove target column if present (assuming it's the last column from training)
                        if st.session_state.dataset_info['target_column'] in pred_df.columns:
                            pred_data = pred_df.drop(columns=[st.session_state.dataset_info['target_column']])
                        else:
                            pred_data = pred_df
                        
                        # Make predictions
                        predictions, probabilities = classifier.predict_new_data(selected_model, pred_data)
                        
                        # Display results
                        st.subheader("🎯 Prediction Results")
                        
                        result_df = pred_df.copy()
                        result_df['Predicted'] = predictions
                        
                        if probabilities is not None:
                            # Add probability columns
                            prob_df = pd.DataFrame(probabilities, 
                                                 columns=[f'Prob_{cls}' for cls in classifier.label_encoder.classes_])
                            result_df = pd.concat([result_df, prob_df], axis=1)
                        
                        st.dataframe(result_df)
                        
                        # Download results
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions",
                            data=csv,
                            file_name=f'predictions_{selected_model.replace(" ", "_")}.csv',
                            mime='text/csv'
                        )
                        
            except Exception as e:
                st.error(f"Error making predictions: {str(e)}")
        
        st.markdown("---")
        st.markdown("""
        ### 💡 Prediction Tips:
        - Upload data with the same features as training data
        - The target column (if present) will be automatically excluded
        - Results include predictions and probability scores
        - Download the results for further analysis
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🤖 ML Classification Models Comparison | Built with Streamlit | © 2024</p>
</div>
""", unsafe_allow_html=True)
