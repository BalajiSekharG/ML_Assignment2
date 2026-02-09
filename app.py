import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                           recall_score, f1_score, matthews_corrcoef,
                           confusion_matrix, classification_report)
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

class SimpleMLClassifier:
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'K-Nearest Neighbor': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
        }
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_train_encoded = None
        self.y_test_encoded = None
        self.results = {}
        
    def load_and_preprocess_data(self, df, target_column):
        """Load and preprocess dataset"""
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle categorical variables
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = pd.get_dummies(X[col], prefix=col, drop_first=True)
        
        # Encode target variable if it's categorical
        if y.dtype == 'object':
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = y
        
        # Check class distribution
        class_counts = pd.Series(y_encoded).value_counts()
        min_class_count = class_counts.min()
        
        if min_class_count < 2:
            st.warning(f"Found classes with only {min_class_count} member(s). This may cause issues with train-test splitting.")
            st.write("Class distribution:")
            st.write(class_counts)
            
            # Option to merge rare classes or use different splitting
            handle_rare_classes = st.selectbox(
                "How to handle rare classes?", 
                ["Use stratified splitting with warning", "Merge rare classes", "Use random splitting"]
            )
            
            if handle_rare_classes == "Merge rare classes":
                # Find classes with less than 2 members and merge them
                rare_classes = class_counts[class_counts < 2].index
                if len(rare_classes) > 0:
                    st.info(f"Merging {len(rare_classes)} rare classes into nearest common class")
                    # Find most common class
                    most_common = class_counts.idxmax()
                    # Merge rare classes with most common
                    y_encoded = np.where(np.isin(y_encoded, rare_classes), most_common, y_encoded)
                    st.success("Rare classes merged successfully")
        
        # Split data with appropriate strategy
        try:
            # Try stratified split first
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            self.y_train_encoded, self.y_test_encoded = train_test_split(
                y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
        except ValueError as e:
            st.warning(f"Stratified splitting failed: {str(e)}")
            st.info("Using random splitting instead...")
            # Fall back to random split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            self.y_train_encoded, self.y_test_encoded = train_test_split(
                y_encoded, test_size=0.2, random_state=42
            )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        return X.shape[0], X.shape[1]
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate all required evaluation metrics"""
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'F1 Score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'MCC': matthews_corrcoef(y_true, y_pred)
        }
        
        # Calculate AUC score with proper error handling
        if y_pred_proba is not None:
            try:
                unique_classes = len(np.unique(y_true))
                if unique_classes == 2:  # Binary classification
                    metrics['AUC'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:  # Multi-class classification
                    # Check if probability matrix shape matches classes
                    if y_pred_proba.shape[1] == unique_classes:
                        metrics['AUC'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
                    else:
                        st.warning(f"AUC calculation skipped: Probability shape {y_pred_proba.shape} doesn't match number of classes {unique_classes}")
                        metrics['AUC'] = 0.0
            except Exception as e:
                st.warning(f"AUC calculation failed: {str(e)}")
                metrics['AUC'] = 0.0
        else:
            metrics['AUC'] = 0.0
            
        return metrics
    
    def train_and_evaluate_all(self):
        """Train and evaluate all models"""
        for name, model in self.models.items():
            st.write(f"Training {name}...")
            
            # Use scaled data for models that benefit from it
            if name in ['Logistic Regression', 'K-Nearest Neighbor']:
                X_train, X_test = self.X_train_scaled, self.X_test_scaled
            else:
                X_train, X_test = self.X_train, self.X_test
            
            # Train model
            model.fit(X_train, self.y_train_encoded)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Get prediction probabilities for AUC
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
            else:
                y_pred_proba = None
            
            # Calculate metrics
            metrics = self.calculate_metrics(self.y_test_encoded, y_pred, y_pred_proba)
            
            # Store results
            self.results[name] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred,
                'confusion_matrix': confusion_matrix(self.y_test_encoded, y_pred),
                'classification_report': classification_report(self.y_test_encoded, y_pred, 
                                                             target_names=[str(cls) for cls in self.label_encoder.classes_] if hasattr(self.label_encoder, 'classes_') else None)
            }
        
        return self.results
    
    def get_results_table(self):
        """Create a comparison table of all model metrics"""
        table_data = []
        for model_name, result in self.results.items():
            row = {'Model': model_name}
            row.update(result['metrics'])
            table_data.append(row)
        
        return pd.DataFrame(table_data)
    
    def get_model_observations(self):
        """Generate observations about each model's performance"""
        observations = {}
        
        for model_name, result in self.results.items():
            metrics = result['metrics']
            obs = []
            
            if metrics['Accuracy'] >= 0.9:
                obs.append("Excellent accuracy performance")
            elif metrics['Accuracy'] >= 0.8:
                obs.append("Good accuracy performance")
            elif metrics['Accuracy'] >= 0.7:
                obs.append("Moderate accuracy performance")
            else:
                obs.append("Poor accuracy performance")
            
            if metrics['F1 Score'] >= 0.9:
                obs.append("Excellent balance between precision and recall")
            elif metrics['F1 Score'] >= 0.8:
                obs.append("Good balance between precision and recall")
            
            if metrics['MCC'] >= 0.7:
                obs.append("Strong correlation coefficient")
            elif metrics['MCC'] >= 0.4:
                obs.append("Moderate correlation coefficient")
            
            # Model-specific observations
            if model_name == 'Random Forest' or model_name == 'XGBoost':
                if metrics['Accuracy'] > 0.85:
                    obs.append("Ensemble method shows strong performance")
            
            if model_name == 'Logistic Regression':
                obs.append("Linear model provides good baseline")
            
            if model_name == 'Decision Tree':
                obs.append("Non-linear model captures complex patterns")
            
            observations[model_name] = "; ".join(obs)
        
        return observations
    
    def predict_new_data(self, model_name, new_data):
        """Make predictions on new data using specified model"""
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found in trained models")
        
        model = self.results[model_name]['model']
        
        # Validate new data structure
        training_features = self.X_train.shape[1]
        if hasattr(new_data, 'columns'):
            prediction_features = new_data.shape[1]
        else:
            prediction_features = new_data.shape[1]
        
        if prediction_features != training_features:
            raise ValueError(f"Feature mismatch! Training data has {training_features} features, but prediction data has {prediction_features} features.")
        
        # Preprocess new data
        if model_name in ['Logistic Regression', 'K-Nearest Neighbor']:
            new_data_scaled = self.scaler.transform(new_data)
            predictions = model.predict(new_data_scaled)
            probabilities = model.predict_proba(new_data_scaled) if hasattr(model, 'predict_proba') else None
        else:
            predictions = model.predict(new_data)
            probabilities = model.predict_proba(new_data) if hasattr(model, 'predict_proba') else None
        
        # Convert predictions back to original labels
        original_predictions = self.label_encoder.inverse_transform(predictions)
        
        return original_predictions, probabilities

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
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Load Sample Dataset", type="primary"):
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
    
    with col2:
        st.subheader("📥 Download Test Data")
        st.markdown("""
        Download sample datasets for testing the application:
        """)
        
        # Generate and provide download links for different test datasets
        if st.button("Generate Training Dataset", help="Download a sample training dataset"):
            training_data = load_sample_dataset()
            csv = training_data.to_csv(index=False)
            st.download_button(
                label="Download Training Dataset (CSV)",
                data=csv,
                file_name='sample_training_data.csv',
                mime='text/csv',
                help="Sample dataset with 1000 instances and 15 features"
            )
        
        if st.button("Generate Test Dataset", help="Download a sample test dataset"):
            # Generate a smaller test dataset
            from sklearn.datasets import make_classification
            X_test, y_test = make_classification(n_samples=200, n_features=15, n_informative=10, 
                                               n_redundant=5, n_classes=3, random_state=123)
            
            feature_names = [f'feature_{i+1}' for i in range(X_test.shape[1])]
            test_df = pd.DataFrame(X_test, columns=feature_names)
            test_df['target'] = y_test
            
            csv = test_df.to_csv(index=False)
            st.download_button(
                label="Download Test Dataset (CSV)",
                data=csv,
                file_name='sample_test_data.csv',
                mime='text/csv',
                help="Sample test dataset with 200 instances for prediction"
            )
        
        st.markdown("""
        ### 📋 Dataset Information:
        - **Training Data**: 1000 instances, 15 features, 3 classes
        - **Test Data**: 200 instances, 15 features (no target needed for prediction)
        - **Format**: CSV files ready for upload
        - **Usage**: Use training data for model training, test data for predictions
        """)

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
                
                # Validate target column type
                target_data = df[target_column]
                unique_values = target_data.nunique()
                target_dtype = target_data.dtype
                
                st.subheader("🎯 Target Column Analysis")
                st.write(f"Data type: {target_dtype}")
                st.write(f"Unique values: {unique_values}")
                st.write(f"Sample values: {target_data.dropna().head(10).tolist()}")
                
                # Check if target is suitable for classification
                if target_dtype in ['int64', 'float64'] and unique_values > 20:
                    st.warning("⚠️ Target appears to be continuous (regression problem)")
                    st.info("This app is designed for classification. Consider:")
                    st.write("1. Binning continuous values into categories")
                    st.write("2. Using a different dataset with categorical target")
                    st.write("3. Converting to discrete classes")
                    
                    # Option to bin continuous values
                    if st.checkbox("Convert continuous target to categories"):
                        num_bins = st.slider("Number of bins:", min_value=2, max_value=10, value=5)
                        df[target_column] = pd.cut(target_data, bins=num_bins, labels=False)
                        st.success(f"Converted target to {num_bins} categories")
                        st.write("New target distribution:")
                        st.write(df[target_column].value_counts())
                
                elif target_dtype == 'object' or unique_values <= 20:
                    st.success("✅ Target is suitable for classification")
                
                # Validate target column
                if df[target_column].isnull().any():
                    st.error(f"Target column '{target_column}' contains NaN values. Please clean the data first.")
                    st.stop()
                
                # Check if dataset meets requirements
                if df.shape[0] < 500:
                    st.warning(f"Dataset has only {df.shape[0]} instances (minimum 500 required)")
                
                if df.shape[1] < 13:  # target column + 12 features
                    st.warning(f"Dataset has only {df.shape[1]-1} features (minimum 12 required)")
                
                # Training options
                training_mode = st.radio(
                    "Training Mode:",
                    ["Train All Models", "Train Individual Model"],
                    help="Choose to train all models at once or select individual models"
                )
                
                if training_mode == "Train Individual Model":
                    # Individual model selection
                    available_models = ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbor', 
                                   'Naive Bayes', 'Random Forest', 'XGBoost']
                    selected_individual_model = st.selectbox("Select model to train:", available_models)
                    
                    if st.button(f"Train {selected_individual_model}", type="primary"):
                        with st.spinner(f"Training {selected_individual_model}..."):
                            # Initialize classifier
                            classifier = SimpleMLClassifier()
                            st.session_state.classifier = classifier
                            
                            # Load and preprocess data
                            instances, features = classifier.load_and_preprocess_data(df, target_column)
                            st.session_state.dataset_info = {
                                'instances': instances,
                                'features': features,
                                'target_column': target_column
                            }
                            
                            # Train single model
                            model = classifier.models[selected_individual_model]
                            
                            # Use scaled data for models that benefit from it
                            if selected_individual_model in ['Logistic Regression', 'K-Nearest Neighbor']:
                                X_train, X_test = classifier.X_train_scaled, classifier.X_test_scaled
                            else:
                                X_train, X_test = classifier.X_train, classifier.X_test
                            
                            # Train model
                            model.fit(X_train, classifier.y_train_encoded)
                            
                            # Make predictions
                            y_pred = model.predict(X_test)
                            
                            # Get prediction probabilities for AUC
                            if hasattr(model, 'predict_proba'):
                                y_pred_proba = model.predict_proba(X_test)
                            else:
                                y_pred_proba = None
                            
                            # Calculate metrics
                            metrics = classifier.calculate_metrics(
                                classifier.y_test_encoded, y_pred, y_pred_proba
                            )
                            
                            # Store results
                            st.session_state.results = {
                                selected_individual_model: {
                                    'model': model,
                                    'metrics': metrics,
                                    'predictions': y_pred,
                                    'confusion_matrix': confusion_matrix(classifier.y_test_encoded, y_pred),
                                    'classification_report': classification_report(
                                        classifier.y_test_encoded, y_pred,
                                        target_names=[str(cls) for cls in classifier.label_encoder.classes_] if hasattr(classifier.label_encoder, 'classes_') else None
                                    )
                                }
                            }
                            classifier.results = st.session_state.results
                            
                            st.success(f"{selected_individual_model} trained successfully!")
                            st.info("Go to Model Comparison page to see results.")
                
                else:  # Train All Models
                    if st.button("Train All Models", type="primary"):
                        with st.spinner("Training all models... This may take a few minutes..."):
                            # Initialize classifier
                            classifier = SimpleMLClassifier()
                            
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
        st.warning("⚠️ No trained models available!")
        st.info("Please go to 'Model Training' page first to train models before making predictions.")
        st.stop()
    else:
        classifier = st.session_state.classifier
        results = st.session_state.results
        
        if not results:
            st.warning("⚠️ No results found!")
            st.info("Please train models first on the 'Model Training' page.")
            st.stop()
        
        # Model selection
        st.subheader("🎯 Model Selection")
        selected_model = st.selectbox("Select model for prediction:", list(results.keys()))
        
        # Display selected model info
        st.info(f"Selected model: **{selected_model}**")
        
        # Data upload for prediction
        st.subheader("Upload data for prediction:")
        prediction_file = st.file_uploader("Upload CSV file for prediction", type=['csv'], key='prediction')
        
        if prediction_file is not None:
            try:
                pred_df = pd.read_csv(prediction_file)
                st.success("Prediction data loaded!")
                
                # Basic validation before processing
                st.subheader("Prediction Data Validation:")
                st.write(f"Shape: {pred_df.shape}")
                st.dataframe(pred_df.head())
                
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
                
                # Early validation with training data
                if st.session_state.get('classifier') and st.session_state.get('dataset_info'):
                    training_features = st.session_state.classifier.X_train.shape[1]
                    prediction_features = pred_df.shape[1]
                    
                    st.write(f"Training data has {training_features} features")
                    st.write(f"Prediction data has {prediction_features} features")
                    
                    # Note: Feature alignment will happen when "Make Predictions" is clicked
                
                if st.button("Make Predictions", type="primary"):
                    with st.spinner("Making predictions..."):
                        # Remove target column if present
                        if st.session_state.dataset_info['target_column'] in pred_df.columns:
                            pred_data = pred_df.drop(columns=[st.session_state.dataset_info['target_column']])
                        else:
                            pred_data = pred_df
                        
                        # Handle feature alignment first
                        if st.session_state.get('classifier') and st.session_state.get('dataset_info'):
                            # Get training feature names
                            if hasattr(st.session_state.classifier.X_train, 'columns'):
                                training_feature_names = list(st.session_state.classifier.X_train.columns)
                            else:
                                training_feature_names = [f'feature_{i}' for i in range(st.session_state.classifier.X_train.shape[1])]
                            
                            # Handle categorical variables in prediction data
                            for col in pred_data.select_dtypes(include=['object']).columns:
                                if col in training_feature_names:
                                    pred_data[col] = pd.get_dummies(pred_data[col], prefix=col, drop_first=True)
                                else:
                                    st.warning(f"Column '{col}' not found in training data. Dropping it.")
                                    pred_data = pred_data.drop(columns=[col])
                            
                            # Remove extra columns not in training data
                            extra_columns = set(pred_data.columns) - set(training_feature_names)
                            if extra_columns:
                                st.warning(f"Removing extra columns not in training data: {extra_columns}")
                                pred_data = pred_data.drop(columns=extra_columns)
                            
                            # Ensure all training features are present in prediction data
                            missing_features = set(training_feature_names) - set(pred_data.columns)
                            if missing_features:
                                st.error(f"Missing required features: {missing_features}")
                                st.stop()
                            
                            # Align columns with training data
                            pred_data = pred_data[training_feature_names]
                            
                            # Now validate feature count
                            training_features = st.session_state.classifier.X_train.shape[1]
                            prediction_features = pred_data.shape[1]
                            
                            if prediction_features != training_features:
                                st.error(f"Feature mismatch after alignment! Training has {training_features} features, prediction has {prediction_features} features.")
                                st.stop()
                            
                            # Make predictions
                            predictions, probabilities = st.session_state.classifier.predict_new_data(selected_model, pred_data)
                        
                        # Display results
                        st.subheader("🎯 Prediction Results")
                        
                        result_df = pred_df.copy()
                        result_df['Predicted'] = predictions
                        
                        if probabilities is not None:
                            # Add probability columns
                            class_names = [str(cls) for cls in st.session_state.classifier.label_encoder.classes_] if hasattr(st.session_state.classifier.label_encoder, 'classes_') else [f'Class_{i}' for i in range(probabilities.shape[1])]
                            prob_df = pd.DataFrame(probabilities, columns=class_names)
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
