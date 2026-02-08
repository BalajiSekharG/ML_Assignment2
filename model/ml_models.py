import numpy as np
import pandas as pd
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
import warnings
warnings.filterwarnings('ignore')

class MLClassifier:
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
        """Load and preprocess the dataset"""
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
            
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.y_train_encoded, self.y_test_encoded = train_test_split(
            y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
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
        
        # Calculate AUC score
        if y_pred_proba is not None:
            if len(np.unique(y_true)) == 2:  # Binary classification
                metrics['AUC'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:  # Multi-class classification
                metrics['AUC'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
        else:
            metrics['AUC'] = 0.0
            
        return metrics
    
    def train_and_evaluate_all(self):
        """Train and evaluate all models"""
        for name, model in self.models.items():
            print(f"Training {name}...")
            
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
                                                             target_names=[str(cls) for cls in self.label_encoder.classes_])
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
