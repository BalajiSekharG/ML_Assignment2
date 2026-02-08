"""
Test script to verify the ML models implementation
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from ml_models import MLClassifier

def test_ml_models():
    """Test the ML models implementation"""
    print("🧪 Testing ML Models Implementation...")
    
    # Generate test dataset
    print("📊 Generating test dataset...")
    X, y = make_classification(
        n_samples=600,  # Minimum 500 instances
        n_features=12,  # Minimum 12 features
        n_informative=8,
        n_redundant=4,
        n_classes=3,
        random_state=42
    )
    
    # Create DataFrame
    feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    print(f"✅ Dataset created: {df.shape}")
    print(f"📈 Target distribution: {df['target'].value_counts().sort_index().to_dict()}")
    
    # Initialize classifier
    print("\n🤖 Initializing ML Classifier...")
    classifier = MLClassifier()
    
    # Load and preprocess data
    print("⚙️ Preprocessing data...")
    instances, features = classifier.load_and_preprocess_data(df, 'target')
    print(f"📊 Processed {instances} instances with {features} features")
    
    # Train and evaluate models
    print("\n🚀 Training and evaluating models...")
    results = classifier.train_and_evaluate_all()
    
    # Display results
    print("\n📋 Results Summary:")
    print("=" * 80)
    
    comparison_df = classifier.get_results_table()
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    print("\n🔍 Model Observations:")
    print("=" * 80)
    observations = classifier.get_model_observations()
    for model, obs in observations.items():
        print(f"{model}: {obs}")
    
    print("\n✅ All tests completed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_ml_models()
        print("\n🎉 Application is ready for deployment!")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
