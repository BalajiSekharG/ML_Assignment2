import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import os

def generate_sample_dataset():
    """Generate a sample classification dataset for testing"""
    print("Generating sample dataset...")
    
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=1000,           # Minimum 500 instances required
        n_features=15,             # Minimum 12 features required
        n_informative=10,          # Number of informative features
        n_redundant=5,             # Number of redundant features
        n_classes=3,               # Multi-class classification
        random_state=42,
        flip_y=0.05,               # Add some noise
        class_sep=1.0              # Class separation
    )
    
    # Create feature names
    feature_names = []
    for i in range(X.shape[1]):
        if i < 5:
            feature_names.append(f'numerical_feature_{i+1}')
        elif i < 10:
            feature_names.append(f'technical_feature_{i+1}')
        else:
            feature_names.append(f'business_metric_{i+1}')
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target_class'] = y
    
    # Add some categorical-like features by binning numerical features
    df['category_A'] = pd.cut(df['numerical_feature_1'], bins=3, labels=['Low', 'Medium', 'High'])
    df['category_B'] = pd.cut(df['technical_feature_1'], bins=2, labels=['Type1', 'Type2'])
    
    # Save to CSV
    output_path = os.path.join('..', 'sample_dataset.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Sample dataset saved to: {output_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:")
    print(df['target_class'].value_counts().sort_index())
    print("\nFeature names:")
    print(list(df.columns))
    
    return df

if __name__ == "__main__":
    dataset = generate_sample_dataset()
