import os
from data_processing import PaperProcessor
from feature_engineering import FeatureExtractor
from model import SemiSupervisedClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def test_data_processing():
    """Test if data processing works correctly"""
    print("\n=== Testing Data Processing ===")
    
    # Initialize processor with your dataset path
    base_path = "C:\\Users\\Aditi\\Desktop\\Machine Learning\\KDSH_2025_Dataset"
    processor = PaperProcessor(base_path)
    
    # Process papers
    papers_df, reference_df = processor.process_papers()
    
    # Print basic statistics
    print(f"\nMain Papers Dataset:")
    print(f"Number of papers: {len(papers_df)}")
    print(f"Columns: {papers_df.columns.tolist()}")
    print(f"\nSample paper text length: {len(papers_df['text'].iloc[0])}")
    
    print(f"\nReference Papers Dataset:")
    print(f"Number of papers: {len(reference_df)}")
    print(f"Number of publishable papers: {reference_df['is_publishable'].sum()}")
    print(f"Number of non-publishable papers: {len(reference_df) - reference_df['is_publishable'].sum()}")
    
    return papers_df, reference_df

def test_feature_engineering(papers_df, reference_df):
    """Test if feature engineering works correctly"""
    print("\n=== Testing Feature Engineering ===")
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    # Extract features for reference papers
    X_labeled = extractor.fit_transform(reference_df['text'].values)
    
    # Extract features for main papers
    X_unlabeled = extractor.transform(papers_df['text'].values)
    
    print(f"\nFeature matrix shape for labeled data: {X_labeled.shape}")
    print(f"Feature matrix shape for unlabeled data: {X_unlabeled.shape}")
    
    return X_labeled, X_unlabeled

def test_model(X_labeled, X_unlabeled, reference_df):
    """Test if the model works correctly"""
    print("\n=== Testing Model ===")
    
    # Prepare labels
    y_labeled = reference_df['is_publishable'].values.astype(int)
    
    # Initialize model
    model = SemiSupervisedClassifier()
    
    # Fit model
    try:
        model.fit(X_labeled, y_labeled, X_unlabeled)
        print("Model training successful")
        
        # Make predictions on labeled data
        y_pred = model.predict(X_labeled)
        
        # Print classification report
        print("\nClassification Report on Labeled Data:")
        print(classification_report(y_labeled, y_pred))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_labeled, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return model
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        return None

def main():
    # Test data processing
    papers_df, reference_df = test_data_processing()
    
    # Test feature engineering
    X_labeled, X_unlabeled = test_feature_engineering(papers_df, reference_df)
    
    # Test model
    model = test_model(X_labeled, X_unlabeled, reference_df)
    
    if model is not None:
        # Make predictions on unlabeled data
        predictions = model.predict(X_unlabeled)
        confidence_scores = model.predict_proba(X_unlabeled).max(axis=1)
        
        # Add predictions to papers_df
        papers_df['predicted_publishable'] = predictions
        papers_df['confidence'] = confidence_scores
        
        # Save predictions
        papers_df.to_csv('predictions.csv', index=False)
        
        # Print distribution of predictions
        print("\nPrediction Distribution:")
        print(papers_df['predicted_publishable'].value_counts())
        
        # Plot confidence distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=papers_df, x='confidence', hue='predicted_publishable', bins=20)
        plt.title('Distribution of Prediction Confidence')
        plt.show()

if __name__ == "__main__":
    main() 