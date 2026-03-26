"""
Utility functions for ML/DM Coursework
Machine Learning & Data Mining - Loan Approval Project
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Handles data preprocessing and encoding"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
    
    def handle_missing_values(self, df):
        """Handle missing values in dataset"""
        df_copy = df.copy()
        
        for col in df_copy.columns:
            if df_copy[col].isnull().sum() > 0:
                if df_copy[col].dtype in ['float64', 'int64']:
                    df_copy[col].fillna(df_copy[col].median(), inplace=True)
                else:
                    df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
        
        return df_copy
    
    def encode_categorical(self, df, categorical_cols):
        """Encode categorical variables"""
        df_copy = df.copy()
        
        for col in categorical_cols:
            if col not in ['id']:
                le = LabelEncoder()
                df_copy[col] = le.fit_transform(df_copy[col])
                self.label_encoders[col] = le
        
        return df_copy
    
    def scale_features(self, X_train, X_test=None):
        """Scale numerical features"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled


class ModelEvaluator:
    """Evaluate model performance"""
    
    @staticmethod
    def classification_metrics(y_true, y_pred, y_pred_proba=None):
        """Calculate classification metrics"""
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                     f1_score, roc_auc_score)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except:
                metrics['roc_auc'] = 0
        
        return metrics
    
    @staticmethod
    def regression_metrics(y_true, y_pred):
        """Calculate regression metrics"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }


class DataSplitter:
    """Split data for different tasks"""
    
    @staticmethod
    def split_classification_data(df_processed):
        """Split data for loan approval classification"""
        X = df_processed.drop(['id', 'loan_approval_status', 'max_allowed_loan'], axis=1)
        y = df_processed['loan_approval_status']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def split_regression_data(df_processed):
        """Split data for maximum lending value regression"""
        approved_mask = df_processed['loan_approval_status'] == 1
        X = df_processed[approved_mask].drop(['id', 'loan_approval_status', 'max_allowed_loan'], axis=1)
        y = df_processed[approved_mask]['max_allowed_loan']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test


def load_and_preprocess_data(filepath):
    """End-to-end data loading and preprocessing pipeline"""
    
    # Load data
    df = pd.read_csv(filepath)
    
    # Preprocess
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.handle_missing_values(df)
    df_processed = df_processed.drop_duplicates()
    
    # Identify and encode categorical variables
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    df_processed = preprocessor.encode_categorical(df_processed, categorical_cols)
    
    return df_processed, preprocessor


if __name__ == "__main__":
    print("✓ Utility module loaded successfully")
