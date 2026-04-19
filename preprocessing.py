"""
Data Preprocessing Module for Bank Marketing Dataset
Handles data loading, cleaning, encoding, normalization, and dimensionality reduction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Comprehensive data preprocessing class for Bank Marketing dataset
    """
    
    def __init__(self, filepath):
        """
        Initialize preprocessor with dataset filepath
        
        Args:
            filepath (str): Path to the CSV file
        """
        self.filepath = filepath
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.pca = None
        self.label_encoders = {}
        self.feature_names = []
        
    def load_data(self):
        """Load dataset from CSV file"""
        print("📂 Loading dataset...")
        self.df = pd.read_csv(self.filepath, sep=';')
        print(f"✅ Dataset loaded successfully!")
        print(f"   Shape: {self.df.shape}")
        print(f"   Columns: {self.df.columns.tolist()}")
        return self.df
    
    def explore_data(self):
        """Explore basic dataset statistics"""
        print("\n📊 Dataset Exploration:")
        print(f"   Total rows: {len(self.df)}")
        print(f"   Total columns: {len(self.df.columns)}")
        print(f"\n   Data types:")
        print(self.df.dtypes)
        print(f"\n   Missing values:")
        print(self.df.isnull().sum())
        print(f"\n   Target variable distribution:")
        print(self.df['y'].value_counts())
        print(f"\n   First few rows:")
        print(self.df.head())
        
    def handle_unknown_values(self):
        """
        Handle 'unknown' values in categorical columns
        Replace with mode of the column
        """
        print("\n🔧 Handling 'unknown' values...")
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        unknown_count = 0
        for col in categorical_cols:
            if col != 'y':  # Don't process target variable
                unknown_mask = self.df[col] == 'unknown'
                unknown_in_col = unknown_mask.sum()
                
                if unknown_in_col > 0:
                    # Get mode (most frequent value) excluding 'unknown'
                    mode_value = self.df[self.df[col] != 'unknown'][col].mode()
                    if len(mode_value) > 0:
                        self.df.loc[unknown_mask, col] = mode_value[0]
                        unknown_count += unknown_in_col
                        print(f"   {col}: {unknown_in_col} 'unknown' values replaced with '{mode_value[0]}'")
        
        print(f"✅ Total {unknown_count} 'unknown' values handled")
        
    def remove_outliers(self):
        """
        Remove outliers using IQR method for numerical columns
        """
        print("\n🔧 Removing outliers using IQR method...")
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        initial_rows = len(self.df)
        
        for col in numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Remove outliers
            outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outliers_removed = outlier_mask.sum()
            
            if outliers_removed > 0:
                self.df = self.df[~outlier_mask]
                print(f"   {col}: {outliers_removed} outliers removed")
        
        final_rows = len(self.df)
        print(f"✅ Outlier removal complete. Rows: {initial_rows} → {final_rows} ({initial_rows - final_rows} removed)")
        
    def encode_categorical_variables(self):
        """
        Encode categorical variables using Label Encoding
        """
        print("\n🔧 Encoding categorical variables...")
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target variable from encoding list
        if 'y' in categorical_cols:
            categorical_cols.remove('y')
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
            print(f"   {col}: {len(le.classes_)} unique values encoded")
        
        # Encode target variable separately
        le_target = LabelEncoder()
        self.df['y'] = le_target.fit_transform(self.df['y'])
        self.label_encoders['y'] = le_target
        
        print(f"✅ {len(categorical_cols) + 1} categorical variables encoded")
        
    def split_features_target(self):
        """
        Split dataset into features (X) and target (y)
        """
        print("\n🔧 Splitting features and target...")
        X = self.df.drop('y', axis=1)
        y = self.df['y']
        
        self.feature_names = X.columns.tolist()
        
        print(f"✅ Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y
    
    def normalize_features(self, X_train, X_test):
        """
        Normalize numerical features using StandardScaler
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            Normalized X_train and X_test
        """
        print("\n🔧 Normalizing features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"✅ Features normalized using StandardScaler")
        return X_train_scaled, X_test_scaled
    
    def apply_pca(self, X_train, X_test, n_components=None):
        """
        Apply PCA for dimensionality reduction if features > 100
        
        Args:
            X_train: Training features
            X_test: Test features
            n_components: Number of components (default: 0.95 variance)
            
        Returns:
            Transformed X_train and X_test
        """
        n_features = X_train.shape[1]
        
        if n_features > 100:
            print(f"\n🔧 Applying PCA (features: {n_features} > 100)...")
            
            if n_components is None:
                # Keep 95% of variance
                self.pca = PCA(n_components=0.95)
            else:
                self.pca = PCA(n_components=n_components)
            
            X_train_pca = self.pca.fit_transform(X_train)
            X_test_pca = self.pca.transform(X_test)
            
            print(f"✅ PCA applied: {n_features} → {X_train_pca.shape[1]} features")
            print(f"   Explained variance: {self.pca.explained_variance_ratio_.sum():.4f}")
            
            return X_train_pca, X_test_pca
        else:
            print(f"\n⏭️  PCA skipped (features: {n_features} ≤ 100)")
            return X_train, X_test
    
    def preprocess_pipeline(self, test_size=0.2, random_state=42, apply_pca_flag=True):
        """
        Complete preprocessing pipeline
        
        Args:
            test_size (float): Proportion of test set (default: 0.2)
            random_state (int): Random seed for reproducibility
            apply_pca_flag (bool): Whether to apply PCA
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print("=" * 70)
        print("🚀 STARTING DATA PREPROCESSING PIPELINE")
        print("=" * 70)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Explore data
        self.explore_data()
        
        # Step 3: Handle unknown values
        self.handle_unknown_values()
        
        # Step 4: Remove outliers
        self.remove_outliers()
        
        # Step 5: Encode categorical variables
        self.encode_categorical_variables()
        
        # Step 6: Split features and target
        X, y = self.split_features_target()
        
        # Step 7: Train-test split
        print(f"\n🔧 Splitting data (train: {int((1-test_size)*100)}%, test: {int(test_size*100)}%)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"✅ Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Step 8: Normalize features
        X_train_scaled, X_test_scaled = self.normalize_features(X_train, X_test)
        
        # Step 9: Apply PCA if needed
        if apply_pca_flag:
            X_train_final, X_test_final = self.apply_pca(X_train_scaled, X_test_scaled)
        else:
            X_train_final, X_test_final = X_train_scaled, X_test_scaled
        
        # Store results
        self.X_train = X_train_final
        self.X_test = X_test_final
        self.y_train = y_train
        self.y_test = y_test
        
        print("\n" + "=" * 70)
        print("✅ PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\nFinal shapes:")
        print(f"   X_train: {self.X_train.shape}")
        print(f"   X_test: {self.X_test.shape}")
        print(f"   y_train: {self.y_train.shape}")
        print(f"   y_test: {self.y_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_feature_names(self):
        """Get feature names after preprocessing"""
        if self.pca is not None:
            return [f'PC{i+1}' for i in range(self.X_train.shape[1])]
        else:
            return self.feature_names


def load_and_preprocess_data(filepath='bank-additional-full.csv'):
    """
    Convenience function to load and preprocess data
    
    Args:
        filepath (str): Path to dataset
        
    Returns:
        X_train, X_test, y_train, y_test, preprocessor
    """
    preprocessor = DataPreprocessor(filepath)
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()
    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    # Test the preprocessing pipeline
    print("\n🧪 Testing preprocessing pipeline...\n")
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data()
    
    print("\n✅ Preprocessing test completed successfully!")
