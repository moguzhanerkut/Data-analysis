"""
Regression Models Module for Bank Marketing Dataset
Implements Linear Regression and Ridge Regression
Uses 'duration' as target variable for regression analysis
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class RegressionModels:
    """
    Comprehensive regression models class
    """
    
    def __init__(self, X_train, X_test, y_train_duration, y_test_duration):
        """
        Initialize with train and test data
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train_duration: Training duration values
            y_test_duration: Test duration values
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train_duration
        self.y_test = y_test_duration
        
        self.models = {}
        self.predictions = {}
        self.results = {}
    
    def train_linear_regression(self):
        """Train Linear Regression model"""
        print("\n📈 Training Linear Regression...")
        
        lr_model = LinearRegression(n_jobs=-1)
        lr_model.fit(self.X_train, self.y_train)
        
        self.models['Linear Regression'] = lr_model
        
        # Make predictions
        y_pred = lr_model.predict(self.X_test)
        self.predictions['Linear Regression'] = y_pred
        
        print(f"✅ Linear Regression trained successfully!")
        
        return lr_model, y_pred
    
    def train_ridge_regression(self, alpha=1.0, random_state=42):
        """
        Train Ridge Regression model
        
        Args:
            alpha (float): Regularization strength
            random_state (int): Random seed
        """
        print("\n📈 Training Ridge Regression...")
        
        ridge_model = Ridge(alpha=alpha, random_state=random_state)
        ridge_model.fit(self.X_train, self.y_train)
        
        self.models['Ridge Regression'] = ridge_model
        
        # Make predictions
        y_pred = ridge_model.predict(self.X_test)
        self.predictions['Ridge Regression'] = y_pred
        
        print(f"✅ Ridge Regression trained successfully!")
        
        return ridge_model, y_pred
    
    def evaluate_model(self, model_name, y_pred):
        """
        Evaluate regression model performance
        
        Args:
            model_name (str): Name of the model
            y_pred: Predictions
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        print(f"\n📊 Evaluating {model_name}...")
        
        # Calculate metrics
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        # Store results
        results = {
            'Model': model_name,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R² Score': r2
        }
        
        self.results[model_name] = results
        
        # Print results
        print(f"\n{'='*50}")
        print(f"Results for {model_name}:")
        print(f"{'='*50}")
        print(f"MSE (Mean Squared Error):  {mse:.4f}")
        print(f"RMSE (Root MSE):           {rmse:.4f}")
        print(f"MAE (Mean Absolute Error): {mae:.4f}")
        print(f"R² Score:                  {r2:.4f}")
        print(f"{'='*50}")
        
        return results
    
    def train_and_evaluate_all(self):
        """Train and evaluate all regression models"""
        print("\n" + "="*70)
        print("🚀 TRAINING AND EVALUATING REGRESSION MODELS")
        print("="*70)
        
        # Train Linear Regression
        _, lr_pred = self.train_linear_regression()
        self.evaluate_model('Linear Regression', lr_pred)
        
        # Train Ridge Regression
        _, ridge_pred = self.train_ridge_regression()
        self.evaluate_model('Ridge Regression', ridge_pred)
        
        print("\n" + "="*70)
        print("✅ ALL REGRESSION MODELS TRAINED AND EVALUATED!")
        print("="*70)
    
    def get_results_dataframe(self):
        """
        Get results as a pandas DataFrame
        
        Returns:
            pd.DataFrame: Results dataframe
        """
        results_list = []
        
        for model_name, results in self.results.items():
            results_list.append({
                'Model': model_name,
                'MSE': results['MSE'],
                'RMSE': results['RMSE'],
                'MAE': results['MAE'],
                'R² Score': results['R² Score']
            })
        
        return pd.DataFrame(results_list)
    
    def plot_model_comparison(self, save_path='visualizations/regression_comparison.png'):
        """
        Plot regression model comparison
        
        Args:
            save_path (str): Path to save the plot
        """
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        df_results = self.get_results_dataframe()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        models = df_results['Model'].tolist()
        
        # MSE comparison
        axes[0, 0].bar(models, df_results['MSE'], color='#ff6b6b', alpha=0.7, edgecolor='black')
        axes[0, 0].set_ylabel('MSE', fontweight='bold')
        axes[0, 0].set_title('Mean Squared Error Comparison', fontweight='bold')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # RMSE comparison
        axes[0, 1].bar(models, df_results['RMSE'], color='#4ecdc4', alpha=0.7, edgecolor='black')
        axes[0, 1].set_ylabel('RMSE', fontweight='bold')
        axes[0, 1].set_title('Root Mean Squared Error Comparison', fontweight='bold')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # MAE comparison
        axes[1, 0].bar(models, df_results['MAE'], color='#95e1d3', alpha=0.7, edgecolor='black')
        axes[1, 0].set_ylabel('MAE', fontweight='bold')
        axes[1, 0].set_title('Mean Absolute Error Comparison', fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # R² Score comparison
        axes[1, 1].bar(models, df_results['R² Score'], color='#f38181', alpha=0.7, edgecolor='black')
        axes[1, 1].set_ylabel('R² Score', fontweight='bold')
        axes[1, 1].set_title('R² Score Comparison', fontweight='bold')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Regression comparison plot saved: {save_path}")
        plt.close()
    
    def plot_predictions_vs_actual(self, save_dir='visualizations'):
        """
        Plot predictions vs actual values for all models
        
        Args:
            save_dir (str): Directory to save plots
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, y_pred in self.predictions.items():
            plt.figure(figsize=(10, 6))
            
            # Sample 1000 points for clarity
            sample_size = min(1000, len(self.y_test))
            indices = np.random.choice(len(self.y_test), sample_size, replace=False)
            
            y_test_sample = self.y_test.iloc[indices] if hasattr(self.y_test, 'iloc') else self.y_test[indices]
            y_pred_sample = y_pred[indices]
            
            plt.scatter(y_test_sample, y_pred_sample, alpha=0.5, s=20, color='#4ecdc4', edgecolor='black', linewidth=0.5)
            
            # Plot perfect prediction line
            min_val = min(y_test_sample.min(), y_pred_sample.min())
            max_val = max(y_test_sample.max(), y_pred_sample.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
            
            plt.xlabel('Actual Duration (seconds)', fontweight='bold')
            plt.ylabel('Predicted Duration (seconds)', fontweight='bold')
            plt.title(f'Predictions vs Actual - {model_name}', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(alpha=0.3)
            
            plt.tight_layout()
            filename = f"{save_dir}/predictions_vs_actual_{model_name.lower().replace(' ', '_')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {filename}")
            plt.close()
    
    def plot_residuals(self, save_dir='visualizations'):
        """
        Plot residuals for all models
        
        Args:
            save_dir (str): Directory to save plots
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, y_pred in self.predictions.items():
            y_test_array = self.y_test.values if hasattr(self.y_test, 'values') else self.y_test
            residuals = y_test_array - y_pred
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Residuals vs Predicted
            axes[0].scatter(y_pred, residuals, alpha=0.5, s=20, color='#4ecdc4', edgecolor='black', linewidth=0.5)
            axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
            axes[0].set_xlabel('Predicted Values', fontweight='bold')
            axes[0].set_ylabel('Residuals', fontweight='bold')
            axes[0].set_title(f'Residual Plot - {model_name}', fontweight='bold')
            axes[0].grid(alpha=0.3)
            
            # Residuals distribution
            axes[1].hist(residuals, bins=50, color='#4ecdc4', alpha=0.7, edgecolor='black')
            axes[1].set_xlabel('Residuals', fontweight='bold')
            axes[1].set_ylabel('Frequency', fontweight='bold')
            axes[1].set_title(f'Residuals Distribution - {model_name}', fontweight='bold')
            axes[1].grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            filename = f"{save_dir}/residuals_{model_name.lower().replace(' ', '_')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {filename}")
            plt.close()


if __name__ == "__main__":
    # Test regression models
    print("\n🧪 Testing regression models...\n")
    
    from preprocessing import DataPreprocessor
    
    # Load and preprocess data
    preprocessor = DataPreprocessor('bank-additional-full.csv')
    preprocessor.load_data()
    preprocessor.handle_unknown_values()
    preprocessor.remove_outliers()
    
    # Extract duration before encoding
    duration = preprocessor.df['duration'].copy()
    
    # Encode and split
    preprocessor.encode_categorical_variables()
    X, y = preprocessor.split_features_target()
    
    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, duration_train, duration_test = train_test_split(
        X, y, duration, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalize
    X_train_scaled, X_test_scaled = preprocessor.normalize_features(X_train, X_test)
    
    # Create regressor
    regressor = RegressionModels(X_train_scaled, X_test_scaled, duration_train, duration_test)
    
    # Train and evaluate all models
    regressor.train_and_evaluate_all()
    
    # Get results dataframe
    results_df = regressor.get_results_dataframe()
    print("\n📊 Results Summary:")
    print(results_df.to_string(index=False))
    
    # Plot comparisons
    regressor.plot_model_comparison()
    regressor.plot_predictions_vs_actual()
    regressor.plot_residuals()
    
    print("\n✅ Regression models test completed successfully!")
