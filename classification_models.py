"""
Classification Models Module for Bank Marketing Dataset
Implements Decision Tree, Random Forest, and Logistic Regression classifiers
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class ClassificationModels:
    """
    Comprehensive classification models class
    """
    
    def __init__(self, X_train, X_test, y_train, y_test):
        """
        Initialize with train and test data
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.models = {}
        self.predictions = {}
        self.results = {}
    
    def train_decision_tree(self, max_depth=10, min_samples_split=20, random_state=42):
        """
        Train Decision Tree Classifier
        
        Args:
            max_depth (int): Maximum depth of the tree
            min_samples_split (int): Minimum samples required to split
            random_state (int): Random seed
        """
        print("\n🌳 Training Decision Tree Classifier...")
        
        dt_model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        )
        
        dt_model.fit(self.X_train, self.y_train)
        self.models['Decision Tree'] = dt_model
        
        # Make predictions
        y_pred = dt_model.predict(self.X_test)
        self.predictions['Decision Tree'] = y_pred
        
        print(f"✅ Decision Tree trained successfully!")
        
        return dt_model, y_pred
    
    def train_random_forest(self, n_estimators=100, max_depth=15, random_state=42):
        """
        Train Random Forest Classifier
        
        Args:
            n_estimators (int): Number of trees
            max_depth (int): Maximum depth of trees
            random_state (int): Random seed
        """
        print("\n🌲 Training Random Forest Classifier...")
        
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        
        rf_model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf_model
        
        # Make predictions
        y_pred = rf_model.predict(self.X_test)
        self.predictions['Random Forest'] = y_pred
        
        print(f"✅ Random Forest trained successfully!")
        
        return rf_model, y_pred
    
    def train_logistic_regression(self, max_iter=1000, random_state=42):
        """
        Train Logistic Regression Classifier
        
        Args:
            max_iter (int): Maximum iterations
            random_state (int): Random seed
        """
        print("\n📊 Training Logistic Regression...")
        
        lr_model = LogisticRegression(
            max_iter=max_iter,
            solver='lbfgs',
            random_state=random_state,
            n_jobs=-1
        )
        
        lr_model.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = lr_model
        
        # Make predictions
        y_pred = lr_model.predict(self.X_test)
        self.predictions['Logistic Regression'] = y_pred
        
        print(f"✅ Logistic Regression trained successfully!")
        
        return lr_model, y_pred
    
    def evaluate_model(self, model_name, y_pred):
        """
        Evaluate model performance
        
        Args:
            model_name (str): Name of the model
            y_pred: Predictions
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        print(f"\n📈 Evaluating {model_name}...")
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='binary')
        recall = recall_score(self.y_test, y_pred, average='binary')
        f1 = f1_score(self.y_test, y_pred, average='binary')
        
        # ROC-AUC score
        try:
            y_pred_proba = self.models[model_name].predict_proba(self.X_test)[:, 1]
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        except:
            roc_auc = None
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Store results
        results = {
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'Confusion Matrix': cm
        }
        
        self.results[model_name] = results
        
        # Print results
        print(f"\n{'='*50}")
        print(f"Results for {model_name}:")
        print(f"{'='*50}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        if roc_auc:
            print(f"ROC-AUC:   {roc_auc:.4f}")
        print(f"\nConfusion Matrix:")
        print(cm)
        print(f"{'='*50}")
        
        return results
    
    def train_and_evaluate_all(self):
        """Train and evaluate all classification models"""
        print("\n" + "="*70)
        print("🚀 TRAINING AND EVALUATING CLASSIFICATION MODELS")
        print("="*70)
        
        # Train Decision Tree
        _, dt_pred = self.train_decision_tree()
        self.evaluate_model('Decision Tree', dt_pred)
        
        # Train Random Forest
        _, rf_pred = self.train_random_forest()
        self.evaluate_model('Random Forest', rf_pred)
        
        # Train Logistic Regression
        _, lr_pred = self.train_logistic_regression()
        self.evaluate_model('Logistic Regression', lr_pred)
        
        print("\n" + "="*70)
        print("✅ ALL CLASSIFICATION MODELS TRAINED AND EVALUATED!")
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
                'Accuracy': results['Accuracy'],
                'Precision': results['Precision'],
                'Recall': results['Recall'],
                'F1-Score': results['F1-Score'],
                'ROC-AUC': results['ROC-AUC'] if results['ROC-AUC'] else 'N/A'
            })
        
        return pd.DataFrame(results_list)
    
    def plot_model_comparison(self, save_path='visualizations/classification_comparison.png'):
        """
        Plot model comparison
        
        Args:
            save_path (str): Path to save the plot
        """
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        df_results = self.get_results_dataframe()
        
        # Prepare data for plotting
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        models = df_results['Model'].tolist()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = df_results[metric].values
            ax.bar(x + i*width, values, width, label=metric, alpha=0.8)
        
        ax.set_xlabel('Models', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Classification Models Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Model comparison plot saved: {save_path}")
        plt.close()
    
    def plot_confusion_matrices(self, save_dir='visualizations'):
        """
        Plot confusion matrices for all models
        
        Args:
            save_dir (str): Directory to save plots
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, results in self.results.items():
            plt.figure(figsize=(8, 6))
            
            cm = results['Confusion Matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                       xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'],
                       square=True, linewidths=1)
            plt.xlabel('Predicted Label', fontweight='bold')
            plt.ylabel('True Label', fontweight='bold')
            plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            filename = f"{save_dir}/confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {filename}")
            plt.close()
    
    def get_feature_importance(self, model_name, feature_names=None):
        """
        Get feature importance for tree-based models
        
        Args:
            model_name (str): Name of the model
            feature_names (list): List of feature names
            
        Returns:
            array: Feature importances
        """
        model = self.models.get(model_name)
        
        if model and hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            if feature_names:
                # Create dataframe
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                print(f"\n📊 Top 10 Feature Importances - {model_name}:")
                print(importance_df.head(10))
            
            return importances
        else:
            print(f"⚠️  {model_name} does not have feature_importances_ attribute")
            return None


if __name__ == "__main__":
    # Test classification models
    print("\n🧪 Testing classification models...\n")
    
    from preprocessing import load_and_preprocess_data
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data()
    
    # Create classifier
    classifier = ClassificationModels(X_train, X_test, y_train, y_test)
    
    # Train and evaluate all models
    classifier.train_and_evaluate_all()
    
    # Get results dataframe
    results_df = classifier.get_results_dataframe()
    print("\n📊 Results Summary:")
    print(results_df.to_string(index=False))
    
    # Plot comparisons
    classifier.plot_model_comparison()
    classifier.plot_confusion_matrices()
    
    # Get feature importance
    feature_names = preprocessor.get_feature_names()
    classifier.get_feature_importance('Random Forest', feature_names)
    
    print("\n✅ Classification models test completed successfully!")
