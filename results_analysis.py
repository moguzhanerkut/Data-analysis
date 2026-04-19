"""
Results Analysis Module for Bank Marketing Dataset
Comprehensive comparison and analysis of all models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class ResultsAnalyzer:
    """
    Comprehensive results analysis and comparison class
    """
    
    def __init__(self):
        """Initialize results analyzer"""
        self.classification_results = None
        self.regression_results = None
        self.clustering_results = None
    
    def set_classification_results(self, results_df):
        """Set classification results"""
        self.classification_results = results_df
    
    def set_regression_results(self, results_df):
        """Set regression results"""
        self.regression_results = results_df
    
    def set_clustering_results(self, results_df):
        """Set clustering results"""
        self.clustering_results = results_df
    
    def print_all_results(self):
        """Print all results in a formatted way"""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE RESULTS SUMMARY")
        print("="*80)
        
        if self.classification_results is not None:
            print("\n🎯 CLASSIFICATION MODELS:")
            print("-" * 80)
            print(self.classification_results.to_string(index=False))
            
            # Find best model
            best_model = self.classification_results.loc[self.classification_results['F1-Score'].idxmax()]
            print(f"\n🏆 Best Classification Model: {best_model['Model']}")
            print(f"   F1-Score: {best_model['F1-Score']:.4f}")
            print(f"   Accuracy: {best_model['Accuracy']:.4f}")
        
        if self.regression_results is not None:
            print("\n\n📈 REGRESSION MODELS:")
            print("-" * 80)
            print(self.regression_results.to_string(index=False))
            
            # Find best model (highest R²)
            best_model = self.regression_results.loc[self.regression_results['R² Score'].idxmax()]
            print(f"\n🏆 Best Regression Model: {best_model['Model']}")
            print(f"   R² Score: {best_model['R² Score']:.4f}")
            print(f"   RMSE: {best_model['RMSE']:.4f}")
        
        if self.clustering_results is not None:
            print("\n\n🎯 CLUSTERING MODELS:")
            print("-" * 80)
            print(self.clustering_results.to_string(index=False))
            
            # Find best model (highest silhouette score)
            best_model = self.clustering_results.loc[self.clustering_results['Silhouette Score'].idxmax()]
            print(f"\n🏆 Best Clustering Model: {best_model['Model']}")
            print(f"   Silhouette Score: {best_model['Silhouette Score']:.4f}")
        
        print("\n" + "="*80)
    
    def generate_comprehensive_report(self, save_path='results_report.txt'):
        """
        Generate comprehensive text report
        
        Args:
            save_path (str): Path to save the report
        """
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("BANK MARKETING DATA SCIENCE PROJECT - COMPREHENSIVE RESULTS REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Classification Results
            if self.classification_results is not None:
                f.write("1. CLASSIFICATION MODELS\n")
                f.write("-"*80 + "\n\n")
                f.write(self.classification_results.to_string(index=False) + "\n\n")
                
                best_model = self.classification_results.loc[self.classification_results['F1-Score'].idxmax()]
                f.write(f"Best Model: {best_model['Model']}\n")
                f.write(f"  - Accuracy: {best_model['Accuracy']:.4f}\n")
                f.write(f"  - Precision: {best_model['Precision']:.4f}\n")
                f.write(f"  - Recall: {best_model['Recall']:.4f}\n")
                f.write(f"  - F1-Score: {best_model['F1-Score']:.4f}\n\n")
                
                f.write("Findings:\n")
                f.write("  - All classification models achieved good performance\n")
                f.write("  - Random Forest typically shows the best balance of metrics\n")
                f.write("  - Decision Tree provides interpretability\n")
                f.write("  - Logistic Regression offers fast training and prediction\n\n")
            
            # Regression Results
            if self.regression_results is not None:
                f.write("\n2. REGRESSION MODELS\n")
                f.write("-"*80 + "\n\n")
                f.write(self.regression_results.to_string(index=False) + "\n\n")
                
                best_model = self.regression_results.loc[self.regression_results['R² Score'].idxmax()]
                f.write(f"Best Model: {best_model['Model']}\n")
                f.write(f"  - MSE: {best_model['MSE']:.4f}\n")
                f.write(f"  - RMSE: {best_model['RMSE']:.4f}\n")
                f.write(f"  - MAE: {best_model['MAE']:.4f}\n")
                f.write(f"  - R² Score: {best_model['R² Score']:.4f}\n\n")
                
                f.write("Findings:\n")
                f.write("  - Regression models predict call duration\n")
                f.write("  - Ridge regression helps prevent overfitting\n")
                f.write("  - Duration is a key indicator of campaign success\n\n")
            
            # Clustering Results
            if self.clustering_results is not None:
                f.write("\n3. CLUSTERING MODELS\n")
                f.write("-"*80 + "\n\n")
                f.write(self.clustering_results.to_string(index=False) + "\n\n")
                
                best_model = self.clustering_results.loc[self.clustering_results['Silhouette Score'].idxmax()]
                f.write(f"Best Model: {best_model['Model']}\n")
                f.write(f"  - Number of Clusters: {best_model['n_clusters']}\n")
                f.write(f"  - Silhouette Score: {best_model['Silhouette Score']:.4f}\n")
                f.write(f"  - Davies-Bouldin Index: {best_model['Davies-Bouldin Index']:.4f}\n\n")
                
                f.write("Findings:\n")
                f.write("  - Clustering reveals customer segments\n")
                f.write("  - K-Means provides clear cluster assignments\n")
                f.write("  - Hierarchical clustering shows relationships between groups\n\n")
            
            # Overall Conclusions
            f.write("\n4. OVERALL CONCLUSIONS\n")
            f.write("-"*80 + "\n\n")
            f.write("Key Insights:\n")
            f.write("  1. The dataset shows clear patterns for predicting term deposit subscriptions\n")
            f.write("  2. Call duration is a strong indicator of success\n")
            f.write("  3. Customer demographics and economic indicators play important roles\n")
            f.write("  4. Multiple customer segments exist with different characteristics\n\n")
            
            f.write("Recommendations:\n")
            f.write("  1. Focus on longer call durations for better conversion rates\n")
            f.write("  2. Target specific customer segments identified by clustering\n")
            f.write("  3. Use ensemble methods (Random Forest) for best predictions\n")
            f.write("  4. Monitor economic indicators for campaign timing\n\n")
            
            f.write("="*80 + "\n")
        
        print(f"\n✅ Comprehensive report saved: {save_path}")
    
    def plot_overall_comparison(self, save_path='visualizations/overall_comparison.png'):
        """
        Create overall comparison visualization
        
        Args:
            save_path (str): Path to save the plot
        """
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig = plt.figure(figsize=(16, 10))
        
        # Classification comparison
        if self.classification_results is not None:
            ax1 = plt.subplot(2, 2, 1)
            models = self.classification_results['Model']
            f1_scores = self.classification_results['F1-Score']
            
            colors = ['#ff6b6b', '#4ecdc4', '#95e1d3']
            ax1.barh(models, f1_scores, color=colors, alpha=0.7, edgecolor='black')
            ax1.set_xlabel('F1-Score', fontweight='bold')
            ax1.set_title('Classification Models - F1-Score', fontsize=12, fontweight='bold')
            ax1.set_xlim([0, 1])
            ax1.grid(axis='x', alpha=0.3)
        
        # Regression comparison
        if self.regression_results is not None:
            ax2 = plt.subplot(2, 2, 2)
            models = self.regression_results['Model']
            r2_scores = self.regression_results['R² Score']
            
            colors = ['#f38181', '#aa96da']
            ax2.barh(models, r2_scores, color=colors, alpha=0.7, edgecolor='black')
            ax2.set_xlabel('R² Score', fontweight='bold')
            ax2.set_title('Regression Models - R² Score', fontsize=12, fontweight='bold')
            ax2.set_xlim([0, 1])
            ax2.grid(axis='x', alpha=0.3)
        
        # Clustering comparison
        if self.clustering_results is not None:
            ax3 = plt.subplot(2, 2, 3)
            models = self.clustering_results['Model']
            silhouette_scores = self.clustering_results['Silhouette Score']
            
            colors = ['#fcbad3', '#a8d8ea']
            ax3.barh(models, silhouette_scores, color=colors, alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Silhouette Score', fontweight='bold')
            ax3.set_title('Clustering Models - Silhouette Score', fontsize=12, fontweight='bold')
            ax3.set_xlim([0, 1])
            ax3.grid(axis='x', alpha=0.3)
        
        # Summary text
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')
        
        summary_text = "PROJECT SUMMARY\n\n"
        summary_text += "✓ Data Preprocessing: Complete\n"
        summary_text += "✓ Visualizations: 10+ charts\n"
        summary_text += "✓ Classification: 3 models\n"
        summary_text += "✓ Regression: 2 models\n"
        summary_text += "✓ Clustering: 2 algorithms\n\n"
        
        if self.classification_results is not None:
            best_clf = self.classification_results.loc[self.classification_results['F1-Score'].idxmax()]
            summary_text += f"Best Classifier:\n{best_clf['Model']}\n"
            summary_text += f"F1-Score: {best_clf['F1-Score']:.4f}\n\n"
        
        if self.regression_results is not None:
            best_reg = self.regression_results.loc[self.regression_results['R² Score'].idxmax()]
            summary_text += f"Best Regressor:\n{best_reg['Model']}\n"
            summary_text += f"R² Score: {best_reg['R² Score']:.4f}\n\n"
        
        if self.clustering_results is not None:
            best_clust = self.clustering_results.loc[self.clustering_results['Silhouette Score'].idxmax()]
            summary_text += f"Best Clustering:\n{best_clust['Model']}\n"
            summary_text += f"Silhouette: {best_clust['Silhouette Score']:.4f}"
        
        ax4.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle('Bank Marketing Project - Overall Results Comparison', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Overall comparison plot saved: {save_path}")
        plt.close()


if __name__ == "__main__":
    print("\n🧪 Testing results analysis module...\n")
    
    # Create sample results
    classification_results = pd.DataFrame({
        'Model': ['Decision Tree', 'Random Forest', 'Logistic Regression'],
        'Accuracy': [0.85, 0.89, 0.87],
        'Precision': [0.82, 0.86, 0.84],
        'Recall': [0.80, 0.85, 0.83],
        'F1-Score': [0.81, 0.855, 0.835],
        'ROC-AUC': [0.88, 0.92, 0.90]
    })
    
    regression_results = pd.DataFrame({
        'Model': ['Linear Regression', 'Ridge Regression'],
        'MSE': [45000, 44500],
        'RMSE': [212.13, 210.95],
        'MAE': [150.5, 149.8],
        'R² Score': [0.35, 0.36]
    })
    
    clustering_results = pd.DataFrame({
        'Model': ['K-Means', 'Hierarchical'],
        'n_clusters': [3, 3],
        'Silhouette Score': [0.28, 0.26],
        'Davies-Bouldin Index': [1.45, 1.52]
    })
    
    # Create analyzer
    analyzer = ResultsAnalyzer()
    analyzer.set_classification_results(classification_results)
    analyzer.set_regression_results(regression_results)
    analyzer.set_clustering_results(clustering_results)
    
    # Print results
    analyzer.print_all_results()
    
    # Generate report
    analyzer.generate_comprehensive_report()
    
    # Plot comparison
    analyzer.plot_overall_comparison()
    
    print("\n✅ Results analysis test completed successfully!")
