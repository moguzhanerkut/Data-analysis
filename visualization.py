"""
Data Visualization Module for Bank Marketing Dataset
Creates comprehensive visualizations for exploratory data analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class DataVisualizer:
    """
    Comprehensive visualization class for Bank Marketing dataset
    """
    
    def __init__(self, df, save_dir='visualizations'):
        """
        Initialize visualizer
        
        Args:
            df (pd.DataFrame): Dataset to visualize
            save_dir (str): Directory to save visualizations
        """
        self.df = df
        self.save_dir = save_dir
        
        # Create save directory if it doesn't exist
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def plot_age_distribution(self):
        """Plot age distribution histogram"""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(self.df['age'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.title('Age Distribution')
        plt.grid(axis='y', alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.boxplot(self.df['age'], vert=True)
        plt.ylabel('Age')
        plt.title('Age Box Plot')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/01_age_distribution.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: 01_age_distribution.png")
        plt.close()
    
    def plot_target_distribution(self):
        """Plot target variable distribution"""
        plt.figure(figsize=(12, 6))
        
        # Count plot
        plt.subplot(1, 2, 1)
        target_counts = self.df['y'].value_counts()
        colors = ['#ff6b6b', '#4ecdc4']
        plt.bar(target_counts.index, target_counts.values, color=colors, edgecolor='black', alpha=0.7)
        plt.xlabel('Target (y)')
        plt.ylabel('Count')
        plt.title('Target Variable Distribution')
        plt.xticks(rotation=0)
        
        # Add value labels on bars
        for i, v in enumerate(target_counts.values):
            plt.text(i, v + 500, str(v), ha='center', fontweight='bold')
        
        # Pie chart
        plt.subplot(1, 2, 2)
        plt.pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90, explode=(0.05, 0))
        plt.title('Target Variable Proportion')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/02_target_distribution.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: 02_target_distribution.png")
        plt.close()
    
    def plot_correlation_heatmap(self):
        """Plot correlation heatmap for numerical features"""
        plt.figure(figsize=(14, 10))
        
        # Select numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate correlation matrix
        corr_matrix = self.df[numerical_cols].corr()
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Heatmap - Numerical Features', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(f'{self.save_dir}/03_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: 03_correlation_heatmap.png")
        plt.close()
    
    def plot_job_vs_target(self):
        """Plot job distribution vs target variable"""
        plt.figure(figsize=(14, 6))
        
        # Create crosstab
        job_target = pd.crosstab(self.df['job'], self.df['y'], normalize='index') * 100
        
        # Plot
        job_target.plot(kind='bar', stacked=False, color=['#ff6b6b', '#4ecdc4'], 
                        edgecolor='black', alpha=0.7)
        plt.xlabel('Job Category')
        plt.ylabel('Percentage (%)')
        plt.title('Job Category vs Target Variable (Percentage)', fontsize=14, fontweight='bold')
        plt.legend(title='Target', labels=['No', 'Yes'])
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/04_job_vs_target.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: 04_job_vs_target.png")
        plt.close()
    
    def plot_campaign_analysis(self):
        """Plot campaign analysis"""
        plt.figure(figsize=(14, 6))
        
        # Campaign count distribution
        plt.subplot(1, 2, 1)
        campaign_success = self.df.groupby('campaign')['y'].apply(lambda x: (x == 'yes').sum() / len(x) * 100)
        campaign_counts = self.df['campaign'].value_counts().sort_index()
        
        # Limit to campaigns <= 10 for clarity
        campaign_success_limited = campaign_success[campaign_success.index <= 10]
        
        plt.bar(campaign_success_limited.index, campaign_success_limited.values, 
                color='#4ecdc4', edgecolor='black', alpha=0.7)
        plt.xlabel('Number of Campaigns')
        plt.ylabel('Success Rate (%)')
        plt.title('Campaign Count vs Success Rate')
        plt.grid(axis='y', alpha=0.3)
        
        # Duration vs target
        plt.subplot(1, 2, 2)
        duration_yes = self.df[self.df['y'] == 'yes']['duration']
        duration_no = self.df[self.df['y'] == 'no']['duration']
        
        plt.hist([duration_no, duration_yes], bins=30, label=['No', 'Yes'], 
                 color=['#ff6b6b', '#4ecdc4'], alpha=0.6, edgecolor='black')
        plt.xlabel('Call Duration (seconds)')
        plt.ylabel('Frequency')
        plt.title('Call Duration Distribution by Target')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/05_campaign_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: 05_campaign_analysis.png")
        plt.close()
    
    def plot_education_marital_analysis(self):
        """Plot education and marital status analysis"""
        plt.figure(figsize=(14, 6))
        
        # Education vs target
        plt.subplot(1, 2, 1)
        edu_target = pd.crosstab(self.df['education'], self.df['y'], normalize='index') * 100
        edu_target.plot(kind='bar', stacked=False, color=['#ff6b6b', '#4ecdc4'],
                        edgecolor='black', alpha=0.7, ax=plt.gca())
        plt.xlabel('Education Level')
        plt.ylabel('Percentage (%)')
        plt.title('Education Level vs Target Variable')
        plt.legend(title='Target', labels=['No', 'Yes'])
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Marital status vs target
        plt.subplot(1, 2, 2)
        marital_target = pd.crosstab(self.df['marital'], self.df['y'], normalize='index') * 100
        marital_target.plot(kind='bar', stacked=False, color=['#ff6b6b', '#4ecdc4'],
                           edgecolor='black', alpha=0.7, ax=plt.gca())
        plt.xlabel('Marital Status')
        plt.ylabel('Percentage (%)')
        plt.title('Marital Status vs Target Variable')
        plt.legend(title='Target', labels=['No', 'Yes'])
        plt.xticks(rotation=0)
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/06_education_marital_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: 06_education_marital_analysis.png")
        plt.close()
    
    def plot_feature_importance(self, feature_names, importances, model_name='Model'):
        """
        Plot feature importance
        
        Args:
            feature_names (list): List of feature names
            importances (array): Feature importance scores
            model_name (str): Name of the model
        """
        plt.figure(figsize=(12, 8))
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:20]  # Top 20 features
        
        plt.barh(range(len(indices)), importances[indices], color='#4ecdc4', 
                 edgecolor='black', alpha=0.7)
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance Score')
        plt.title(f'Top 20 Feature Importance - {model_name}', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/07_feature_importance_{model_name.lower().replace(" ", "_")}.png', 
                    dpi=300, bbox_inches='tight')
        print(f"✅ Saved: 07_feature_importance_{model_name.lower().replace(' ', '_')}.png")
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name='Model'):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name (str): Name of the model
        """
        plt.figure(figsize=(8, 6))
        
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'],
                    square=True, linewidths=1)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/confusion_matrix_{model_name.lower().replace(" ", "_")}.png', 
                    dpi=300, bbox_inches='tight')
        print(f"✅ Saved: confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
        plt.close()
    
    def generate_all_basic_visualizations(self):
        """Generate all basic visualizations"""
        print("\n" + "=" * 70)
        print("📊 GENERATING VISUALIZATIONS")
        print("=" * 70)
        
        self.plot_age_distribution()
        self.plot_target_distribution()
        self.plot_correlation_heatmap()
        self.plot_job_vs_target()
        self.plot_campaign_analysis()
        self.plot_education_marital_analysis()
        
        print("\n✅ All basic visualizations generated successfully!")
        print("=" * 70)


if __name__ == "__main__":
    # Test visualization module
    print("\n🧪 Testing visualization module...\n")
    
    # Load data
    df = pd.read_csv('bank-additional-full.csv', sep=';')
    
    # Create visualizer
    visualizer = DataVisualizer(df)
    
    # Generate all visualizations
    visualizer.generate_all_basic_visualizations()
    
    print("\n✅ Visualization test completed successfully!")
