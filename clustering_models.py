"""
Clustering Models Module for Bank Marketing Dataset
Implements K-Means and Hierarchical Clustering
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class ClusteringModels:
    """
    Comprehensive clustering models class
    """
    
    def __init__(self, X_data):
        """
        Initialize with data
        
        Args:
            X_data: Features for clustering
        """
        self.X_data = X_data
        self.models = {}
        self.labels = {}
        self.results = {}
    
    def find_optimal_k(self, k_range=range(2, 11)):
        """
        Find optimal number of clusters using elbow method and silhouette score
        
        Args:
            k_range: Range of k values to test
            
        Returns:
            dict: Inertia and silhouette scores for each k
        """
        print("\n🔍 Finding optimal number of clusters...")
        
        inertias = []
        silhouette_scores = []
        k_values = list(k_range)
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.X_data)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.X_data, labels))
            
            print(f"   k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_score(self.X_data, labels):.4f}")
        
        # Find optimal k (highest silhouette score)
        optimal_k = k_values[np.argmax(silhouette_scores)]
        print(f"\n✅ Optimal k based on silhouette score: {optimal_k}")
        
        return {
            'k_values': k_values,
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'optimal_k': optimal_k
        }
    
    def train_kmeans(self, n_clusters=3, random_state=42):
        """
        Train K-Means clustering
        
        Args:
            n_clusters (int): Number of clusters
            random_state (int): Random seed
        """
        print(f"\n🎯 Training K-Means Clustering (k={n_clusters})...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(self.X_data)
        
        self.models['K-Means'] = kmeans
        self.labels['K-Means'] = labels
        
        # Calculate metrics
        inertia = kmeans.inertia_
        silhouette = silhouette_score(self.X_data, labels)
        davies_bouldin = davies_bouldin_score(self.X_data, labels)
        
        self.results['K-Means'] = {
            'Model': 'K-Means',
            'n_clusters': n_clusters,
            'Inertia': inertia,
            'Silhouette Score': silhouette,
            'Davies-Bouldin Index': davies_bouldin
        }
        
        print(f"✅ K-Means trained successfully!")
        print(f"   Inertia: {inertia:.2f}")
        print(f"   Silhouette Score: {silhouette:.4f}")
        print(f"   Davies-Bouldin Index: {davies_bouldin:.4f}")
        
        return kmeans, labels
    
    def train_hierarchical(self, n_clusters=3, linkage_method='ward'):
        """
        Train Hierarchical Clustering
        
        Args:
            n_clusters (int): Number of clusters
            linkage_method (str): Linkage method ('ward', 'complete', 'average')
        """
        print(f"\n🌳 Training Hierarchical Clustering (k={n_clusters}, linkage={linkage_method})...")
        
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        labels = hierarchical.fit_predict(self.X_data)
        
        self.models['Hierarchical'] = hierarchical
        self.labels['Hierarchical'] = labels
        
        # Calculate metrics
        silhouette = silhouette_score(self.X_data, labels)
        davies_bouldin = davies_bouldin_score(self.X_data, labels)
        
        self.results['Hierarchical'] = {
            'Model': 'Hierarchical',
            'n_clusters': n_clusters,
            'Silhouette Score': silhouette,
            'Davies-Bouldin Index': davies_bouldin
        }
        
        print(f"✅ Hierarchical Clustering trained successfully!")
        print(f"   Silhouette Score: {silhouette:.4f}")
        print(f"   Davies-Bouldin Index: {davies_bouldin:.4f}")
        
        return hierarchical, labels
    
    def train_all(self, n_clusters=3):
        """
        Train all clustering models
        
        Args:
            n_clusters (int): Number of clusters
        """
        print("\n" + "="*70)
        print("🚀 TRAINING CLUSTERING MODELS")
        print("="*70)
        
        # Find optimal k
        optimal_k_info = self.find_optimal_k()
        optimal_k = optimal_k_info['optimal_k']
        
        # Train K-Means with optimal k
        self.train_kmeans(n_clusters=optimal_k)
        
        # Train Hierarchical with optimal k
        self.train_hierarchical(n_clusters=optimal_k)
        
        print("\n" + "="*70)
        print("✅ ALL CLUSTERING MODELS TRAINED!")
        print("="*70)
        
        return optimal_k_info
    
    def get_results_dataframe(self):
        """
        Get results as a pandas DataFrame
        
        Returns:
            pd.DataFrame: Results dataframe
        """
        results_list = []
        
        for model_name, results in self.results.items():
            results_list.append(results)
        
        return pd.DataFrame(results_list)
    
    def plot_elbow_silhouette(self, optimal_k_info, save_path='visualizations/clustering_elbow_silhouette.png'):
        """
        Plot elbow and silhouette analysis
        
        Args:
            optimal_k_info (dict): Dictionary with k values, inertias, and silhouette scores
            save_path (str): Path to save the plot
        """
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        k_values = optimal_k_info['k_values']
        inertias = optimal_k_info['inertias']
        silhouette_scores = optimal_k_info['silhouette_scores']
        optimal_k = optimal_k_info['optimal_k']
        
        # Elbow plot
        axes[0].plot(k_values, inertias, marker='o', linewidth=2, markersize=8, color='#4ecdc4')
        axes[0].set_xlabel('Number of Clusters (k)', fontweight='bold')
        axes[0].set_ylabel('Inertia', fontweight='bold')
        axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
        axes[0].grid(alpha=0.3)
        
        # Silhouette plot
        axes[1].plot(k_values, silhouette_scores, marker='o', linewidth=2, markersize=8, color='#ff6b6b')
        axes[1].axvline(x=optimal_k, color='green', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k}')
        axes[1].set_xlabel('Number of Clusters (k)', fontweight='bold')
        axes[1].set_ylabel('Silhouette Score', fontweight='bold')
        axes[1].set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Elbow and silhouette plot saved: {save_path}")
        plt.close()
    
    def plot_dendrogram(self, max_samples=1000, save_path='visualizations/dendrogram.png'):
        """
        Plot dendrogram for hierarchical clustering
        
        Args:
            max_samples (int): Maximum number of samples to use (for performance)
            save_path (str): Path to save the plot
        """
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print(f"\n📊 Creating dendrogram (using {max_samples} samples)...")
        
        # Sample data if too large
        if len(self.X_data) > max_samples:
            indices = np.random.choice(len(self.X_data), max_samples, replace=False)
            X_sample = self.X_data[indices]
        else:
            X_sample = self.X_data
        
        # Calculate linkage
        linkage_matrix = linkage(X_sample, method='ward')
        
        # Plot
        plt.figure(figsize=(14, 7))
        dendrogram(linkage_matrix, no_labels=True)
        plt.xlabel('Sample Index', fontweight='bold')
        plt.ylabel('Distance', fontweight='bold')
        plt.title('Hierarchical Clustering Dendrogram', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Dendrogram saved: {save_path}")
        plt.close()
    
    def plot_cluster_distribution(self, save_dir='visualizations'):
        """
        Plot cluster distribution for all models
        
        Args:
            save_dir (str): Directory to save plots
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, labels in self.labels.items():
            plt.figure(figsize=(10, 6))
            
            unique, counts = np.unique(labels, return_counts=True)
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique)))
            plt.bar(unique, counts, color=colors, edgecolor='black', alpha=0.7)
            
            plt.xlabel('Cluster', fontweight='bold')
            plt.ylabel('Number of Samples', fontweight='bold')
            plt.title(f'Cluster Distribution - {model_name}', fontsize=14, fontweight='bold')
            plt.xticks(unique)
            plt.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(counts):
                plt.text(unique[i], v + max(counts)*0.01, str(v), ha='center', fontweight='bold')
            
            plt.tight_layout()
            filename = f"{save_dir}/cluster_distribution_{model_name.lower().replace(' ', '_')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {filename}")
            plt.close()
    
    def plot_cluster_comparison(self, save_path='visualizations/clustering_comparison.png'):
        """
        Plot clustering models comparison
        
        Args:
            save_path (str): Path to save the plot
        """
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        df_results = self.get_results_dataframe()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        models = df_results['Model'].tolist()
        
        # Silhouette Score comparison
        axes[0].bar(models, df_results['Silhouette Score'], color='#4ecdc4', alpha=0.7, edgecolor='black')
        axes[0].set_ylabel('Silhouette Score', fontweight='bold')
        axes[0].set_title('Silhouette Score Comparison', fontweight='bold')
        axes[0].set_ylim([0, 1])
        axes[0].grid(axis='y', alpha=0.3)
        
        # Davies-Bouldin Index comparison (lower is better)
        axes[1].bar(models, df_results['Davies-Bouldin Index'], color='#ff6b6b', alpha=0.7, edgecolor='black')
        axes[1].set_ylabel('Davies-Bouldin Index', fontweight='bold')
        axes[1].set_title('Davies-Bouldin Index Comparison (Lower is Better)', fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Clustering comparison plot saved: {save_path}")
        plt.close()


if __name__ == "__main__":
    # Test clustering models
    print("\n🧪 Testing clustering models...\n")
    
    from preprocessing import load_and_preprocess_data
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data()
    
    # Use combined data for clustering
    X_combined = np.vstack([X_train, X_test])
    
    # Create clusterer
    clusterer = ClusteringModels(X_combined)
    
    # Train all models
    optimal_k_info = clusterer.train_all()
    
    # Get results dataframe
    results_df = clusterer.get_results_dataframe()
    print("\n📊 Results Summary:")
    print(results_df.to_string(index=False))
    
    # Plot visualizations
    clusterer.plot_elbow_silhouette(optimal_k_info)
    clusterer.plot_dendrogram()
    clusterer.plot_cluster_distribution()
    clusterer.plot_cluster_comparison()
    
    print("\n✅ Clustering models test completed successfully!")
