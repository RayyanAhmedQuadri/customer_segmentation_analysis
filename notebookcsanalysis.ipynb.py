# Customer Segmentation Analysis
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from src.data_processing import load_and_clean_data
from src.visualization import plot_elbow_method, plot_clusters

# Load and clean data
df = load_and_clean_data('../data/customers.csv')

# Exploratory Data Analysis
print(df.head())
print(df.describe())

# Visualize distributions
plt.figure(figsize=(15, 10))
for i, col in enumerate(df.select_dtypes(include=['float64', 'int64']).columns):
    plt.subplot(3, 3, i+1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Correlation analysis
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Feature scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Determine optimal number of clusters using Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)
    
plot_elbow_method(range(1, 11), wcss)

# Silhouette Analysis
range_n_clusters = range(2, 6)
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    preds = clusterer.fit_predict(scaled_data)
    silhouette_avg = silhouette_score(scaled_data, preds)
    print(f"For n_clusters = {n_clusters}, the silhouette score is {silhouette_avg}")

# Final clustering with K=4
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Add clusters to original data
df['Cluster'] = clusters

# Visualize clusters
plot_clusters(df, 'Age', 'Annual Income (k$)', 'Cluster')
plot_clusters(df, 'Age', 'Spending Score (1-100)', 'Cluster')
plot_clusters(df, 'Annual Income (k$)', 'Spending Score (1-100)', 'Cluster')

# Cluster analysis
cluster_analysis = df.groupby('Cluster').mean()
print(cluster_analysis)