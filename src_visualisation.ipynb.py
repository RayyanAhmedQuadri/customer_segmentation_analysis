import matplotlib.pyplot as plt
import seaborn as sns

def plot_elbow_method(k_range, wcss):
    """
    Plot elbow method results
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, wcss, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Method for Optimal K')
    plt.show()

def plot_clusters(df, x_col, y_col, cluster_col):
    """
    Visualize clusters in 2D space
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=cluster_col, palette='viridis')
    plt.title(f'Clusters by {x_col} and {y_col}')
    plt.show()