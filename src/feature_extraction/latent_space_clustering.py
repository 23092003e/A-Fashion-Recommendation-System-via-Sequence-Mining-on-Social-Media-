import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
def load_latent_space(input_path):
    df = pd.read_hdf(input_path, key='df_items')
    df['latent_space'] = df['latent_space'].apply(lambda x: np.array(x.split(','), dtype=float))
    return df

# 2. Convert to Matrix
def get_latent_matrix(df):
    return np.stack(df['latent_space'].values)

# 3. Elbow Method for Optimal K
def find_optimal_clusters(latent_matrix, max_clusters=10):
    distortions = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(latent_matrix)
        distortions.append(kmeans.inertia_)
    
    # Plot Elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), distortions, marker='o', color='b')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal K')
    plt.grid(True)
    plt.show()

# 4. Apply KMeans Clustering
def apply_kmeans(latent_matrix, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(latent_matrix)
    return clusters

# 5. Visualize Clusters (PCA 2D)
def visualize_clusters(latent_matrix, clusters):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(latent_matrix)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=clusters, palette='Set2', s=70)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('2D Visualization of Clusters')
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.show()

# 6. Save Results
def save_results(df, clusters, output_path):
    df['cluster'] = clusters
    df[['path', 'cluster']].to_csv(output_path, index=False)
    print(f"✅ Clustering results saved to: {output_path}")

# 7. Main Function
def main():
    INPUT_PATH = "C:/Users/ADMIN/Desktop/ITDSIU21099_HoangVanManh/Fashion-Marketing-Automation-Solutions/data/processed/latent_spaces.h5"
    OUTPUT_PATH = "C:/Users/ADMIN/Desktop/ITDSIU21099_HoangVanManh/Fashion-Marketing-Automation-Solutions/data/processed/clustering_results.csv"
    
    print("🔄 Loading latent spaces...")
    df = load_latent_space(INPUT_PATH)
    latent_matrix = get_latent_matrix(df)
    
    print("📊 Finding optimal clusters...")
    find_optimal_clusters(latent_matrix)

    n_clusters = int(input("💡 Enter the optimal number of clusters: "))
    print(f"🚀 Applying KMeans with {n_clusters} clusters...")
    clusters = apply_kmeans(latent_matrix, n_clusters)

    print("🎨 Visualizing clusters...")
    visualize_clusters(latent_matrix, clusters)

    print("💾 Saving results...")
    save_results(df, clusters, OUTPUT_PATH)

    print("🎉 Clustering pipeline completed successfully!")

if __name__ == "__main__":
    main()
