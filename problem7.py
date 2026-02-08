# -------------------------------------------------
# PROBLEM 7: UNSUPERVISED LEARNING
# Clustering of Pre-Flight Mental States
# -------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

sns.set(style="whitegrid")

# -------------------------------------------------
# Step 1: Load Dataset
# -------------------------------------------------

data = pd.read_csv("cleaned_preflight_mental_state_data.csv")

# -------------------------------------------------
# Step 2: Select Features for Clustering
# -------------------------------------------------

features = [
    'Mental_Workload',
    'Sleep_Quality',
    'Emotional_Distraction',
    'In_Flight_Focus',
    'Communication_Load',
    'Peer_Support_Importance'
]

X = data[features]

# -------------------------------------------------
# Step 3: Standardize Features
# -------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------------------
# Step 4: Elbow Method (K-Means)
# -------------------------------------------------

wcss = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------------------------
# Step 5: K-Means Clustering (k = 3)
# -------------------------------------------------

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
data['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

sil_score = silhouette_score(X_scaled, data['KMeans_Cluster'])
print(f"Silhouette Score (K-Means): {sil_score:.3f}")

# -------------------------------------------------
# Step 6: Cluster Visualization (K-Means)
# -------------------------------------------------

plt.figure(figsize=(7,5))
sns.scatterplot(
    x=data['Mental_Workload'],
    y=data['In_Flight_Focus'],
    hue=data['KMeans_Cluster'],
    palette='Set2'
)
plt.title("K-Means Clusters (Mental Workload vs Focus)")
plt.xlabel("Mental Workload")
plt.ylabel("In-Flight Focus")
plt.tight_layout()
plt.show()

# -------------------------------------------------
# Step 7: Hierarchical Clustering
# -------------------------------------------------

hierarchical = AgglomerativeClustering(n_clusters=3)
data['Hierarchical_Cluster'] = hierarchical.fit_predict(X_scaled)

# -------------------------------------------------
# Step 8: Dendrogram
# -------------------------------------------------

linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(10,5))
dendrogram(
    linked,
    truncate_mode='lastp',
    p=10,
    leaf_rotation=45,
    leaf_font_size=10
)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Cluster Size")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

# -------------------------------------------------
# Step 9: Cluster Profile Summary
# -------------------------------------------------

print("\nCluster-wise Feature Means (K-Means):")
print(data.groupby('KMeans_Cluster')[features].mean())

# -------------------------------------------------
# Step 10: Cluster Distribution
# -------------------------------------------------

plt.figure(figsize=(6,5))
data['KMeans_Cluster'].value_counts().sort_index().plot(
    kind='bar', edgecolor='black'
)
plt.title("Distribution of K-Means Clusters")
plt.xlabel("Cluster Label")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
