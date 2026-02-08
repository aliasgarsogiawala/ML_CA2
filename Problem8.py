# -------------------------------------------------
# PROBLEM 8: UNSUPERVISED LEARNING
# Dimensionality Reduction using PCA
# FINAL DEFINITIVE VERSION
# -------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sns.set(style="whitegrid")

# -------------------------------------------------
# Step 1: Load Dataset
# -------------------------------------------------

data = pd.read_csv("cleaned_preflight_mental_state_data.csv")

# -------------------------------------------------
# Step 2: Select Candidate Features
# -------------------------------------------------

features = [
    'Decision_Confidence',
    'Mental_Workload',
    'Sleep_Quality',
    'Emotional_Distraction',
    'In_Flight_Focus',
    'Communication_Load',
    'Peer_Support_Importance'
]

X = data[features]

# -------------------------------------------------
# Step 3: DROP COLUMNS WITH ALL NaN VALUES (CRITICAL)
# -------------------------------------------------

X = X.dropna(axis=1, how='all')

print("\nFeatures used for PCA:")
print(list(X.columns))

# -------------------------------------------------
# Step 4: Handle Remaining Missing Values
# -------------------------------------------------

imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# -------------------------------------------------
# Step 5: Standardize the Data
# -------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# -------------------------------------------------
# Step 6: Apply PCA (All Components)
# -------------------------------------------------

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = explained_variance.cumsum()

# -------------------------------------------------
# Step 7: Explained Variance Plot
# -------------------------------------------------

plt.figure(figsize=(8,5))
plt.bar(
    range(1, len(explained_variance) + 1),
    explained_variance,
    alpha=0.7,
    label="Individual Variance"
)
plt.step(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    where='mid',
    label="Cumulative Variance"
)
plt.xlabel("Principal Components")
plt.ylabel("Explained Variance Ratio")
plt.title("Explained Variance by Principal Components")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------------------------
# Step 8: PCA with 2 Components (Visualization)
# -------------------------------------------------

pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    X_pca_2,
    columns=["PC1", "PC2"]
)

# -------------------------------------------------
# Step 9: 2D PCA Scatter Plot
# -------------------------------------------------

plt.figure(figsize=(7,5))
sns.scatterplot(
    x="PC1",
    y="PC2",
    data=pca_df,
    alpha=0.7
)
plt.title("2D PCA Projection of Pre-Flight Mental States")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.show()

# -------------------------------------------------
# Step 10: PCA Component Loadings (FIXED)
# -------------------------------------------------

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    index=X.columns
)

print("\nPCA Component Loadings:")
print(loadings)

# -------------------------------------------------
# Step 11: Explained Variance Summary Table
# -------------------------------------------------

variance_df = pd.DataFrame({
    "Principal Component": [f"PC{i+1}" for i in range(len(explained_variance))],
    "Explained Variance Ratio": explained_variance,
    "Cumulative Variance": cumulative_variance
})

print("\nExplained Variance Summary:")
print(variance_df)
