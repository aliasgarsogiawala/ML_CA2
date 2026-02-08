# -------------------------------------------------
# PROBLEM 9: UNSUPERVISED LEARNING
# Anomaly Detection of Pre-Flight Mental States
# -------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

sns.set(style="whitegrid")

# -------------------------------------------------
# Step 1: Load Dataset
# -------------------------------------------------

data = pd.read_csv("cleaned_preflight_mental_state_data.csv")

# -------------------------------------------------
# Step 2: Select Features
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
# Step 3: Handle Missing Values
# -------------------------------------------------

imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# -------------------------------------------------
# Step 4: Standardize Data
# -------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# -------------------------------------------------
# Step 5: Isolation Forest
# -------------------------------------------------

iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

data['IsolationForest_Anomaly'] = iso_forest.fit_predict(X_scaled)
# -1 = anomaly, 1 = normal

print("\nIsolation Forest Anomaly Counts:")
print(data['IsolationForest_Anomaly'].value_counts())

# -------------------------------------------------
# Step 6: Local Outlier Factor (LOF)
# -------------------------------------------------

lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.05
)

data['LOF_Anomaly'] = lof.fit_predict(X_scaled)
# -1 = anomaly, 1 = normal

print("\nLocal Outlier Factor Anomaly Counts:")
print(data['LOF_Anomaly'].value_counts())

# -------------------------------------------------
# Step 7: Anomaly Visualization (2D)
# -------------------------------------------------

plt.figure(figsize=(7,5))
sns.scatterplot(
    x=data['Mental_Workload'],
    y=data['In_Flight_Focus'],
    hue=data['IsolationForest_Anomaly'],
    palette={1: 'blue', -1: 'red'}
)
plt.title("Isolation Forest – Anomaly Detection")
plt.xlabel("Mental Workload")
plt.ylabel("In-Flight Focus")
plt.legend(title="Anomaly")
plt.tight_layout()
plt.show()

# -------------------------------------------------
# Step 8: LOF Visualization (2D)
# -------------------------------------------------

plt.figure(figsize=(7,5))
sns.scatterplot(
    x=data['Mental_Workload'],
    y=data['In_Flight_Focus'],
    hue=data['LOF_Anomaly'],
    palette={1: 'blue', -1: 'red'}
)
plt.title("Local Outlier Factor – Anomaly Detection")
plt.xlabel("Mental Workload")
plt.ylabel("In-Flight Focus")
plt.legend(title="Anomaly")
plt.tight_layout()
plt.show()

# -------------------------------------------------
# Step 9: Compare Detected Anomalies
# -------------------------------------------------

comparison = pd.crosstab(
    data['IsolationForest_Anomaly'],
    data['LOF_Anomaly'],
    rownames=['IsolationForest'],
    colnames=['LOF']
)

print("\nAnomaly Detection Comparison (Isolation Forest vs LOF):")
print(comparison)
