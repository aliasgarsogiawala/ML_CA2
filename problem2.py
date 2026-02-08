# -------------------------------------------------
# PROBLEM 2: CLASSIFICATION
# Classification of In-Flight Decision Confidence
# -------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

sns.set(style="whitegrid")

# -------------------------------------------------
# Step 1: Load Dataset
# -------------------------------------------------

data = pd.read_csv("cleaned_preflight_mental_state_data.csv")

# -------------------------------------------------
# Step 2: Feature Selection
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
# Step 3: Create Classification Target (Low / Medium / High)
# -------------------------------------------------

def classify_confidence(val):
    if val <= 0.33:
        return "Low"
    elif val <= 0.66:
        return "Medium"
    else:
        return "High"

data['Confidence_Level'] = data['In_Flight_Confidence'].apply(classify_confidence)
y = data['Confidence_Level']

# -------------------------------------------------
# Step 4: Train-Test Split
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# -------------------------------------------------
# Step 5: Define Classifiers (NaN-SAFE)
# -------------------------------------------------

models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", DecisionTreeClassifier(random_state=42))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ]),

    "Support Vector Machine": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", SVC())
    ])
}

# -------------------------------------------------
# Step 6: Train & Evaluate Models
# -------------------------------------------------

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted")
    rec = recall_score(y_test, preds, average="weighted")
    f1 = f1_score(y_test, preds, average="weighted")

    results.append([name, acc, prec, rec, f1])

    print(f"\n{name}")
    print(classification_report(y_test, preds))

results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
)

print("\nMODEL COMPARISON")
print(results_df)

# -------------------------------------------------
# Step 7: Model Comparison Diagram
# -------------------------------------------------

plt.figure(figsize=(9,5))
sns.barplot(data=results_df, x="Model", y="Accuracy")
plt.title("Model Comparison Based on Accuracy")
plt.ylabel("Accuracy")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# -------------------------------------------------
# Step 8: Target Distribution Diagram
# -------------------------------------------------

plt.figure(figsize=(7,5))
y.value_counts().plot(kind='bar', edgecolor='black')
plt.title("Distribution of In-Flight Confidence Levels")
plt.xlabel("Confidence Level")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# -------------------------------------------------
# Step 9: Confusion Matrix (Best Model)
# -------------------------------------------------

best_model_name = results_df.sort_values(
    by="Accuracy", ascending=False
).iloc[0]["Model"]

best_model = models[best_model_name]
best_preds = best_model.predict(X_test)

cm = confusion_matrix(y_test, best_preds, labels=["Low", "Medium", "High"])

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Low", "Medium", "High"],
    yticklabels=["Low", "Medium", "High"]
)
plt.title(f"Confusion Matrix – {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# -------------------------------------------------
# Step 10: Feature Importance (Tree-Based Only)
# -------------------------------------------------

if best_model_name in ["Decision Tree", "Random Forest"]:
    model_step = best_model.named_steps["model"]
    importance = pd.Series(
        model_step.feature_importances_,
        index=features
    ).sort_values(ascending=False)

    plt.figure(figsize=(8,5))
    sns.barplot(x=importance, y=importance.index)
    plt.title(f"Feature Importance – {best_model_name}")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()
