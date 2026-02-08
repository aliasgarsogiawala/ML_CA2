# -------------------------------------------------
# PROBLEM 6: CLASSIFICATION
# Overall Performance Perception During Flight
# FINAL FIXED VERSION (NO STRATIFY ISSUE)
# -------------------------------------------------

import pandas as pd
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
# Step 2: Feature & Target Selection
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
y = data['Performance_Comparison']

print("\nClass Distribution:")
print(y.value_counts())

# -------------------------------------------------
# Step 3: Train-Test Split (NO STRATIFICATION)
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

# -------------------------------------------------
# Step 4: Define Classifiers (NaN-SAFE)
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
# Step 5: Train & Evaluate Models
# -------------------------------------------------

results = []

class_labels = sorted(y.unique())
class_names = [str(label) for label in class_labels]

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

    results.append([name, acc, prec, rec, f1])

    print("\n====================================")
    print(f"Algorithm: {name}")
    print("====================================")
    print(classification_report(
        y_test,
        preds,
        labels=class_labels,
        target_names=class_names,
        zero_division=0
    ))

results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
)

print("\nMODEL PERFORMANCE COMPARISON")
print(results_df)

# -------------------------------------------------
# Step 6: Model Comparison Diagram
# -------------------------------------------------

plt.figure(figsize=(9,5))
sns.barplot(data=results_df, x="Model", y="Accuracy")
plt.title("Model Comparison Based on Accuracy")
plt.ylabel("Accuracy")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# -------------------------------------------------
# Step 7: Target Distribution Diagram
# -------------------------------------------------

plt.figure(figsize=(7,5))
y.value_counts().plot(kind='bar', edgecolor='black')
plt.title("Distribution of Performance Comparison Levels")
plt.xlabel("Performance Comparison")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# -------------------------------------------------
# Step 8: Confusion Matrix (Best Model)
# -------------------------------------------------

best_model_name = results_df.sort_values(
    by="Accuracy", ascending=False
).iloc[0]["Model"]

best_model = models[best_model_name]
best_preds = best_model.predict(X_test)

cm = confusion_matrix(y_test, best_preds, labels=class_labels)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title(f"Confusion Matrix – {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# -------------------------------------------------
# Step 9: Actual vs Predicted (Categorical)
# -------------------------------------------------

comparison_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": best_preds
})

plt.figure(figsize=(8,5))
sns.countplot(
    data=comparison_df,
    x="Actual",
    hue="Predicted"
)
plt.title("Actual vs Predicted Performance Comparison")
plt.xlabel("Actual Performance")
plt.ylabel("Count")
plt.legend(title="Predicted")
plt.tight_layout()
plt.show()

# -------------------------------------------------
# Step 10: Feature Importance (Tree-Based Models)
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
