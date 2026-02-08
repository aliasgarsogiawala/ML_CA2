# -------------------------------------------------
# PROBLEM 1: REGRESSION
# Predicting In-Flight Decision Confidence
# WITH SUPPORTING GRAPHS
# -------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

target = 'In_Flight_Confidence'

X = data[features]
y = data[target]

# -------------------------------------------------
# Step 3: TARGET DISTRIBUTION (REFERENCE STYLE)
# -------------------------------------------------

plt.figure(figsize=(8,5))
y.round().astype(int).value_counts().sort_index().plot(
    kind='bar', edgecolor='black'
)
plt.title("Distribution of In-Flight Decision Confidence")
plt.xlabel("Confidence Level")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# -------------------------------------------------
# Step 4: FEATURE vs TARGET RELATIONSHIPS
# -------------------------------------------------

for col in features:
    plt.figure(figsize=(7,4))
    sns.regplot(
        x=data[col],
        y=y,
        scatter_kws={'alpha': 0.5},
        line_kws={'color': 'red'}
    )
    plt.title(f"{col} vs In-Flight Confidence")
    plt.xlabel(col)
    plt.ylabel("In-Flight Confidence")
    plt.tight_layout()
    plt.show()

# -------------------------------------------------
# Step 5: Train-Test Split
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------------------------------------
# Step 6: Define Models (NaN-SAFE)
# -------------------------------------------------

models = {
    "Linear Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Ridge Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ))
    ]),

    "Gradient Boosting": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", GradientBoostingRegressor(
            n_estimators=100,
            random_state=42
        ))
    ])
}

# -------------------------------------------------
# Step 7: Train & Evaluate
# -------------------------------------------------

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results.append([name, mae, mse, r2])

    print(f"\n{name}")
    print("MAE:", mae)
    print("MSE:", mse)
    print("R²:", r2)

results_df = pd.DataFrame(
    results, columns=["Model", "MAE", "MSE", "R²"]
)

# -------------------------------------------------
# Step 8: MODEL COMPARISON GRAPH
# -------------------------------------------------

plt.figure(figsize=(8,5))
sns.barplot(
    data=results_df,
    x="Model",
    y="R²"
)
plt.title("Model Comparison Based on R² Score")
plt.ylabel("R² Score")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# -------------------------------------------------
# Step 9: BEST MODEL ANALYSIS
# -------------------------------------------------

best_model_name = results_df.sort_values(
    "R²", ascending=False
).iloc[0]["Model"]

best_model = models[best_model_name]
best_preds = best_model.predict(X_test)

# Actual vs Predicted
plt.figure(figsize=(7,6))
plt.scatter(y_test, best_preds, alpha=0.6)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)
plt.xlabel("Actual In-Flight Confidence")
plt.ylabel("Predicted In-Flight Confidence")
plt.title(f"Actual vs Predicted ({best_model_name})")
plt.tight_layout()
plt.show()

# Residual Plot
residuals = y_test - best_preds

plt.figure(figsize=(7,5))
plt.scatter(best_preds, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Analysis")
plt.tight_layout()
plt.show()

# -------------------------------------------------
# Step 10: FEATURE IMPORTANCE (ENSEMBLE ONLY)
# -------------------------------------------------

if "Random Forest" in best_model_name:
    rf = best_model.named_steps["model"]
    importance = pd.Series(
        rf.feature_importances_,
        index=features
    ).sort_values(ascending=False)

    plt.figure(figsize=(8,5))
    sns.barplot(
        x=importance,
        y=importance.index
    )
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()
