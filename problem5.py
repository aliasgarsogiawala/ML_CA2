# -------------------------------------------------
# PROBLEM 5: REGRESSION
# Prediction of Mental Workload Before Takeoff
# -------------------------------------------------

import pandas as pd
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
    'Sleep_Quality',
    'Emotional_Distraction',
    'In_Flight_Focus',
    'Communication_Load',
    'Peer_Support_Importance'
]

X = data[features]
y = data['Mental_Workload']

# -------------------------------------------------
# Step 3: Train-Test Split
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

# -------------------------------------------------
# Step 4: Define Regression Models (NaN-SAFE)
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

    "Random Forest Regressor": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ))
    ]),

    "Gradient Boosting Regressor": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", GradientBoostingRegressor(
            n_estimators=100,
            random_state=42
        ))
    ])
}

# -------------------------------------------------
# Step 5: Train & Evaluate Models
# -------------------------------------------------

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results.append([name, mae, mse, r2])

    print("\n====================================")
    print(f"Algorithm: {name}")
    print("====================================")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

results_df = pd.DataFrame(
    results,
    columns=["Model", "MAE", "MSE", "R² Score"]
)

print("\nMODEL PERFORMANCE COMPARISON")
print(results_df)

# -------------------------------------------------
# Step 6: Model Comparison Diagram
# -------------------------------------------------

plt.figure(figsize=(9,5))
sns.barplot(data=results_df, x="Model", y="R² Score")
plt.title("Model Comparison Based on R² Score")
plt.ylabel("R² Score")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# -------------------------------------------------
# Step 7: Actual vs Predicted (Best Model)
# -------------------------------------------------

best_model_name = results_df.sort_values(
    by="R² Score", ascending=False
).iloc[0]["Model"]

best_model = models[best_model_name]
best_preds = best_model.predict(X_test)

plt.figure(figsize=(7,6))
plt.scatter(y_test, best_preds, alpha=0.6)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)
plt.xlabel("Actual Mental Workload")
plt.ylabel("Predicted Mental Workload")
plt.title(f"Actual vs Predicted – {best_model_name}")
plt.tight_layout()
plt.show()

# -------------------------------------------------
# Step 8: Residual Analysis
# -------------------------------------------------

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
# Step 9: Feature Importance (Tree-Based Models)
# -------------------------------------------------

if best_model_name == "Random Forest Regressor":
    rf = best_model.named_steps["model"]
    importance = pd.Series(
        rf.feature_importances_,
        index=features
    ).sort_values(ascending=False)

    plt.figure(figsize=(8,5))
    sns.barplot(x=importance, y=importance.index)
    plt.title("Feature Importance – Random Forest Regressor")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()
