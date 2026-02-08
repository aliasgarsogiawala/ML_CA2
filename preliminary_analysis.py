# -------------------------------------------------
# PRELIMINARY ANALYSIS
# Pre-Flight Mental State & Decision Confidence
# FINAL VERIFIED VERSION
# -------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

# -------------------------------------------------
# Step 1: Load Cleaned Dataset
# -------------------------------------------------

data = pd.read_csv("cleaned_preflight_mental_state_data.csv")

print("\nDataset Info:")
data.info()

# -------------------------------------------------
# Step 2: Define Ordinal Numeric Columns
# -------------------------------------------------

ordinal_columns = [
    'Decision_Confidence',
    'Mental_Workload',
    'Sleep_Quality',
    'Emotional_Distraction',
    'In_Flight_Focus',
    'Communication_Load',
    'In_Flight_Confidence',
    'Peer_Support_Importance'
]

# -------------------------------------------------
# Step 3: Basic Statistical Summary
# -------------------------------------------------

print("\nBasic Statistical Summary:")
print(data[ordinal_columns].describe())

# -------------------------------------------------
# Step 4: Detailed Statistical Measures
# -------------------------------------------------

stats_table = pd.DataFrame({
    'Mean': data[ordinal_columns].mean(),
    'Median': data[ordinal_columns].median(),
    'Mode': data[ordinal_columns].mode().iloc[0],
    'Minimum': data[ordinal_columns].min(),
    'Maximum': data[ordinal_columns].max(),
    'Range': data[ordinal_columns].max() - data[ordinal_columns].min(),
    'Variance': data[ordinal_columns].var(),
    'Standard Deviation': data[ordinal_columns].std(),
    'Skewness': data[ordinal_columns].apply(lambda x: skew(x.dropna())),
    'Kurtosis': data[ordinal_columns].apply(lambda x: kurtosis(x.dropna())),
    'Coefficient of Variation': data[ordinal_columns].std() / data[ordinal_columns].mean()
})

print("\nDetailed Statistical Measures:")
print(stats_table)

# -------------------------------------------------
# Step 5: Quartiles & Deciles
# -------------------------------------------------

print("\nQuartiles:")
print(data[ordinal_columns].quantile([0.25, 0.5, 0.75]))

print("\nDeciles:")
print(data[ordinal_columns].quantile(np.arange(0.1, 1.0, 0.1)))

# -------------------------------------------------
# Step 6: SAFE BAR PLOT FUNCTION (ORDINAL DATA)
# -------------------------------------------------

def safe_bar_plot(series, title, xlabel):
    counts = series.dropna().round(2).value_counts().sort_index()

    print(f"\nValue Counts – {title}")
    print(counts)

    if counts.empty:
        print("⚠️ Skipping plot (no data)")
        return

    plt.figure(figsize=(8,5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(axis='y')
    plt.show()

# -------------------------------------------------
# Step 7: Ordinal Variable Plots
# -------------------------------------------------

safe_bar_plot(data['Decision_Confidence'], "Decision Confidence Under Pressure", "Confidence Level")
safe_bar_plot(data['Mental_Workload'], "Mental Workload Before Takeoff", "Workload Level")
safe_bar_plot(data['Sleep_Quality'], "Sleep Quality Before Flight", "Sleep Quality Level")
safe_bar_plot(data['In_Flight_Focus'], "In-Flight Focus Level", "Focus Level")
safe_bar_plot(data['Communication_Load'], "Communication Load During Flight", "Load Level")
safe_bar_plot(data['In_Flight_Confidence'], "In-Flight Decision Confidence", "Confidence Level")
safe_bar_plot(data['Peer_Support_Importance'], "Importance of Peer Support", "Importance Level")

# -------------------------------------------------
# Step 8: Box Plot – Emotional Distraction
# -------------------------------------------------

plt.figure(figsize=(6,5))
sns.boxplot(y=data['Emotional_Distraction'])
plt.title("Emotional Distraction Before Flight")
plt.ylabel("Distraction Level")
plt.grid(axis='y')
plt.show()

# -------------------------------------------------
# Step 9: Categorical Variable Plots
# -------------------------------------------------

plt.figure(figsize=(7,5))
sns.countplot(x=data['Task_Difficulty'])
plt.title("Task Difficulty During Flight")
plt.xlabel("Difficulty Level")
plt.ylabel("Count")
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(6,6))
data['Aviation_Exposure'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    startangle=90
)
plt.title("Aviation Exposure Distribution")
plt.ylabel("")
plt.show()

plt.figure(figsize=(6,6))
data['Mental_State_Impact'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    startangle=90
)
plt.title("Perceived Impact of Pre-Flight Mental State")
plt.ylabel("")
plt.show()

# -------------------------------------------------
# Step 10: Correlation Heatmap (ORDINAL ONLY)
# -------------------------------------------------

plt.figure(figsize=(12,8))
corr = data[ordinal_columns].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Pre-Flight Mental State & Decision Variables")
plt.show()
