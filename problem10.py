# -------------------------------------------------
# PROBLEM 10: UNSUPERVISED LEARNING
# Association Rule Mining on Pre-Flight Mental States
# -------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mlxtend.frequent_patterns import apriori, association_rules

sns.set(style="whitegrid")

# -------------------------------------------------
# Step 1: Load Dataset
# -------------------------------------------------

data = pd.read_csv("cleaned_preflight_mental_state_data.csv")

# -------------------------------------------------
# Step 2: Select Relevant Features
# -------------------------------------------------

features = [
    'Mental_Workload',
    'Sleep_Quality',
    'Emotional_Distraction',
    'In_Flight_Focus',
    'Communication_Load',
    'Peer_Support_Importance'
]

df = data[features]

# -------------------------------------------------
# Step 3: Discretize Features (Low / High)
# -------------------------------------------------
# Since values are normalized (0–1)

binary_df = df.applymap(lambda x: 1 if x >= 0.5 else 0)

binary_df.columns = [
    f"{col}_High" for col in binary_df.columns
]

# -------------------------------------------------
# Step 4: Apply Apriori Algorithm
# -------------------------------------------------

frequent_itemsets = apriori(
    binary_df,
    min_support=0.2,
    use_colnames=True
)

print("\nFrequent Itemsets:")
print(frequent_itemsets)

# -------------------------------------------------
# Step 5: Generate Association Rules
# -------------------------------------------------

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.6
)

rules = rules.sort_values(
    by="lift",
    ascending=False
)

print("\nTop Association Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# -------------------------------------------------
# Step 6: Save Rules (Optional)
# -------------------------------------------------

rules.to_csv("association_rules_problem10.csv", index=False)
print("\nAssociation rules saved successfully.")
