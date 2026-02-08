# ---------------------------------------------
# DATA CLEANING AND TRANSFORMATION
# Pre-Flight Mental State Survey
# ---------------------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ---------------------------------------------
# Step 1: Load Dataset
# ---------------------------------------------

data = pd.read_csv("/Users/Asus/Downloads/CA2_dataset.csv")

print("Initial Data Preview:")
print(data.head())

print("\nInitial Dataset Info:")
data.info()

# ---------------------------------------------
# Step 2: Clean Column Names (IMPORTANT)
# ---------------------------------------------

# Remove leading/trailing spaces from column names
data.columns = data.columns.str.strip()

# Rename columns to ML-friendly names
data.rename(columns={
    '1. Age Group': 'Age_Group',
    '2. Do you have any interest or exposure to aviation topics?': 'Aviation_Exposure',
    '3. How confident are you in making decisions under pressure in general?': 'Decision_Confidence',
    '4. Common sources of pre-flight stress (according to you)': 'Preflight_Stress_Source',
    '5. Mental workload you think you would have before takeoff': 'Mental_Workload',
    '6. Sleep quality you think would affect a pilot before flight': 'Sleep_Quality',
    '7. Emotional distraction before starting a flight': 'Emotional_Distraction',
    '8. During a flight, how focused do you think a pilot would feel?': 'In_Flight_Focus',
    '9. In your opinion, how often might a pilot hesitate during decisions?': 'Decision_Hesitation',
    '10. How difficult do you think managing tasks during flight would feel?': 'Task_Difficulty',
    '11. How high do you think the communication load is during flight?': 'Communication_Load',
    '12. According to you, how confident would a pilot feel while making decisions in-flight?': 'In_Flight_Confidence',
    '13. Confidence in handling unexpected situations': 'Unexpected_Situation_Confidence',
    '14. Compared to a normal day, how do you think a pilot’s performance might feel?': 'Performance_Comparison',
    '15. Do you think pre-flight mental state affects a pilot’s decisions?': 'Mental_State_Impact',
    '16. Do you think pilots are aware of their mental state during flight?': 'Mental_Awareness',
    '17. How often do you think a pilot uses formal mental preparation techniques (e.g., visualization, checklists) before a flight?': 'Mental_Preparation',
    '18. In your opinion, how important is peer support (from co-pilots or crew) to a pilot’s confidence during flight?': 'Peer_Support_Importance',
    '19-. In your opinion, what helps a pilot stay focused? (Optional short answer)': 'Focus_Support_Factors'
}, inplace=True)

print("\nColumns after renaming:")
print(data.columns)

# ---------------------------------------------
# Step 3: Handle Missing Values
# ---------------------------------------------

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

for col in ordinal_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    data[col] = data[col].fillna(data[col].mean())

categorical_columns = [
    'Age_Group',
    'Aviation_Exposure',
    'Preflight_Stress_Source',
    'Decision_Hesitation',
    'Task_Difficulty',
    'Unexpected_Situation_Confidence',
    'Performance_Comparison',
    'Mental_State_Impact',
    'Mental_Awareness',
    'Mental_Preparation'
]

for col in categorical_columns:
    data[col] = data[col].fillna(data[col].mode()[0])

data['Focus_Support_Factors'] = data['Focus_Support_Factors'].fillna("Not Specified")

# ---------------------------------------------
# Step 4: Encode Categorical Variables
# ---------------------------------------------

encoder = LabelEncoder()
for col in categorical_columns:
    data[col] = encoder.fit_transform(data[col])

# ---------------------------------------------
# Step 5: Normalize Ordinal Features
# ---------------------------------------------

scaler = MinMaxScaler()
data[ordinal_columns] = scaler.fit_transform(data[ordinal_columns])

# ---------------------------------------------
# Step 6: Final Validation
# ---------------------------------------------

print("\nCleaned Dataset Preview:")
print(data.head())

print("\nCleaned Dataset Info:")
data.info()

# ---------------------------------------------
# Step 7: Save Cleaned Dataset
# ---------------------------------------------

data.to_csv("cleaned_preflight_mental_state_data.csv", index=False)
print("\nCleaned dataset saved successfully.")
