import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# === Load your dataset (replace 'your_dataset.csv' with your actual dataset) ===
df = pd.read_csv("cognitive_nutrition_dataset_25000 (1).csv")  # ← put your dataset file here

# === Define categorical columns to encode ===
categorical_cols = ['Gender', 'Menstrual', 'Behaviour', 'Character', 'Favourite_Food', 'Mood']
encoders = {}

# === Encode categorical features ===
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# === Define input features and target ===
X = df.drop(columns=["Mood"])
y = df["Mood"]

# === Split and train ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Evaluation (optional) ===
y_pred = model.predict(X_test)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=encoders['Mood'].classes_))

# === Save the trained model and encoders ===
joblib.dump(model, "mood_model.pkl")
joblib.dump(encoders, "encoders.pkl")

print("\n✅ Model and encoders saved as 'mood_model.pkl' and 'encoders.pkl'")
