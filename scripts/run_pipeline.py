print("Script started")


import pandas as pd
import joblib
import os
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# -----------------------------
# Load the dataset
# -----------------------------

iris = load_iris(as_frame=True)

df = iris.frame

X = df.drop(columns=["target"])

y = df["target"]

# -----------------------------
# Split the dataset
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# -----------------------------
# Create the pipeline
# -----------------------------

numeric_features = X.columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(random_state=42))
])

# -----------------------------
# Train the pipeline
# -----------------------------

pipeline.fit(X_train, y_train)

#-----------------------------
# artifacts
#-----------------------------

os.makedirs("artifacts", exist_ok=True)

joblib.dump(
    pipeline,
    "artifacts/model_pipeline.pkl"
)

print("Pipeline saved successfully.")


# -----------------------------
# Make predictions
# -----------------------------

predictions = pipeline.predict(X_test)

# -----------------------------
# Calculate metrics
# -----------------------------

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    average="weighted"
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted"
)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted"
)

#-----------------------------
# metrics dataframe
#-----------------------------

metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1
    ]
})

#save metrics 

metrics.to_csv(
    "artifacts/metrics.csv",
    index=False
)

print("Metrics saved successfully.")

#verrify the saved files

print("\nArtifacts Created:")

print(os.listdir("artifacts"))

#verify the saved pipeline

loaded_pipeline = joblib.load(
    "artifacts/model_pipeline.pkl"
)

print("Saved pipeline loaded successfully.")


# -----------------------------
# Display the results
# -----------------------------

print("Model Evaluation")
print("----------------")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

