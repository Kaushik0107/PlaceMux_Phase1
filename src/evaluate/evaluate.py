import os
import pandas as pd
from sklearn.metrics import accuracy_score

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("Model Accuracy:", accuracy)

    os.makedirs("logs", exist_ok=True)

    metrics = pd.DataFrame({
        "Model": ["DummyClassifier"],
        "Accuracy": [accuracy]
    })

    metrics.to_csv("logs/metrics.csv", index=False)

    print("Metrics successfully saved to logs/metrics.csv")

    return accuracy