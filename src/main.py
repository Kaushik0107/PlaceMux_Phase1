from model.train import train_model
from evaluate.evaluate import evaluate_model

print("Starting Machine Learning Pipeline...\n")

model, X_test, y_test = train_model()

accuracy = evaluate_model(model, X_test, y_test)

print("\nPipeline Completed Successfully.")