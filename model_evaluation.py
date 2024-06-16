import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Parameters
categories = ["Cephalometric", "Anteroposterior", "OPG"]

# Load data
X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

# Load model
model = load_model("best_model.keras")

# Compile the model with a loss function and metrics
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Predict
predictions = model.predict(X_val)
y_pred = np.argmax(predictions, axis=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Calculate metrics
conf_matrix = confusion_matrix(y_val, y_pred)
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average="macro")
recall = recall_score(y_val, y_pred, average="macro")
f1 = f1_score(y_val, y_pred, average="macro")

# Visualize confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=categories,
    yticklabels=categories,
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Print classification report
print(
    "Classification Report:\n",
    classification_report(y_val, y_pred, target_names=categories),
)

# Metrics for visualization
metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
}

# Create a DataFrame for the metrics
metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Score"])

# Visualize metrics
plt.figure(figsize=(10, 6))
sns.barplot(x="Metric", y="Score", data=metrics_df, palette="viridis")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.title("Evaluation Metrics")
plt.show()
