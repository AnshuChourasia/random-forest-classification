# ============================================================
# RANDOM FOREST CLASSIFICATION - IRIS DATASET
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

data = load_iris()

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df["target"] = data.target


# ============================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n========== First 5 Rows ==========")
print(df.head())


print("\n========== Dataset Info ==========")
df.info()


print("\n========== Statistics ==========")
print(df.describe())


print("\n========== Shape ==========")
print(df.shape)


print("\n========== Missing Values ==========")
print(df.isnull().sum())


print("\n========== Duplicates ==========")
print(df.duplicated().sum())


print("\n========== Target Distribution ==========")
print(df["target"].value_counts())


print("\n========== Target Names ==========")
print(data.target_names)


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("target", axis=1)
y = df["target"]


print("\n========== X and y ==========")
print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\n========== Train/Test Split ==========")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ============================================================
# 6. CREATE RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42
)


# ============================================================
# 7. TRAIN MODEL
# ============================================================

model.fit(X_train, y_train)


print("\n========== Model Information ==========")
print("Number of Trees:", len(model.estimators_))


# ============================================================
# 8. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


print("\n========== First 10 Predictions ==========")
print(y_pred[:10])


# ============================================================
# 9. PREDICTION PROBABILITIES
# ============================================================

y_prob = model.predict_proba(X_test)


print("\n========== First 10 Prediction Probabilities ==========")
print(y_prob[:10])


# ============================================================
# 10. TRAINING VS TESTING ACCURACY
# ============================================================

train_accuracy = model.score(
    X_train,
    y_train
)

test_accuracy = model.score(
    X_test,
    y_test
)


print("\n========== Training vs Testing Accuracy ==========")
print(f"Training Accuracy : {train_accuracy:.4f}")
print(f"Testing Accuracy  : {test_accuracy:.4f}")


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


print("\n========== Confusion Matrix ==========")
print(cm)


# ============================================================
# 12. ACCURACY, PRECISION, RECALL AND F1
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)


print("\n========== Final Model Performance ==========")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# ============================================================
# 13. CLASSIFICATION REPORT
# ============================================================

print("\n========== Classification Report ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=data.target_names
    )
)


# ============================================================
# 14. FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})


importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)


print("\n========== Random Forest Feature Importance ==========")
print(importance_df)


# ============================================================
# 15. CONFUSION MATRIX HEATMAP
# ============================================================

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=data.target_names,
    yticklabels=data.target_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest - Confusion Matrix")

plt.tight_layout()
plt.show()


# ============================================================
# 16. FEATURE IMPORTANCE VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()