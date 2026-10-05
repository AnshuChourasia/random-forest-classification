# ============================================================
# RANDOM FOREST IRIS CLASSIFICATION
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["target"] = iris.target

print("\n========== FIRST 5 ROWS ==========")
print(df.head())


# ============================================================
# 3. BASIC DATASET INFORMATION
# ============================================================

print("\n========== DATASET INFO ==========")
print(df.info())

print("\n========== DESCRIPTIVE STATISTICS ==========")
print(df.describe())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATES ==========")
print(df.duplicated().sum())


# ============================================================
# 4. TARGET DISTRIBUTION
# ============================================================

print("\n========== TARGET DISTRIBUTION ==========")
print(df["target"].value_counts())

print("\n========== TARGET NAMES ==========")
print(iris.target_names)


# ============================================================
# 5. DEFINE FEATURES AND TARGET
# ============================================================

X = df.drop("target", axis=1)
y = df["target"]


# ============================================================
# 6. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n========== DATA SPLIT ==========")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ============================================================
# 7. BASELINE RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

print("\n========== RANDOM FOREST MODEL ==========")
print(f"Number of trees: {model.n_estimators}")


# ============================================================
# 8. PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

y_proba = model.predict_proba(X_test)

print("\n========== PREDICTIONS ==========")
print(y_pred)

print("\n========== PREDICTION PROBABILITIES ==========")
print(y_proba)


# ============================================================
# 9. TRAINING AND TEST ACCURACY
# ============================================================

train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print("\n========== TRAINING vs TEST ACCURACY ==========")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\n========== CONFUSION MATRIX ==========")
print(cm)


# ============================================================
# 11. EVALUATION METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

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

print("\n========== EVALUATION METRICS ==========")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)


# ============================================================
# 13. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(feature_importance)


# ============================================================
# 14. CONFUSION MATRIX VISUALIZATION
# ============================================================

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

plt.show()


# ============================================================
# 15. FEATURE IMPORTANCE VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 5))

sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_importance
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()

plt.show()


# ============================================================
# 16. N_ESTIMATORS TUNING
# ============================================================

n_estimators_values = [10, 25, 50, 100, 150, 200]

print("\n========== N_ESTIMATORS vs TEST ACCURACY ==========")

for n in n_estimators_values:

    model = RandomForestClassifier(
        n_estimators=n,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    accuracy = model.score(
        X_test,
        y_test
    )

    print(f"Trees={n}: Accuracy={accuracy:.4f}")


# ============================================================
# 17. MAX_DEPTH TUNING
# ============================================================

max_depth_values = [1, 2, 3, 4, 5, 6, 8, 10, None]

print("\n========== MAX_DEPTH vs TEST ACCURACY ==========")

for depth in max_depth_values:

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    accuracy = model.score(
        X_test,
        y_test
    )

    print(f"Max Depth={depth}: Accuracy={accuracy:.4f}")


# ============================================================
# 18. CROSS-VALIDATION
# ============================================================

final_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42
)

cv_scores = cross_val_score(
    final_model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\n========== RANDOM FOREST CROSS-VALIDATION ==========")

print("CV Scores:", cv_scores)

print(
    f"Mean CV Accuracy: {cv_scores.mean():.4f}"
)

print(
    f"Standard Deviation: {cv_scores.std():.4f}"
)


# ============================================================
# 19. FINAL RANDOM FOREST MODEL
# ============================================================

final_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42
)

final_model.fit(X_train, y_train)

final_predictions = final_model.predict(X_test)

final_accuracy = accuracy_score(
    y_test,
    final_predictions
)

print("\n========== FINAL RANDOM FOREST MODEL ==========")

print(
    f"Final Test Accuracy: {final_accuracy:.4f}"
)
