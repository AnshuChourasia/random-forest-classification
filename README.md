# Random Forest Iris Classification 🌲

A machine learning classification project using **Random Forest** to classify Iris flowers into three species:

* Setosa
* Versicolor
* Virginica

## Dataset

This project uses the built-in **Iris dataset** from Scikit-learn.

The dataset contains **150 samples** and **4 features**:

* Sepal length
* Sepal width
* Petal length
* Petal width

The target contains three classes:

* `0` → Setosa
* `1` → Versicolor
* `2` → Virginica

## Machine Learning Algorithm

### Random Forest Classifier

Random Forest is an ensemble learning algorithm that combines multiple Decision Trees to make predictions.

Instead of relying on a single Decision Tree, Random Forest creates multiple trees and combines their predictions to produce a final prediction.

## Model Configuration

The final model uses:

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42
)
```

## Hyperparameter Tuning

Two important Random Forest hyperparameters were tested.

### 1. Number of Trees (`n_estimators`)

The following values were tested:

| Trees | Test Accuracy |
| ----: | ------------: |
|    10 |        96.67% |
|    25 |        93.33% |
|    50 |        96.67% |
|   100 |        96.67% |
|   150 |        96.67% |

Increasing the number of trees beyond 100 did not improve the test accuracy.

Therefore, `n_estimators=100` was selected.

### 2. Maximum Tree Depth (`max_depth`)

The following values were tested:

| Max Depth | Test Accuracy |
| --------: | ------------: |
|         1 |        96.67% |
|         2 |        90.00% |
|         3 |        96.67% |
|         4 |        96.67% |
|         5 |        93.33% |
|         6 |        90.00% |
|         8 |        90.00% |
|        10 |        90.00% |
|      None |        90.00% |

`max_depth=3` was selected because it achieved the highest test accuracy while keeping the model relatively simple.

## Cross-Validation

5-fold cross-validation was performed to evaluate the model across multiple data splits.

The results were:

```text
96.67%
96.67%
93.33%
96.67%
100.00%
```

### Cross-Validation Results

* **Mean CV Accuracy:** 96.67%
* **Standard Deviation:** 2.11%

The similar test and cross-validation performance suggests that the model performs consistently across different data splits.

## Model Evaluation

The final Random Forest model achieved:

* **Test Accuracy:** 96.67%
* **Mean 5-Fold Cross-Validation Accuracy:** 96.67%
* **Cross-Validation Standard Deviation:** 2.11%

## Evaluation Metrics

The project evaluates the model using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Classification Report
* Feature Importance

## Feature Importance

Random Forest provides feature importance scores that indicate which features contributed most to the model's predictions.

The project visualizes feature importance using a bar chart.

## Visualizations

The project includes:

* Confusion Matrix Heatmap
* Feature Importance Bar Chart

## What I Learned

Through this project, I practiced:

* Loading datasets using Scikit-learn
* Exploratory Data Analysis
* Train-test splitting
* Random Forest classification
* Model evaluation
* Confusion matrices
* Classification reports
* Feature importance
* Hyperparameter tuning
* Cross-validation
* Comparing model configurations
* Selecting a final model based on experimental results

## Project Structure

```text
random-forest-classification/
│
├── random-forest.py
├── README.md
└── requirements.txt
```

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python random-forest.py
```

## Conclusion

The Random Forest classifier achieved **96.67% test accuracy** on the Iris dataset.

The final configuration of:

```text
n_estimators = 100
max_depth = 3
```

provided strong performance while keeping the model relatively simple.

Cross-validation also produced a mean accuracy of **96.67%**, supporting the final model selection.

This project provided practical experience with **ensemble learning, hyperparameter tuning, model evaluation, feature importance, and cross-validation**.
