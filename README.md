# Random Forest Iris Classification 🌲

A machine learning classification project using Random Forest to classify Iris flowers into three species:

- Setosa
- Versicolor
- Virginica

## Dataset

This project uses the built-in Iris dataset from Scikit-learn.

The dataset contains 150 samples and 4 features:

- Sepal length
- Sepal width
- Petal length
- Petal width

The target contains three classes:

- 0 → Setosa
- 1 → Versicolor
- 2 → Virginica

## Machine Learning Algorithm

### Random Forest Classifier

Random Forest is an ensemble learning algorithm that combines multiple Decision Trees to make predictions.

Instead of relying on a single Decision Tree, Random Forest creates multiple trees and combines their predictions.

## Model Configuration

The final model uses:

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42
)