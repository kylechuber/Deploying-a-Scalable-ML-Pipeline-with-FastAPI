# add necessary import

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


# implement the first test. Change the function name and input as needed
def test_train_model_returns_random_forest():
    """
    Test that train_model returns a RandomForestClassifier.
    """
    data = pd.read_csv("data/census.csv")
    X, y, _, _ = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


# implement the second test. Change the function name and input as needed
def test_inference_returns_array_of_right_length():
    """
    Test that inference returns a numpy array with one prediction per row.
    """
    data = pd.read_csv("data/census.csv")
    X, y, _, _ = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == X.shape[0]


# implement the third test. Change the function name and input as needed
def test_compute_model_metrics_returns_three_floats():
    """
    Test that compute_model_metrics returns precision, recall, and fbeta as floats.
    """
    y = np.array([0, 1, 1, 0, 1])
    preds = np.array([0, 1, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
