import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.svm import SVC

DATASET_REGISTRY = {}

MODEL_REGISTRY = {
    "linear_regression": LinearRegression(),
    "random_forest_regressor": RandomForestRegressor(),
    "xgboost_regressor": XGBRegressor(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=100),
    "svm_classifier": SVC(kernel="linear"),
    "XGBClassifier": XGBClassifier()
}
