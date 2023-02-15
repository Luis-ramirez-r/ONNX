import numpy as np
from sklearn.datasets import make_regression
import catboost
import pandas as pd

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
# import  random forest regressor
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load
import onnxmltools


from os import path
from os import listdir
from os.path import isfile, join
from skl2onnx.common.data_types import FloatTensorType


# Generación del dataset
n_samples = 1000
n_features = 40
X, y = make_regression(n_samples=n_samples, n_features=n_features, random_state=0)
