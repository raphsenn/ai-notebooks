# Author: Raphael Senn

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class PolynomialRegression:
    def __init__(self, degree: int) -> None:
        self.degree = degree
        self.linear_model = LinearRegression()
        self.feature_transformation = PolynomialFeatures(degree=degree)

    def fit(self, X_train: np.array, y_train: np.array) -> None:
        X_transformed = self.feature_transformation.fit_transform(X_train)
        self.linear_model.fit(X_transformed, y_train)

    def predict(self, X: np.array) -> np.array:
        X_transformed = self.feature_transformation.fit_transform(X)
        return self.linear_model.predict(X_transformed)

    def __repr__(self) -> str:
        return f"Polynomial Model with degree {self.degree}"