import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLinearRegression


class CustomLinearRegression:
    """Simple Linear Regression trained with Gradient Descent from scratch."""

    def __init__(self, lr: float = 0.05, epochs: int = 1000):
        self.lr = lr
        self.epochs = epochs
        self.theta_0 = 0.0  # Intercept
        self.theta_1 = 0.0  # Slope
        self.cost_history = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        m = len(y)
        X_flat = X.flatten()

        for _ in range(self.epochs):
            predictions = self.theta_0 + self.theta_1 * X_flat
            cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
            self.cost_history.append(cost)

            # Gradients
            d_theta_0 = (1 / m) * np.sum(predictions - y)
            d_theta_1 = (1 / m) * np.sum((predictions - y) * X_flat)

            # Parameter updates
            self.theta_0 -= self.lr * d_theta_0
            self.theta_1 -= self.lr * d_theta_1

        print(f"[Model] Custom GD Trained -> Intercept: {self.theta_0:.4f}, Slope: {self.theta_1:.4f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.theta_0 + self.theta_1 * X.flatten()


def train_sklearn_model(X_train: np.ndarray, y_train: np.ndarray):
    """Trains a Scikit-Learn LinearRegression model."""
    model = SklearnLinearRegression()
    model.fit(X_train, y_train)
    print(f"[Model] Sklearn Trained -> Intercept: {model.intercept_:.4f}, Coef: {model.coef_[0]:.4f}")
    return model


if __name__ == "__main__":
    # Test execution
    X_dummy = np.array([[1.0], [2.0], [3.0], [4.0]])
    y_dummy = np.array([2.0, 4.0, 6.0, 8.0])

    model = CustomLinearRegression(lr=0.1, epochs=500)
    model.fit(X_dummy, y_dummy)
    preds = model.predict(np.array([[5.0]]))
    print("[Model Test] Prediction for input [5.0]:", preds[0])
