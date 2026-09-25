import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculates RMSE, MAE, and R^2 score for regression predictions."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    metrics = {"RMSE": rmse, "MAE": mae, "R2": r2}

    print("\n" + "=" * 35)
    print("      MODEL EVALUATION REPORT      ")
    print("=" * 35)
    print(f" Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f" Mean Absolute Error (MAE)    : {mae:.4f}")
    print(f" R² Score (Variance Explained): {r2:.4f}")
    print("=" * 35 + "\n")

    return metrics


def plot_regression_results(
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    feature_name: str = "Feature",
    target_name: str = "Target",
    save_path: str = None,
):
    """Plots actual test data points against the fitted linear regression line."""
    plt.figure(figsize=(9, 5))
    plt.scatter(X_test, y_test, color="teal", alpha=0.2, label="Actual Test Data")
    plt.plot(X_test, y_pred, color="crimson", linewidth=2, label="Regression Line")
    plt.title(f"Linear Regression: {feature_name} vs {target_name}")
    plt.xlabel(f"{feature_name} (Standardized)")
    plt.ylabel(target_name)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"[Evaluation] Plot saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    
    y_true_dummy = np.array([10, 20, 30, 40, 50])
    y_pred_dummy = np.array([12, 19, 29, 41, 52])
    metrics = evaluate_predictions(y_true_dummy, y_pred_dummy)