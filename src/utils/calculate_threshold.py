from typing import Tuple
import numpy as np
from sklearn.metrics import fbeta_score


def find_best_threshold(
    y_true: np.ndarray, risk_predictions: np.ndarray
) -> Tuple[float, float]:
    """
    Find the best threshold for converting risk predictions to binary predictions
    based on maximizing the F-beta score.

    Parameters:
    - y_true (np.ndarray): True labels (binary).
    - risk_predictions (np.ndarray): Predicted risk probabilities.

    Returns:
    - Tuple[float, float]: Best threshold and its corresponding F-beta score.
    """
    best_threshold = None
    best_score = -1

    # Iterate over a range of thresholds
    for threshold in np.linspace(0, 1, 1000):
        # Convert risk predictions to binary predictions based on the current threshold
        y_pred = np.where(risk_predictions >= threshold, 1, 0)

        # Calculate the F-beta score for the current threshold
        score = fbeta_score(y_true, y_pred, beta=2)

        # Update the best threshold and score if the current score is better
        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score
