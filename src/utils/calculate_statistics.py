from sklearn.metrics import fbeta_score
from typing import List


def calculate_statistics_for_evaluation(
    y_true: List[int], y_pred: List[float], beta: float = 2, threshold: float = -6
) -> float:
    """
    Calculate F-beta score for binary classification evaluation.

    Parameters:
    - y_true (List[int]): True labels.
    - y_pred (List[float]): Predicted probabilities or scores.
    - beta (float, optional): Beta parameter for the F-beta score. Defaults to 2.
    - threshold (float, optional): Threshold for converting predictions to binary labels. Defaults to -6.

    Returns:
    - f_beta (float): F-beta score.
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    y_true_binary = (y_true >= threshold).astype(int)
    f_beta = fbeta_score(y_true_binary, y_pred_binary, beta=beta)
    return f_beta
