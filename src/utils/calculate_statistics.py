from sklearn.metrics import fbeta_score


def calculate_statistics_for_evaluation(y_true, y_pred, beta=2, threshold=-6):
    """
    Calculate F-beta score for binary classification evaluation.

    Parameters:
    - y_true (array-like): True labels.
    - y_pred (array-like): Predicted probabilities or scores.
    - beta (float, optional): Beta parameter for the F-beta score. Defaults to 2.
    - threshold (float, optional): Threshold for converting predictions to binary labels. Defaults to -6.

    Returns:
    - f_beta (float): F-beta score.
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    y_true_binary = (y_true >= threshold).astype(int)
    f_beta = fbeta_score(y_true_binary, y_pred_binary, beta=beta)
    return print(f"F-beta Score: {f_beta}")
