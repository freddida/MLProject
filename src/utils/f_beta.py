import numpy as np
from sklearn.metrics import mean_squared_error, fbeta_score


def calculate_f_beta_sklearn(y_true, y_pred, beta=2, threshold=-6):
    # Convert regression predictions to binary classification based on threshold
    y_pred_binary = (y_pred >= threshold).astype(int)

    # Convert true values to binary
    y_true_binary = (y_true >= threshold).astype(int)

    # Calculate F-beta score using sklearn's fbeta_score
    f_beta = fbeta_score(y_true_binary, y_pred_binary, beta=beta)

    # Calculate mean squared error for high-risk events
    mse_hr = mean_squared_error(y_true[y_true >= threshold], y_pred[y_true >= threshold])

    # Calculate the final score
    score = mse_hr / f_beta

    return score, f_beta, mse_hr
