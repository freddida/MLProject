import numpy as np
from sklearn.metrics import mean_squared_error, fbeta_score

'''
predicted = np.zeros(y_test.shape[0])
predicted.fill(-5)
predicted = predicted.reshape(-1, 1)
'''


def f_beta_score(predicted, real, verbose=False):
    # predicted = predictions_real
    # real = y_test_real

    real_binary = []
    predicted_binary = []

    for a in list(real):
        if (a >= -6):
            real_binary.append(1)
        else:
            real_binary.append(0)

    for a in list(predicted):
        if (a >= -6):
            predicted_binary.append(1)
        else:
            predicted_binary.append(0)

    fscore = fbeta_score(real_binary, predicted_binary, 2)
    real_mse = real[np.where(real >= -6)]
    predicted_mse = predicted[np.where(real >= -6)]
    mse = mean_squared_error(real_mse, predicted_mse)
    score = mse / fscore
    if verbose == True:
        print(f"F_score = {fscore}")
        print(f"MSE = {mse}")
        print(f"F_Beta Score (Beta=2): {score}")
    return score
