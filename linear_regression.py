from sklearn.linear_model import LinearRegression
import numpy as np

def get_true_betas(X, y):
    betas = []
    for i in range(y.shape[1]):
        reg = LinearRegression().fit(X, y[:, i])
        betas.append(np.array(reg.coef_))
    return betas

def get_oracle_preds(X, y):
    reg = LinearRegression().fit(X, y)
    preds = reg.predict(X)
    return preds




