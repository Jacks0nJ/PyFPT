import numpy as np


# The most basic way to estimate the error, assuming symmetric errors
# SHOULD THIS USE n-1 or n IN THE STD??????????????????????????
def log_normal_errors(ws, Z_alpha=1):
    log_w = np.log(ws)
    log_var = np.var(log_w, ddof=1)  # unbiased variance
    log_mean = np.mean(log_w)
    n = len(ws)
    log_err = Z_alpha*np.sqrt(log_var/n+(log_var**2)/(2*n-2))
    upper_err = n*np.exp(log_mean+log_var/2)*(np.exp(log_err)-1)
    lower_err = n*np.exp(log_mean+log_var/2)*(1-np.exp(-log_err))
    return np.array([lower_err, upper_err])
