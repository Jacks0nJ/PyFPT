import numpy as np


# The data provided needs to be raw data - the log is then taken in the
# function uses the maximum likelihood
# Shen 2006 Statist. Med. 2006; 25:3023–3038
def log_normal_mean(data, position=None):
    data_log = np.log(data)
    data_log_mean = np.mean(data_log)
    data_log_std = np.std(data_log, ddof=1)  # Unbiased standard deviation
    n = len(data_log)
    if data_log_std**2 >= (n+4)/2:
        if isinstance(position, float):
            print('Possible convergance error in Shen2006 method at ' +
                  str(position))
        else:
            print('Possible convergance error in Shen2006 method')

    mean = np.exp(data_log_mean+0.5*data_log_std**2)
    return mean
