import numpy as np
import scipy.stats as stats 

def corrected_repeated_kFold_cv_test(data1, data2, n1, n2, alpha):
    """
    Perform corrected repeated k-fold cross-validation test to evaluate the replicability of significance tests for comparing learning algorithms.
    This implments the test as suggested in the paper "A corrected repeated k-fold cross-validation test for replicability in psychophysiology" by Bouckaert et al. (2004).

    Parameters:
    data1 (array-like): The first dataset.
    data2 (array-like): The second dataset.
    n1 (int): The number of training samples in each fold.
    n2 (int): The number of test samples ind each fold.
    alpha (float): The significance level.

    Returns:
    tuple: A tuple containing the critical value and the p-value.
    """
    n = len(data1)
    if n != len(data2):
        raise ValueError("The datasets must have the same length.")
    # estimate the mean
    m = 1 / n * sum([data1[i] - data2[i] for i in range(n)])
    # estimate the standard deviation
    stdv_sq = np.sqrt(1 / (n - 1) * sum([(data1[i] - data2[i] - m) ** 2 for i in range(n)]))
    # calculate the test statistic
    t = m / np.sqrt((1/n + n2/n1)* stdv_sq)
    # degrees of freedom
    df = n - 1
    # calculate the critical value
    cv = stats.t.ppf(1.0 - alpha, df)
    # calculate the p-value
    p = (1.0 - stats.t.cdf(abs(t), df)) * 2.0
    return cv, p
