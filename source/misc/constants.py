import os

PWD = os.getcwdb().decode()

hFea = {
        0: 'Mean of the integrated profile.',
        1: 'Standard deviation of the integrated profile',
        2: 'Excess kurtosis of the integrated profile',
        3: 'Skewness of the integrated profile',
        4: 'Mean of the DM-SNR curve.',
        5: 'Standard deviation of the DM-SNR curve.',
        6: 'Excess kurtosis of the DM-SNR curve.',
        7: 'Skewness of the DM-SNR curve.'
    }

CLASS_LABELS = [0, 1]

MVG_CLASS_PRIORS = [0.5 , 0.5]
SVM_K = 0

FMIN_L_BFGS_B_factr=0.0
FMIN_L_BFGS_B_maxiter=100000
FMIN_L_BFGS_B_maxfun=100000

GMM_PSI=0.01