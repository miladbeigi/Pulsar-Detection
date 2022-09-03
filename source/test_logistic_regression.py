from utils.load import load_data
from models.logistic_regression import logistic_regression, lg_evaluation

if __name__ == "__main__":

    # Load Data
    Train_D, Train_L = load_data('Train')
    

    # Define Applications
    applications = [0.5, 0.1, 0.9]
    K = 5

    # Test Logistic Regression
    imbalanced = False
    lambda_list = [10**-5, 10**-4, 10**-3,
                   10**-2, 10**-1, 1, 10, 100, 1000, 10000]
    prior = 0.5
    options = {"m_pca": None, "quadratic": False,
               "gaussianize": False, "figures": False}

    logistic_regression(Train_D, Train_L, applications,
                        K, lambda_list, prior, imbalanced, options)


    # Evaluation
    Evaluation_D, Evaluation_L = load_data('Test')
    
    lg_evaluation(Train_D, Train_L, Evaluation_D,
                  Evaluation_L, 10**-5, prior, imbalanced)
