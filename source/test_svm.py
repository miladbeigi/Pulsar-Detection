from utils.load import load_data
from models.svm import svm_model, svm_evaluation

if __name__ == "__main__":

    # Load Data
    Train_Data, Train_Labels = load_data('Train')
    

    # Define Applications
    applications = [0.5, 0.1, 0.9]
    K = 3

    # Test SVM    
    imbalanced_data = True
    prior = 0.5
    # C_list = [10**-2, 10**-1, 1, 10, 100]
    C_list = [1]
    
    gamma = 0.001

    options = {
        "m_pca": None,
        "gaussianize": False,
        "figures": False,
        "kernel_type": None
    }


    svm_model(Train_Data, Train_Labels, applications, K, C_list, gamma,  prior, imbalanced_data, options)
    
    # C = 0.1
    # svm_evaluation(Train_D, Train_L, Evaluation_D,
    #                Evaluation_L, 0.001, 0.5, True, C, 0, None)
