from utils.load import load_data
from models.svm import svm_model

if __name__ == "__main__":

    # Load Data
    Train_Data, Train_Labels = load_data('Train')
    

    # Define Applications
    applications = [0.5, 0.1, 0.9]
    K = 5

    # Test SVM    
    imbalanced_data = True
    prior = 0.5
    C_list = [10**-2, 10**-1, 1, 10, 100]
    
    gamma = 0.001

    options = {
        "m_pca": None,
        "gaussianize": False,
        "figures": False,
        "kernel_type": None
    }


    svm_model(Train_Data, Train_Labels, applications, K, C_list, gamma,  prior, imbalanced_data, options)