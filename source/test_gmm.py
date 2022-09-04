from utils.load import load_data
from models.gmm import gmm_model

if __name__ == "__main__":

    # Load Data
    Train_Data, Train_Labels = load_data('Train')
    

    # Define Applications
    applications = [0.5, 0.1, 0.9]
    K = 3

    # Test GMM
    imbalanced_data = True
    prior = 0.5
    
    number_of_components = 4
    model_type = "full"
    
    options = {
        "m_pca": None,
        "gaussianize": False,
        "figures": True,
        "model_type": "naive-tied"
    }

    gmm_model(Train_Data, Train_Labels, applications, K, number_of_components, options)