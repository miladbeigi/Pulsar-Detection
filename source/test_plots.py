from utils.load import load_data
from pre_processing.gaussianize import features_gaussianization
from utils.plot import plot_data, bayes_plot
from models.multivariate_gaussian import train_mvg_models
from models.logistic_regression import train_logistic_regression
import misc.misc as misc
import matplotlib.pyplot as plt
from utils.model_evaluation import bayes_error_plot, assign_labels


if __name__ == '__main__':
    Train_Data, Train_Labels = load_data('Train')
    
    Train_Data = features_gaussianization(Train_Data)
    Evaluation_Data, Evaluation_Labels = load_data('Test')
    
    plot_data(Train_Data, Train_Labels)