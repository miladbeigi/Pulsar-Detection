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
    
    # plot_data(Train_Data, Train_Labels)
 
    # MVG    
    
    # model_type = "Tied"
    # MVG_Scores = train_mvg_models(Train_Data, Train_Labels, Evaluation_Data, model_type)
    
    # Predicted_Labels = assign_labels(MVG_Scores, 0.5, 1, 1)
    
    # Calibrate_MVG_Scores = train_logistic_regression(misc.make_row_shape(MVG_Scores), Evaluation_Labels, misc.make_row_shape(MVG_Scores), 10**-5, 0.5, False)

    # LR    
    scores = train_logistic_regression(Train_Data, Train_Labels, Evaluation_Data, 10**-5, 0.5, True)
    lg_scores = misc.make_row_shape(scores)
    LP = (scores > 0) * 1
    calibrated_scores = train_logistic_regression(lg_scores, LP, lg_scores, 10**-5, 0.5, True)

    bayes_plot(scores, calibrated_scores, Evaluation_Labels)