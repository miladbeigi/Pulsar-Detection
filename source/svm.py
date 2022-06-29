import numpy
import pylab
import scipy.optimize
from scipy.linalg import norm
import sklearn.datasets
from load import load_data
from pca_lda import calculate_pca, normalize_data
from gaussianize import gaussianization
import misc

def kfold_split(D, L, k = 10, seed=0):
    D_split = []
    L_split = []
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1]) 
    D = D[:, idx]
    L = L[idx]
    for i in range(0, k):
        D_split.append(D[:, int(i/k*D.shape[1]):int((i+1)/k*D.shape[1])])
        L_split.append(L[int(i/k*D.shape[1]):int((i+1)/k*D.shape[1])])
    return D_split, L_split

def kfold_validation(D, L, to_eval = 0, k = 10):

    DTR = numpy.empty(shape=[D.shape[0], 0])
    LTR = numpy.empty(shape = 0)
    DTE = numpy.empty(shape=[D.shape[0], 0])
    LTE = numpy.empty(shape = 0)

    D_kfold, L_kfold = kfold_split(D, L, k)
    
    for j in range (0, k):
        if(j != to_eval):
            to_add_data = numpy.array(D_kfold[to_eval])
            to_add_label = numpy.array(L_kfold[to_eval])
            DTR = numpy.hstack((DTR, to_add_data))
            LTR = numpy.hstack((LTR, to_add_label))
    
        else :
            to_add_data = numpy.array(D_kfold[to_eval])
            to_add_label = numpy.array(L_kfold[to_eval])
            DTE = numpy.hstack((DTE, to_add_data))
            LTE = numpy.hstack((LTE, to_add_label))
    return DTR, LTR, DTE, LTE


def train_SVM_linear(DTR, LTR, DTE, LTE, C = 1):

    Z = numpy.zeros(LTR.shape)
    Z[ LTR == 1] = 1
    Z[ LTR == 0] = -1
    
    DTREXT = numpy.vstack([DTR, numpy.ones((1, DTR.shape[1]))]) 

    H = numpy.dot(DTREXT.T, DTREXT)
    H = misc.mcol(Z) * misc.vrow(Z) * H

    def JDual(alpha):
        Ha = numpy.dot(H, misc.mcol(alpha))
        aHa = numpy.dot(misc.vrow(alpha), Ha)
        a1 = alpha.sum()

        return -0.5* aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size)

    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -loss, -grad

    def JPrimal(w):
        S = numpy.dot(misc.vrow(w), DTREXT)
        loss = numpy.maximum(numpy.zeros(S.shape), 1 -Z*S).sum()
        return -0.5*numpy.linalg.norm(w)**2 + C*loss
    
    def evaluation_SVM_linear(DTE, LTE, wStar):
     
        w = wStar[0:-1]
        b = wStar[-1]
        D = numpy.dot(misc.vrow(w), DTE) + b

        print(D.shape)
        D = numpy.int32(D > 0)
        err = (D != LTE).sum()
        err = err / D.shape[1] * 100
        return err
    
    print("Error Rate is Loading: ")

    alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
        LDual,
        numpy.zeros(DTR.shape[1]),
        bounds = [(0,C)] * DTR.shape[1],
        factr = 0.0,
        maxiter = 100000,
        maxfun = 100000,
    )

    print("Error Rate Loaded: ")

    wStar = numpy.dot(DTREXT, misc.mcol(alphaStar) * misc.mcol(Z))
    
    print("Error Rate is here Loaded: ")

    err = evaluation_SVM_linear(DTE, LTE, wStar)
    print("Primal Loss: ", JPrimal(wStar))
    print("Dual Loss: ", JDual(alphaStar)[0])
    print("Duality Gap: ", LDual(alphaStar)[0])
    print("Error Rate: ", err)

    return err

def train_SVM_kernel(DTR, LTR, DTE, LTE, C = 1, K = 0, gamma = 1, d = 2, c = 1, RBF = False):
    
    Z = numpy.zeros(LTR.shape)
    Z[ LTR == 1] = 1
    Z[ LTR == 0] = -1

    #Radial Basis Function
    if RBF:
        Dist = misc.mcol((DTR**2).sum(0)) + misc.vrow((DTR**2).sum(0)) - 2*numpy.dot(DTR.T, DTR)
        kernel = numpy.exp(-gamma*Dist) + K
    #Polinomial
    else:
        kernel = (numpy.dot(DTR.T, DTR) + c)**d
        
    H = misc.mcol(Z)*misc.vrow(Z)*kernel

    def JDual(alpha):
        Ha = numpy.dot(H, misc.mcol(alpha))
        aHa = numpy.dot(misc.vrow(alpha), Ha)
        a1 = alpha.sum()

        return -0.5* aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size)

    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -loss, -grad

    
    def evaluation_SVM_kernel(s):
        label = numpy.int32(s > 0)
        err = (label != LTE).sum()
        err = err / LTE.shape[0] * 100
        print(err)
        return err
    

    alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
        LDual,
        numpy.zeros(DTR.shape[1]),
        bounds = [(0,C)] * DTR.shape[1],
        factr = 0.0,
        maxiter = 100000,
        maxfun = 100000,
    )

    
    s = numpy.zeros(DTE.shape[1])

    #Radial Basis Function
    if RBF:
        for i in range(0, DTE.shape[1]):
            for j in range(0, DTR.shape[1]):
                exp = numpy.linalg.norm(DTR[:, j] - DTE[:, i]) ** 2 * gamma
                kern = numpy.exp(-exp) + K
                s[i] += alphaStar[j] * Z[j] * kern
    #Polinomial
    else:
        for i in range(0, DTE.shape[1]):
            for j in range(0, DTR.shape[1]):

                kern = (numpy.dot(DTR[:, j], DTE[:, i]) + c)**d
                s[i] += alphaStar[j] * Z[j] * kern


    err = evaluation_SVM_kernel(s)
    print("Dual Loss: ", JDual(alphaStar)[0])
    print("Error Rate: ", err)


    return err
   
def svm_model(D, L):
    # K-Fold
    K = 5
    kf = misc.k_fold(K)

    for app in [0.5, 0.1, 0.9]:
        
        print(app)
        
        min_minDCF = []

        for train_index, test_index in kf.split(D.T):

            DTR = D[:, train_index]
            LTR = L[train_index]

            DTE = D[:, test_index]
            LTE = L[test_index]

            min_minDCF.append(train_SVM_linear(DTR, LTR, DTE, LTE))

            print(min_minDCF)
        
        print(f"{app} Full :", min(min_minDCF))



if __name__ == '__main__':

    D, L = load_data()
    
    # Using gaussianized data
    # D = gaussianization(D.T)
    # D = D.T

    # Using Normalization and PCA before
    # D = normalize_data(D)
    # D = calculate_pca(D, 6)
    
    # D = D[:, 0:100]
    # L = L[0:100]

    svm_model(D, L)

