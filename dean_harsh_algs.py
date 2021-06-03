import os
import csv
import pandas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import norm, skew
from scipy import stats
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV



class Data:
    def __init__(self, X=None, y=None):
        self.X = X
        self.y = y
    def load(self, filename):
        with open(filename, "r") as fid:
            data = np.genfromtxt(fid, delimiter=",")

        # separate features and labels
        self.X = data[:, :-1]
        self.y = data[:, -1]
    def plot(self, **kwargs):
        if "color" not in kwargs:
            kwargs["color"] = "b"

        plt.scatter(self.X, self.y, **kwargs)
        plt.xlabel("x", fontsize=16)
        plt.ylabel("y", fontsize=16)
        plt.show()

def load_data(filename):
    data = Data()
    data.load(filename)
    return data

def plot_data(X, y, **kwargs):
    data = Data(X, y)
    data.plot(**kwargs)


class PolynomialRegression:

    def __init__(self, m=1, reg_param=0):
        # self.coef_ represents the weights of the regression model
        self.coef_ = None
        self.m_ = m
        self.lambda_ = reg_param

    def generate_polynomial_features(self, X):
        n, d = X.shape
        m = self.m_
        if d == (m + 1):
          Phi = X
        else:
          Phi = np.ones_like(X)
          for i in range(1, m+1):
            Phi = np.concatenate((Phi, X**(i-1)), 1)
        return Phi

    def fit_GD(self, X, y, eta = None, eps=0, tmax=10000, verbose = True):

        if self.lambda_ != 0:
            raise Exception("GD with regularization not implemented")

        X = self.generate_polynomial_features(X)  # map features
        n, d = X.shape
        eta_input = eta
        self.coef_ = np.zeros(d)  # coefficients
        err_list = np.zeros((tmax, 1))  # errors per iteration

        for t in range(tmax):

            if eta_input is None:
                #pretty straight forward
                eta = 1/ (1 + float(t))
            else:
                eta = eta_input

            self.coef_ = self.coef_ - (2 * eta * np.dot(np.transpose(X), (self.predict(X) - y )) )
            y_pred = np.dot(X, self.coef_)  # change this line
            err_list[t] = np.sum(np.power(y - y_pred, 2)) / float(n)

            if t > 0 and abs(err_list[t] - err_list[t - 1]) <= eps:
                break

            if verbose :
                x = np.reshape(X[:,1], (n,1))
                cost = self.cost(x,y)
                print ("iteration: %d, cost: %f" % (t+1, cost))

        print("number of iterations: %d" % (t + 1))
        print(self.coef_)

        return self

    def fit(self, X, y, l2regularize=None):

        X = self.generate_polynomial_features(X)  # map features
        self.coef_ = np.dot( np.linalg.pinv(np.dot(np.transpose(X), X )), np.dot(np.transpose(X), y ))
        print(self.coef_)

    def predict(self, X):
        if self.coef_ is None:
            raise Exception("Model not initialized. Perform a fit first.")
        X = self.generate_polynomial_features(X)
        y = np.dot(X, self.coef_)

        return y


    def cost(self, X, y):

        cost = np.sum((self.predict(X) - y)**2)
        return cost

    def rms_error(self, X, y):

        error = np.sqrt( self.cost(X,y) / np.shape(y))
        return error

    def plot_regression(self, xmin=0, xmax=1, n=50, **kwargs):
        if "color" not in kwargs:
            kwargs["color"] = "r"
        if "linestyle" not in kwargs:
            kwargs["linestyle"] = "-"

        X = np.reshape(np.linspace(0, 1, n), (n, 1))
        y = self.predict(X)
        plot_data(X, y, **kwargs)
        plt.show()



def main():
    print("Entered main right now")



##### EDIT HERE ######################### EDIT HERE ######################### EDIT HERE ######################### EDIT HERE ####################
    dir_path = "C:\\Users\\DeanTheBean\\Dean_Main\\UCLA_migrate_201118\\Dean_Transfer\\Work\\Ingage\\"
##### EDIT HERE ######################### EDIT HERE ######################### EDIT HERE ######################### EDIT HERE ####################




    all_data = load_data(os.path.join(dir_path, 'ads_raw.csv'))
    #meta data about the true/false attention data
    fold_num = 5
    test_size = 0.10

    num_1 = int(sum(all_data.y))
    num_0 = all_data.y.shape[0] - num_1
    fold_size_0 = int(num_0*test_size)
    fold_size_1 = int(num_1*test_size)


    #initialize some trackers to track accuracy
    index = []
    accuracy_track_lr = []
    accuracy_track_svm = []
    accuracy_track_gauss = []
    accuracy_track_KNN = []
    accuracy_track_ridge = []
    accuracy_track_GBC = []
    accuracy_track_rfc_g = []



    #Basically Hard Coded K-Folds validiation. If using sklean's KFold, no need to use this!
    for fold in range(fold_num):
        index.append(fold)

        test_data = all_data.X[fold* fold_size_0 : (fold + 1)* (fold_size_0), :] #part 1
        test_data = np.concatenate((test_data, all_data.X[num_0 + fold*fold_size_1 : num_0 + (fold+1)*fold_size_1, :]), 0) #part 2
        test_data_y = all_data.y[fold* fold_size_0 :(fold + 1)* (fold_size_0)]
        test_data_y = np.concatenate((test_data_y, all_data.y[num_0 + fold*fold_size_1 : num_0 + (fold+1)*fold_size_1]), 0)


        #come up with an algorithm that basically does !(test), then train to get rid of all this...
        train_data = all_data.X[0 : fold * fold_size_0, :] #part 1
        train_data = np.concatenate((train_data, all_data.X[(fold +1)*fold_size_0 : num_0 +(fold +1)*fold_size_0 , : ]), 0) #part 2
        train_data = np.concatenate((train_data, all_data.X[num_0 +(fold +1)*fold_size_0 + fold_size_1: num_0 + num_1, : ]), 0) #part 3
        train_data_y = all_data.y[0 : fold * fold_size_0] #p1
        train_data_y = np.concatenate((train_data_y, all_data.y[(fold +1)*fold_size_0 : num_0 +(fold +1)*fold_size_0]), 0) #part 2
        train_data_y = np.concatenate((train_data_y, all_data.y[num_0 +(fold +1)*fold_size_0 + fold_size_1: num_0 + num_1]), 0) #part 3

        test_data = Data(test_data, test_data_y)
        train_data = Data(train_data, train_data_y)

        #LOGISTIC REGRESSION CLF
        y_pred_lr = LogisticRegression(solver = 'lbfgs').fit(train_data.X, train_data.y).predict(test_data.X)

        #RBF KERNEL SVM CLF
        svc_rbf = SVC(kernel = 'rbf', random_state = 0)
        y_pred_rbf = svc_rbf.fit(train_data.X, train_data.y).predict(test_data.X)

        #GAUSS NB CLF
        gauss_clf = GaussianNB()
        y_pred_gauss = gauss_clf.fit(train_data.X, train_data.y).predict(test_data.X)

        # KNN
        KNN_clf = KNeighborsClassifier(n_neighbors = 71, metric = 'minkowski', p = 2)
        y_pred_KNN = KNN_clf.fit(train_data.X, train_data.y).predict(test_data.X)

        #Ridge Classifier
        ridge_clf = RidgeClassifier(alpha=100,tol=0.001)
        y_pred_ridge = ridge_clf.fit(train_data.X, train_data.y).predict(test_data.X)

        # GradientBoostingClassifier
        gbc_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.005, max_depth=5, random_state=0)
        y_pred_gbc = gbc_clf.fit(train_data.X, train_data.y).predict(test_data.X)

        # Random Forest Classifier with Grid Search
        parameters = {'n_estimators':[100,150,200,250,300], 'max_depth':[10,15,20,25]}
        forest = RandomForestClassifier()
        rfg_clf = GridSearchCV(estimator=forest, param_grid=parameters, n_jobs=-1, cv=5, verbose=2)
        rfg_clf.fit(train_data.X, train_data.y)
        rfc_clf_best = RandomForestClassifier(max_depth=rfg_clf.best_params_['max_depth'], random_state=0, n_estimators=rfg_clf.best_params_['n_estimators'])
        y_pred_rfc_grid = rfc_clf_best.fit(train_data.X, train_data.y).predict(test_data.X)


        #append accuracy
        accuracy_track_lr.append(accuracy_score(test_data.y, y_pred_lr))
        accuracy_track_svm.append(accuracy_score(test_data.y, y_pred_rbf))
        accuracy_track_gauss.append(accuracy_score(test_data.y, y_pred_gauss))
        accuracy_track_KNN.append(accuracy_score(test_data.y, y_pred_KNN))
        accuracy_track_ridge.append(accuracy_score(test_data.y, y_pred_ridge))
        accuracy_track_GBC.append(accuracy_score(test_data.y, y_pred_gbc))
        accuracy_track_rfc_g.append(accuracy_score(test_data.y, y_pred_rfc_grid))



#HAVE YET TO DO SEQUENTIAL AND STACKING CLASSIFIER






    #ymcrgbwk
    plt.plot(index, accuracy_track_lr,      'y', label="Accuracy Track")
    plt.plot(index, accuracy_track_svm,     'm', label="Accuracy Track")
    plt.plot(index, accuracy_track_gauss,   'c', label="Accuracy Track")
    plt.plot(index, accuracy_track_KNN,     'r', label="Accuracy Track")
    plt.plot(index, accuracy_track_ridge,   'g', label="Accuracy Track")
    plt.plot(index, accuracy_track_GBC,     'b', label="Accuracy Track")
    plt.plot(index, accuracy_track_rfc_g,   'k', label="Accuracy Track")




    plt.title('Accuracy Across '+str(fold_num)+' Folds')
    plt.xlabel("Fold #")
    plt.ylabel("Accuracy")
    labelsh = ["Log Regression", "SVM RBF", "Gaussian Naive Beyes", "KNN", "Ridge", "GBC", "RandomForest_grid"]
    plt.legend(labelsh)
    plt.show()


    print("Done with Main!")

if __name__ == "__main__":
    main()
