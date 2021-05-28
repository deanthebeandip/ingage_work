import os
import csv
import pandas
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

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
    dir_path = "C:\\Users\\DeanTheBean\\Dean_Main\\UCLA_migrate_201118\\Dean_Transfer\\Work\\Ingage\\log_regression\\"
##### EDIT HERE ######################### EDIT HERE ######################### EDIT HERE ######################### EDIT HERE ####################




    all_data = load_data(os.path.join(dir_path, 'ads_raw.csv'))
    #meta data about the true/false attention data
    fold_num = 10

    num_1 = int(sum(all_data.y))
    num_0 = all_data.y.shape[0] - num_1
    fold_size_0 = int(num_0/fold_num)
    fold_size_1 = int(num_1/fold_num)


    #initialize some trackers to track accuracy
    index = []
    accuracy_track = []

    #START HERE FOR THE FOR LOOPS
    for fold in range(fold_num):
        index.append(fold)

        test_data = all_data.X[fold* fold_size_0 : (fold + 1)* (fold_size_0), :] #part 1
        test_data = np.concatenate((test_data, all_data.X[num_0 + fold*fold_size_1 : num_0 + (fold+1)*fold_size_1, :]), 0) #part 2
        test_data_y = all_data.y[fold* fold_size_0 :(fold + 1)* (fold_size_0)]
        test_data_y = np.concatenate((test_data_y, all_data.y[num_0 + fold*fold_size_1 : num_0 + (fold+1)*fold_size_1]), 0)

        train_data = all_data.X[0 : fold * fold_size_0, :] #part 1
        train_data = np.concatenate((train_data, all_data.X[(fold +1)*fold_size_0 : num_0 +(fold +1)*fold_size_0 , : ]), 0) #part 2
        train_data = np.concatenate((train_data, all_data.X[num_0 +(fold +1)*fold_size_0 + fold_size_1: num_0 + num_1, : ]), 0) #part 3
        train_data_y = all_data.y[0 : fold * fold_size_0] #p1
        train_data_y = np.concatenate((train_data_y, all_data.y[(fold +1)*fold_size_0 : num_0 +(fold +1)*fold_size_0]), 0) #part 2
        train_data_y = np.concatenate((train_data_y, all_data.y[num_0 +(fold +1)*fold_size_0 + fold_size_1: num_0 + num_1]), 0) #part 3

        test_data = Data(test_data, test_data_y)
        train_data = Data(train_data, train_data_y)
        print("Finished loading data")

        #create classifier
        clf = LogisticRegression(solver = 'lbfgs')
        #train the classifier
        clf.fit(train_data.X, train_data.y)

        # m coef & b intercept
        print(clf.intercept_)
        print(clf.coef_)

        # accuracy
        print(clf.score(train_data.X, train_data.y))

        #predict
        print("Predict Gang")
        print(clf.predict(test_data.X))
        print(clf.predict_proba(test_data.X))

        accuracy_track.append( clf.score(test_data.X, test_data.y))



    plt.plot(index, accuracy_track, 'r', label="Accuracy Track")
    plt.title('Accuracy Across 10 Folds')
    plt.xlabel("Fold #")
    plt.ylabel("Accuracy")
    labelsh = ["Accuracy"]
    plt.legend(labelsh)
    plt.show()


    print("Done with Main!")

if __name__ == "__main__":
    main()
