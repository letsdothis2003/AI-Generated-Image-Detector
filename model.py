#Fahim Tanvir and Ahmed Ali
#CSCI-367 Term Project
#For our SVM implementation


import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC


class MnistSvmModel:
    def __init__(self, pca_components=50, C=5.0, gamma="scale", kernel="rbf"):
        self.pca = PCA(n_components=pca_components)

        self.svm = SVC(C=C, gamma=gamma, kernel=kernel)
        self.feature_dim = None

    def fit(self, X_train, y_train):

        X_pca = self.pca.fit_transform(X_train)
        self.feature_dim = X_pca.shape[1]
        self.svm.fit(X_pca, y_train)

    def predict(self, X):
        X_pca = self.pca.transform(X)
        return self.svm.predict(X_pca)

    def evaluate(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)
