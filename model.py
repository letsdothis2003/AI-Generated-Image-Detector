
# For our SVM implementation  

import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler

class MnistSvmModel:
    def __init__(self, pca_components=50, C=1.0, gamma="scale", kernel="linear"):
        """
        Optimized for AI Detection using Gradient-based features (HOG).
        Uses StandardScaler for faster convergence and LinearSVC for speed.
        """
        # StandardScaler is vital when using HOG/Gradients.
        # It ensures that high-magnitude gradient artifacts don't 
        # overwhelm smaller but more discriminative texture features.
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_components)
        
        if kernel == "linear":
            # LinearSVC is effective for finding the hyperplane between 
            # the clean gradients of real images and the noisy gradients of AI.
            # dual=False is preferred when n_samples > n_features.
            self.svm = LinearSVC(C=C, dual=False, max_iter=5000, tol=1e-4)
        else:
            # Fallback for non-linear, though linear is recommended for speed at scale.
            self.svm = SVC(C=C, gamma=gamma, kernel=kernel)
            
        self.feature_dim = None

    def fit(self, X_train, y_train):
        """
        Trains the model by scaling gradient data, reducing dimensionality 
        via PCA, and fitting the SVM.
        """
        # 1. Scale features: Normalizing the gradient distributions.
        X_scaled = self.scaler.fit_transform(X_train)
        
        # 2. Reduce dimensionality: Condenses HOG vectors to capture the 
        # most significant "gradient signatures" of AI vs Real images.
        X_pca = self.pca.fit_transform(X_scaled)
        self.feature_dim = X_pca.shape[1]
        
        # 3. Fit classifier: Learns the boundary based on gradient patterns.
        self.svm.fit(X_pca, y_train)

    def predict(self, X):
        """
        Transforms new images using the learned gradient scale and PCA
        space to predict if they are AI-generated.
        """
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return self.svm.predict(X_pca)

    def evaluate(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)
