#2/14/2026 This should comtain our per
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
import joblib

class MnistSvmModel:
    def __init__(self, pca_components=50, C=1.0, gamma="scale", kernel="linear"):
        """
        SVM implementation optimized for AI detection artifacts using Gradient-based features (HOG).
        Uses StandardScaler for faster convergence and LinearSVC for optimized inference speed.
        """
        # StandardScaler is vital when using HOG/Gradients. It ensures that high-magnitude gradient artifacts don't overwhelm smaller but more discriminative texture features.
        self.scaler = StandardScaler()
        self.pca_components = pca_components
        self.pca = PCA(n_components=pca_components)
        
        if kernel == "linear":
            #For finding the hyperplane between  the clean gradients of real images and the noisy gradients of AI.
            # dual=False is preferred when n_samples > n_features.
            self.svm = LinearSVC(C=C, dual=False, max_iter=5000, tol=1e-4)
        else:
            # Fallback for non-linear kernels like RBF.
            self.svm = SVC(C=C, gamma=gamma, kernel=kernel)
            
        self.is_fitted = False
        self.feature_dim = None
        self.expected_input_dim = None

    def fit(self, X_train, y_train):
        """
        Trains the model by scaling gradient data, reducing dimensionality 
        via PCA, and fitting the SVM.
        """
        self.expected_input_dim = X_train.shape[1]
        
        # Step 1. Scale features: Normalizing the gradient distributions.
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Step 2. Reduce dimensionality: Condenses HOG vectors to capture the 
        # most significant "gradient signatures" of AI vs Real images.
        # Dynamic adjustment check to ensure components don't exceed samples.
        n_comp = min(self.pca_components, X_train.shape[0], X_train.shape[1])
        if n_comp != self.pca_components:
            self.pca = PCA(n_components=n_comp)

        X_pca = self.pca.fit_transform(X_scaled)
        self.feature_dim = X_pca.shape[1]
        
        # Step 3. Fit classifier: Learns the boundary based on gradient patterns.
        self.svm.fit(X_pca, y_train)
        self.is_fitted = True

    def predict(self, X):
        """
        Transforms new images using the learned gradient scale and PCA
        space to predict if they are AI-generated. Includes dimension guarding.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
            
        if X.shape[1] != self.expected_input_dim:
            raise ValueError(f"Feature mismatch: Expected {self.expected_input_dim} features, got {X.shape[1]}.")

        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return self.svm.predict(X_pca)

    def evaluate(self, X, y):
        """
        Directly returns the mean accuracy of the model on the provided 
        feature set and labels.
        """
        preds = self.predict(X)
        return np.mean(preds == y)

    def save(self, filename):
        """Saves the entire pipeline (Scaler, PCA, SVM) to a file."""
        joblib.dump(self, filename)

    @staticmethod
    def load(filename):
        """Loads a previously saved MnistSvmModel instance."""
        return joblib.load(filename)
