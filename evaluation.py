# Evaluation logic for AI Detection (Binary Classification)
# 2/14/2025 Update: Had to modify to make room for our new methods, as applit was being a pain. 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from model import MnistSvmModel

def k_fold_cv_scores(X, y, n_splits=5, pca_components=50, kernel="linear", C=1.0, gamma="scale"):
    """
    Performs cross-validation and returns statistical performance.
    Retains support for explicit SVM parameters used in legacy training scripts.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    print(f"Starting {n_splits}-fold Cross-Validation with Kernel={kernel}...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_f, y_train_f = X[train_idx], y[train_idx]
        X_val_f, y_val_f = X[val_idx], y[val_idx]

        # Initialize model with the full parameter set
        model = MnistSvmModel(pca_components=pca_components, C=C, kernel=kernel, gamma=gamma)
        model.fit(X_train_f, y_train_f)

        preds = model.predict(X_val_f)
        acc = accuracy_score(y_val_f, preds)
        scores.append(acc)
        print(f"  Fold {fold} Accuracy: {acc:.4f}")

    scores = np.array(scores)
    return scores, scores.mean(), scores.std()

def train_final_model(X_train, y_train, pca_components=50, kernel="linear", C=1.0, gamma="scale"):
    """
    Trains the final model for production deployment using specific hyperparameters.
    """
    model = MnistSvmModel(pca_components=pca_components, C=C, kernel=kernel, gamma=gamma)
    model.fit(X_train, y_train)
    return model

def evaluate_on_test(model, X_test, y_test):
    """
    Detailed forensic evaluation on the test set.
    Calculates accuracy, confusion matrix, inference latency, and ROC AUC.
    """
    start_time = time.time()

    # 1. Prediction Latency
    y_pred = model.predict(X_test)
    end_time = time.time()
    latency = (end_time - start_time) / len(X_test)

    # 2. Standard Metrics
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # 3. Decision/Probability Scores for ROC AUC
    # Attempts to extract the decision function from the underlying sklearn object
    try:
        # Transform features to the PCA space used by the SVM
        X_scaled = model.scaler.transform(X_test)
        X_pca = model.pca.transform(X_scaled)
        
        if hasattr(model.svm, "decision_function"):
            y_scores = model.svm.decision_function(X_pca)
            auc = roc_auc_score(y_test, y_scores)
        elif hasattr(model.svm, "predict_proba"):
            y_scores = model.svm.predict_proba(X_pca)[:, 1]
            auc = roc_auc_score(y_test, y_scores)
        else:
            auc = None
    except Exception:
        auc = None

    # 4. Classification Report
    report = classification_report(y_test, y_pred, target_names=["REAL", "FAKE"])

    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "latency": latency,
        "auc": auc,
        "report": report
    }

def plot_confusion_matrix(cm, class_names=None, title="Forensic Confusion Matrix", save_path=None):
    """
    Visualizes the detection results with a heatmap.
    """
    if class_names is None:
        class_names = ["REAL", "FAKE"]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
