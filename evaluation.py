# Evaluation logic for AI Detection (Binary Classification)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from model import MnistSvmModel

def k_fold_cv_scores(X, y, n_splits=3, pca_components=50, kernel="linear", C=1.0, gamma="scale"):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_f, y_train_f = X[train_idx], y[train_idx]
        X_val_f, y_val_f = X[val_idx], y[val_idx]

        # Pass both kernel and gamma to the model
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
    Trains the final model. Added 'gamma' support to resolve TypeError.
    """
    model = MnistSvmModel(pca_components=pca_components, C=C, kernel=kernel, gamma=gamma)
    model.fit(X_train, y_train)
    return model

def evaluate_on_test(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "y_pred": preds
    }

def plot_confusion_matrix(cm, class_names=None, title="Confusion Matrix", save_path=None):
    if class_names is None:
        class_names = ["REAL", "FAKE"]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()