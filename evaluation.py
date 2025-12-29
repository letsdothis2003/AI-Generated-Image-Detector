#Fahim Tanvir and Ahmed Ali
#CSCI-367 Term Project

##** This is to evaluate our accuracy and generate confusion matrix
###**
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from model import MnistSvmModel


def k_fold_cv_scores(X,y,n_splits=5,pca_components=50,kernel="rbf",C=5.0,gamma="scale"):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    fold = 1
    for train_idx, val_idx in skf.split(X, y):
        X_train_f = X[train_idx]
        y_train_f = y[train_idx]
        X_val_f = X[val_idx]
        y_val_f = y[val_idx]

        #PCA and SVM hyperparameters
        model = MnistSvmModel(pca_components=pca_components,C=C,gamma=gamma,kernel=kernel)
        model.fit(X_train_f, y_train_f)

        preds = model.predict(X_val_f)
        acc = accuracy_score(y_val_f, preds)
        scores.append(acc)
        print(f"  Fold {fold}: accuracy = {acc:.4f}")
        fold += 1

    scores = np.array(scores)
    print("\nCross validation summary:")
    print(f"  Mean accuracy = {scores.mean():.4f}")
    print(f"  Std  accuracy = {scores.std():.4f}")
    return scores, scores.mean(), scores.std()


def train_final_model(X_train,y_train,pca_components=50,kernel="rbf",C=5.0,gamma="scale"):

    model = MnistSvmModel(pca_components=pca_components,C=C,gamma=gamma,kernel=kernel,)
    model.fit(X_train, y_train)
    return model


def evaluate_on_test(model, X_test, y_test):
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    report_text = classification_report(y_test, preds)

    return {"accuracy": acc,
            "confusion_matrix": cm,
            "report_text": report_text,
            "y_pred": preds
        }


def plot_confusion_matrix(cm, class_names=None, normalize=False, title="Confusion Matrix", save_path=None,):
    if class_names is None:
        class_names = [str(i) for i in range(10)]

    if normalize:
        # normalize rows so each row sums to 1
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        fmt = "d"

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm,annot=True,fmt=fmt,cmap="Blues",xticklabels=class_names,yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved confusion matrix to {save_path}")
    plt.close()
