#Fahim Tanvir and Ahmed Ali
# Please make sure all dependencies are available using this:
#pip install scikit-image
#pip install seaborn
#pip install matplotlib

##** We used this datatset:
#  https://www.kaggle.com/datasets/hojjatk/mnist-dataset/data
# 
# And go to data_loader.py if datatset is in a directory that isn't optimal
# ##**

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from data_loader import load_mnist_dataset
from features import extract_features
from model import MnistSvmModel
from evaluation import k_fold_cv_scores,train_final_model,evaluate_on_test,plot_confusion_matrix

def show_example_digits(X_test, y_test, y_pred, num_examples=10, save_path=None):
    correct = (y_test == y_pred)
    wrong = ~correct

    good_idx = np.where(correct)[0][: num_examples // 2]
    bad_idx = np.where(wrong)[0][: num_examples // 2]
    show_idx = list(good_idx) + list(bad_idx)

    if len(show_idx) == 0:
        print("No examples to display.")
        return

    fig, axes = plt.subplots(2, num_examples // 2, figsize=(10, 4))
    axes = axes.flatten()

    for ax, idx in zip(axes, show_idx):
        ax.imshow(X_test[idx], cmap="gray")
        t = y_test[idx]
        p = y_pred[idx]
        ax.set_title(f"{t} → {p}", color=("green" if t == p else "red"))
        ax.axis("off")

    plt.suptitle("Example test digits (green = correct, red = wrong)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved examples to {save_path}")

    plt.close()


def main():
    data_dir = Path("data")

    print("Loading MNIST dataset.")
    X_train, y_train, X_test, y_test = load_mnist_dataset(data_dir)
    print(f"Training images: {X_train.shape[0]}")
    print(f"Test images: {X_test.shape[0]}")

    subset_size = 10000
    print(f"\nUsing a small subset of {subset_size} training images for quick experiments.")
    X_small = X_train[:subset_size]
    y_small = y_train[:subset_size]

    #HOG only vs HOG + stats
    print("\nAblation study: HOG only vs HOG + extra stats (on the subset)")

    print("Extracting HOG-only features for the subset.")
    X_small_hog = extract_features(X_small, use_extra_stats=False)
    scores_hog, mean_hog, std_hog = k_fold_cv_scores(X_small_hog,y_small,n_splits=3,pca_components=50,kernel="rbf",C=5.0,gamma="scale")

    print(f"HOG-only (subset, 3-fold) accuracy: {mean_hog:.4f} (std {std_hog:.4f})")

    print("\nExtracting HOG + stats features for the subset.")
    X_small_full = extract_features(X_small, use_extra_stats=True)
    scores_full, mean_full, std_full = k_fold_cv_scores(X_small_full,y_small,n_splits=3,pca_components=50,kernel="rbf",C=5.0,gamma="scale")

    print(f"HOG + stats (subset, 3-fold) accuracy: {mean_full:.4f} (std {std_full:.4f})")


    print("\nSmall experiment: different C values for SVM (on the subset, HOG + stats)")
    C_values = [1.0, 5.0, 10.0]
    for C in C_values:
        print(f"\nTrying C = {C}")
        scores_c, mean_c, std_c = k_fold_cv_scores(X_small_full,y_small,n_splits=3,pca_components=50,kernel="rbf",C=C,gamma="scale")

        print(f"Subset accuracy with C={C}: {mean_c:.4f} (std {std_c:.4f})")

    print("\nExtracting features on the full training and test sets (HOG + stats).")
    X_train_f = extract_features(X_train, use_extra_stats=True)
    X_test_f = extract_features(X_test, use_extra_stats=True)
    print("Feature extraction done.")
    print(f"Train features shape: {X_train_f.shape}")
    print(f"Test features shape: {X_test_f.shape}")

    print("\nRunning 5-fold cross validation on the full training set.")
    scores, mean_acc, std_acc = k_fold_cv_scores(X_train_f,y_train,n_splits=5,pca_components=50,kernel="rbf",C=5.0,gamma="scale")
    print(f"Cross-validation accuracy: {mean_acc:.4f} (std {std_acc:.4f})")

    print("\nTraining final model on full training set.")
    model = train_final_model(X_train_f,y_train,pca_components=50,kernel="rbf",C=5.0,gamma="scale")

    print("\nEvaluating on test set.")
    results = evaluate_on_test(model, X_test_f, y_test)
    print(f"Final test accuracy: {results['accuracy']:.4f}")
    print("\nClassification report:")
    print(results["report_text"])

    print("\nGenerating confusion matrices.")
    plot_confusion_matrix(results["confusion_matrix"],title="Confusion Matrix",save_path="confusion_matrix.png")
    plot_confusion_matrix(results["confusion_matrix"],normalize=True,title="Normalized Confusion Matrix",save_path="confusion_matrix_normalized.png")

    print("\nSaving example predictions.")
    show_example_digits(X_test,y_test,results["y_pred"],save_path="example_digits.png")

    print("\nDone. Images and results saved in the project folder.")


if __name__ == "__main__":
    main()

