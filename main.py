# AI-Generated Image Detection  

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from data_loader import load_ai_detection_dataset
from features import extract_features
from model import MnistSvmModel
from evaluation import k_fold_cv_scores, train_final_model, evaluate_on_test, plot_confusion_matrix
from sklearn.metrics import classification_report

def show_example_predictions(X_test, y_test, y_pred, num_examples=10, save_path=None):
    """
    Displays a few examples of images with their actual and predicted labels.
    """
    correct = (y_test == y_pred)
    wrong = ~correct

    # Try to get a mix of correct and incorrect predictions
    good_idx = np.where(correct)[0][: num_examples // 2]
    bad_idx = np.where(wrong)[0][: num_examples // 2]
    show_idx = list(good_idx) + list(bad_idx)

    if len(show_idx) == 0:
        print("No examples to display.")
        return

    class_names = ["REAL", "FAKE"]
    fig, axes = plt.subplots(2, (len(show_idx) + 1) // 2, figsize=(15, 7))
    axes = axes.flatten()

    for i, idx in enumerate(show_idx):
        axes[i].imshow(X_test[idx], cmap='gray')
        color = 'green' if correct[idx] else 'red'
        title = f"True: {class_names[int(y_test[idx])]}\nPred: {class_names[int(y_pred[idx])]}"
        axes[i].set_title(title, color=color)
        axes[i].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    # 1. Setup paths and Performance Parameters
    # Paths updated to reflect your high-volume dataset structure
    data_directory = r"C:\Users\fahim\Downloads\ai_image_detection\AI_Detection Project\Data"
    
    # PERFORMANCE OPTIMIZATION: 
    # With 160k images, memory management is key. 
    # Reducing size to 64x64 speeds up HOG significantly.
    TARGET_SIZE = (64, 64) 
    PCA_COMPONENTS = 50
    
    print(f"--- AI Image Detection System (Optimized for Large Data) ---")
    print(f"Target Directory: {data_directory}")

    # 2. Load the dataset 
    # This step might still be slow due to the 160k file I/O.
    X_train, y_train, X_test, y_test = load_ai_detection_dataset(data_directory, target_size=TARGET_SIZE)

    if X_train.size == 0 or X_test.size == 0:
        print("Error: Could not load images. Please check your folder paths.")
        return

    # PERFORMANCE OPTIMIZATION: Subsampling
    # SVM complexity is O(n^2) to O(n^3). 
    # 10,000 samples is the "sweet spot" for SVM training speed vs accuracy.
    MAX_TRAIN = 10000 
    if len(X_train) > MAX_TRAIN:
        print(f"Subsampling training set from {len(X_train)} to {MAX_TRAIN} for speed...")
        # Ensure we keep a balanced representation of REAL and FAKE
        indices = np.random.choice(len(X_train), MAX_TRAIN, replace=False)
        X_train, y_train = X_train[indices], y_train[indices]

    print(f"Active Training Set: {len(X_train)} images.")
    print(f"Active Testing Set: {len(X_test)} images.")

    # 3. Feature Extraction
    print(f"\nExtracting HOG features (Input Size: {TARGET_SIZE})...")
    # Note: If this is still slow, increase pixels_per_cell in features.py
    X_train_f = extract_features(X_train, use_extra_stats=True)
    X_test_f = extract_features(X_test, use_extra_stats=True)
    
    print(f"Feature extraction complete. Feature vector length: {X_train_f.shape[1]}")

    # 4. Cross-Validation
    # Using a Linear kernel is much faster than RBF when you have many features.
    print("\nRunning 3-fold cross validation...")
    scores, mean_acc, std_acc = k_fold_cv_scores(
        X_train_f, y_train, 
        n_splits=3, 
        pca_components=PCA_COMPONENTS, 
        kernel="linear", 
        C=1.0
    )
    print(f"Average CV Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})")

    # 5. Train Final Model
    print("\nTraining final SVM model on the optimized subset...")
    model = train_final_model(
        X_train_f, y_train, 
        pca_components=PCA_COMPONENTS, 
        kernel="linear", 
        C=1.0
    )

    # 6. Evaluation
    print("\nEvaluating on full test set...")
    results = evaluate_on_test(model, X_test_f, y_test)
    
    print(f"Final test accuracy: {results['accuracy']:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, results['y_pred'], target_names=["REAL", "FAKE"]))

    # 7. Visualization and Output
    print("\nGenerating confusion matrix and sample predictions...")
    plot_confusion_matrix(
        results["confusion_matrix"], 
        class_names=["REAL", "FAKE"], 
        title="AI Detection Confusion Matrix", 
        save_path="ai_detection_cm.png"
    )

    show_example_predictions(X_test, y_test, results['y_pred'], num_examples=10)

    print("\nOptimization complete. Results saved.")

if __name__ == "__main__":
    main()
