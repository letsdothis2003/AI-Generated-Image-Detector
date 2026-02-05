
# Contains our paths for our data used in training

import os
from pathlib import Path
import numpy as np
from PIL import Image

def load_images_from_folder(folder_path, target_size=(64, 64)):
    folder_path = Path(folder_path)
    images = []
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    if not folder_path.exists():
        return np.array([])

    files = [f for f in folder_path.iterdir() if f.suffix.lower() in valid_extensions]
    
    for img_path in files:
        try:
            with Image.open(img_path) as img:
                if img.mode in ("P", "RGBA"):
                    img = img.convert("RGB")
                img = img.convert('L') 
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                images.append(np.array(img))
        except Exception:
            pass
    return np.array(images)

def load_ai_detection_dataset(base_dir, target_size=(64, 64)):
    """
    Loads dataset using relative paths.
    Expects structure: base_dir/train/REAL, base_dir/train/FAKE, etc.
    """
    base_path = Path(base_dir)
    
    # Helper to find paths regardless of exact casing in folder names
    def get_path(parent, sub):
        p = parent / sub
        if not p.exists():
            # Try lowercase fallback
            p = parent / sub.lower()
        return p

    train_path = get_path(base_path, "train")
    test_path = get_path(base_path, "test")

    train_real_path = get_path(train_path, "REAL")
    train_fake_path = get_path(train_path, "FAKE")
    test_real_path = get_path(test_path, "REAL")
    test_fake_path = get_path(test_path, "FAKE")

    train_real = load_images_from_folder(train_real_path, target_size)
    train_fake = load_images_from_folder(train_fake_path, target_size)
    test_real = load_images_from_folder(test_real_path, target_size)
    test_fake = load_images_from_folder(test_fake_path, target_size)

    X_train = np.array([])
    y_train = np.array([])
    if len(train_real) > 0 or len(train_fake) > 0:
        X_train = np.concatenate([train_real, train_fake], axis=0) if len(train_real) and len(train_fake) else (train_real if len(train_real) else train_fake)
        y_train = np.concatenate([np.zeros(len(train_real)), np.ones(len(train_fake))])

    X_test = np.array([])
    y_test = np.array([])
    if len(test_real) > 0 or len(test_fake) > 0:
        X_test = np.concatenate([test_real, test_fake], axis=0) if len(test_real) and len(test_fake) else (test_real if len(test_real) else test_fake)
        y_test = np.concatenate([np.zeros(len(test_real)), np.ones(len(test_fake))])

    if X_train.size > 0: X_train = X_train.astype(np.float32) / 255.0
    if X_test.size > 0: X_test = X_test.astype(np.float32) / 255.0

    return X_train, y_train, X_test, y_test
