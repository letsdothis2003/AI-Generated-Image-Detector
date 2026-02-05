# Fahim Tanvir and Ahmed Ali
# CSCI-367 Term Project
# Feature extraction for HOG - Adjusted for AI Detection

import numpy as np
from skimage.feature import hog

def compute_hog_features(images):
    n = images.shape[0]
    feats = []

    # Dynamically adjust pixels_per_cell based on image size
    # For AI images (64x64 or 128x128), (8,8) or (16,16) is faster and more robust
    p_per_cell = (8, 8) if images[0].shape[0] >= 64 else (4, 4)

    for i in range(n):
        img = images[i]
        # HOG extraction
        h = hog(img, orientations=9, 
                pixels_per_cell=p_per_cell, 
                cells_per_block=(2, 2), 
                block_norm="L2-Hys", 
                transform_sqrt=True)
        feats.append(h)
    return np.array(feats)

def compute_simple_stats(images):
    """
    Extracts zone-based means and projections. 
    Modified to work with any image size, not just 28x28.
    """
    n = images.shape[0]
    all_feats = []

    for img in images:
        h, w = img.shape
        zones = []
        # Divide image into a 4x4 grid of zones
        for i in range(4):
            for j in range(4):
                r0, r1 = int(i * h / 4), int((i + 1) * h / 4)
                c0, c1 = int(j * w / 4), int((j + 1) * w / 4)
                patch = img[r0:r1, c0:c1]
                zones.append(patch.mean() if patch.size > 0 else 0)

        # Projections (normalized)
        row_sum = img.sum(axis=1) / w
        col_sum = img.sum(axis=0) / h
        
        # Take 16 samples from each projection to keep feature size consistent
        h_proj = np.interp(np.linspace(0, h-1, 16), np.arange(h), row_sum)
        v_proj = np.interp(np.linspace(0, w-1, 16), np.arange(w), col_sum)

        combined = np.concatenate([zones, h_proj, v_proj])
        all_feats.append(combined)

    return np.array(all_feats)

def extract_features(images, use_extra_stats=True):
    print(f"Processing {len(images)} images...")
    hog_f = compute_hog_features(images)
    
    if use_extra_stats:
        stats_f = compute_simple_stats(images)
        return np.hstack([hog_f, stats_f])
    
    return hog_f