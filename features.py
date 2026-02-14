
# Feature extraction for HOG 
# 2/10/2026 Update: Implemented  LBP + GLCM Hybrid + YCbCr logic to ensure better accuracy in detection.

import numpy as np
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage.color import rgb2ycbcr
from PIL import Image

def extract_spatial_domain_features(img):
    """
    Analyzes spatial relationships and statistical distributions.
    AI often lacks the natural randomness of camera sensor noise.
    """
    # Ensure image is in 0-255 range for GLCM
    img_uint = (img * 255).astype(np.uint8)
    
    # Gray-Level Co-occurrence Matrix (GLCM)
    # Captures spatial dependency of pixel intensities
    # AI images often have lower 'Entropy' or higher 'Homogeneity' in specific patterns
    distances = [1, 2]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    # Generate the GLCM
    glcm = graycomatrix(img_uint, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    
    # Extract properties
    contrast = graycoprops(glcm, 'contrast').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    
    # High-Frequency Noise Analysis
    # Calculates the 'Spatial Gradient Variance'
    # AI images tend to have smoother high-frequency regions compared to real noise
    dy, dx = np.gradient(img)
    grad_variance = [np.var(dx), np.var(dy)]
    
    return np.concatenate([contrast, correlation, energy, homogeneity, grad_variance])

def extract_features(images_np, use_extra_stats=True):
    """
    Comprehensive Hybrid Pipeline:
    1. Structural: HOG (Edges/Shapes)
    2. Texture: LBP (Micro-patterns/Smoothness)
    3. Spatial: GLCM (Pixel correlations)
    4. Domain: Spatial Error/Noise Variance
    """
    feature_list = []
    
    # LBP Parameters
    radius = 3
    n_points = 8 * radius
    
    for img in images_np:
        # A. HOG Features
        fd = hog(img, orientations=9, pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2), visualize=False)
        
        if not use_extra_stats:
            feature_list.append(fd)
            continue
            
        # B. LBP (Texture regularity)
        lbp = local_binary_pattern(img, n_points, radius, method='uniform')
        (lbp_hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-7)
        
        # C. Spatial Domain Analysis (GLCM & Noise)
        spatial_feats = extract_spatial_domain_features(img)
        
        # D. Spatial Zoning (Global distribution)
        # Slices the image into a 4x4 grid and gets mean/std of each zone
        h, w = img.shape
        grid_size = 4
        zoning_stats = []
        for i in range(grid_size):
            for j in range(grid_size):
                region = img[i*h//grid_size:(i+1)*h//grid_size, j*w//grid_size:(j+1)*w//grid_size]
                zoning_stats.append(np.mean(region))
                zoning_stats.append(np.std(region))

        # Merge all into one high-dimensional vector
        combined = np.concatenate([fd, lbp_hist, spatial_feats, zoning_stats])
        feature_list.append(combined)
        
    return np.array(feature_list)
    return np.array(feature_list)
