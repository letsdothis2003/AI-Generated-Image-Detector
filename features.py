#Fahim Tanvir and Ahmed Ali
#For feature extraction for HOG
import numpy as np
from skimage.feature import hog

def compute_hog_features(images):
    n = images.shape[0]
    feats = []

    for i in range(n):
        img = images[i]

        #hog parameters chosen so the feature size is reasonable for mnist
        h = hog(img,orientations=9,pixels_per_cell=(4, 4),cells_per_block=(2, 2),block_norm="L2-Hys",transform_sqrt=True)
        feats.append(h)
    return np.array(feats)


def compute_simple_stats(images):
    n = images.shape[0]
    all_feats = []

    zone_size = 7  #each zone is roughly 7x7
    proj_bins = 16
    step = 28 / proj_bins

    for img in images:
        zones = []
        for i in range(4):
            for j in range(4):
                r0 = int(i * zone_size)
                r1 = int((i + 1) * zone_size)
                c0 = int(j * zone_size)
                c1 = int((j + 1) * zone_size)
                patch = img[r0:r1, c0:c1]
                zones.append(patch.mean())

        #horizontal projection
        row_sum = img.sum(axis=1)
        h_proj = []
        for k in range(proj_bins):
            a = int(k * step)
            b = int((k + 1) * step)
            if a == b:
                b = a + 1
            h_proj.append(row_sum[a:b].mean())

        #vertical projection: 
        col_sum = img.sum(axis=0)
        v_proj = []
        for k in range(proj_bins):
            a = int(k * step)
            b = int((k + 1) * step)
            if a == b:
                b = a + 1
            v_proj.append(col_sum[a:b].mean())

        feats = zones + h_proj + v_proj
        all_feats.append(feats)

    return np.array(all_feats)


def extract_features(images, use_extra_stats=True):

    hog_feats = compute_hog_features(images)

    if use_extra_stats:
        stats_feats = compute_simple_stats(images)
        #hog features and statsn concatenation
        return np.concatenate([hog_feats, stats_feats], axis=1)
    else:
        return hog_feats

