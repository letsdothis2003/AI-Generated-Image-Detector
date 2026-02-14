# AI-Generated Image Detection System - Final Interactive UI
# AI-Generated Image Detection System - Final Interactive UI

import streamlit as st
import numpy as np
from PIL import Image
@@ -8,22 +10,33 @@
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from skimage.feature import hog
from skimage import exposure

#For Streamlit Cloud / Environments  
# Ensures the current directory is in the python path so local imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import backend scripts
try:
    from data_loader import load_ai_detection_dataset
    from features import extract_features
    from model import MnistSvmModel
    from evaluation import evaluate_on_test
except ImportError as e:
    st.error(f"Critical Error: Could not find required project files. {e}")
    st.stop()

# Page Configuration 
st.set_page_config(page_title="AI Image Detector", page_icon="🔍", layout="wide")

# File where the trained model will be stored
MODEL_FILE = "ai_detection_model.joblib"

# Session State Initialization 
def initialize_state():
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
@@ -41,9 +54,10 @@ def initialize_state():
# Auto-load existing model
@st.cache_resource
def load_stored_model():
    model_path = Path(current_dir) / MODEL_FILE
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception:
            return None
    return None
@@ -57,15 +71,15 @@ def load_stored_model():
Inspired by how people be using AI to impersonate or make illegal images. 
""")

# Sidebar: Configuration & Training  
with st.sidebar:
    st.header("Model Management")
    default_path = os.path.join(current_dir, "Data")
    data_path = st.text_input("Dataset Directory", default_path)

    st.subheader("Training Parameters")
    target_size = st.slider("Image Resize Dimension", 32, 128, st.session_state.target_size)
    pca_comps = st.slider("PCA Components", 10, 200, 120)

    if st.button("Train and Save Model"):
        start_time = time.time()
@@ -75,19 +89,21 @@ def load_stored_model():
            X_train, y_train, X_test, y_test = load_ai_detection_dataset(data_path, target_size=(target_size, target_size))

            if X_train is None or X_train.size == 0:
                st.error("Dataset not found at specified path. Please ensure 'Data' folder exists.")
            else:
                progress_bar.progress(20)
                st.write("Extracting Hybrid Features (HOG + LBP + GLCM)...")
                X_train_f = extract_features(X_train, use_extra_stats=True)
                X_test_f = extract_features(X_test, use_extra_stats=True)
                progress_bar.progress(50)

                st.write("Fitting Optimized SVM Classifier...")
                # Using RBF Kernel and C=15.0 for high-performance detection
                model = MnistSvmModel(pca_components=pca_comps, kernel="rbf", C=15.0)
                model.fit(X_train_f, y_train)
                
                model_path = Path(current_dir) / MODEL_FILE
                joblib.dump(model, model_path)
                progress_bar.progress(80)

                st.write("Performing Evaluation...")
@@ -105,8 +121,9 @@ def load_stored_model():
    st.text(f"Model Status: {'Loaded' if st.session_state.trained_model else 'Not Found'}")
    if st.session_state.trained_model:
        st.text(f"Resolution: {st.session_state.target_size}x{st.session_state.target_size}")
        st.text(f"PCA Comps: {st.session_state.trained_model.pca.n_components}")

# Utility Functions for Main UI  
@st.cache_data
def get_hog_viz(image_np):
    """Generates a visualization of the HOG descriptors."""
@@ -115,12 +132,12 @@ def get_hog_viz(image_np):
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return hog_image_rescaled

# Main Interface  
tab1, tab2, tab3 = st.tabs(["Batch Detection", "Model Performance", "Technical Analysis"])

with tab1:
    st.header("Upload Analysis Subject")
    uploaded_files = st.file_uploader("Select images for analysis", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)

    if st.session_state.trained_model is None:
        st.warning("The detection engine is not ready. Please train the model via the sidebar.")
@@ -146,7 +163,6 @@ def get_hog_viz(image_np):

            # Estimate confidence using decision function
            try:

                processed_feat = st.session_state.trained_model.pca.transform(
                    st.session_state.trained_model.scaler.transform(feat)
                )
@@ -202,7 +218,7 @@ def get_hog_viz(image_np):
        m1, m2 = st.columns(2)
        accuracy_pct = st.session_state.accuracy * 100
        m1.metric("Overall System Accuracy", f"{accuracy_pct:.2f}%")
        m2.metric("Classification Type", "Binary (SVM-RBF)")

        st.divider()

@@ -227,55 +243,55 @@ def get_hog_viz(image_np):

with tab3:
    st.header("Methodology")
    st.write("This project utilizes a multi-domain forensic approach:")

    col_a, col_b, col_c, col_d, col_e = st.columns(5)

    with col_a:
        st.subheader("Feature Extraction")
        st.info("""
        **HOG:** Looks at the edges and shapes. 
        AI images often leave watermarks and regularity in gradients that are invisible to the eye but clear to HOG descriptors.
        """)

    with col_b:
        st.subheader("Dimension Reduction")
        current_pca = st.session_state.trained_model.pca.n_components if st.session_state.trained_model else pca_comps

        st.info(f"""
        **PCA:**
        The raw HOG/LBP output is huge. We use PCA to reduce data down to the top {current_pca} components. 
        This removes noise and helps the SVM find the boundary.
        """)

    with col_c:
        st.subheader("Texture Analysis")
        st.info("""
        **LBP:**
        Scans pixel-level textures. AI imagery frequently has uncanny 'smoothness' or repetitive textures compared to natural photographic grain. 
        """)

    with col_d:
        st.subheader("Spatial Domain")
        st.info("""
        **GLCM Analysis:**
        Examines pixel-to-pixel relationships. AI struggles to replicate the stochastic randomness of real-world light scattering. 
        """)

    with col_e:
        st.subheader("Classification")
        st.info("""
        **SVM (RBF):**
        Using a penalty C-value (15.0), it maps non-linear artifacts into a clear 'Real' or 'Fake' classification using high-dimensional boundaries.

        """)

    st.divider()
    st.markdown("""
    <div style="text-align: left;">
        <p>Source code and technical documentation are available on our official repository:</p>
        <a href="https://github.com/letsdothis2003/AI-Generated-Image-Detector" target="_blank">View on GitHub Repository</a>
        <p style="font-size: 0.8em; color: gray; margin-top: 10px;">Contains documentation and thought process that went into this project.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
 
    initialize_state()
