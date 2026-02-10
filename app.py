import streamlit as st
import numpy as np
from PIL import Image
import joblib
import os
import time
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import hog
from skimage import exposure

# Import backend scripts
from data_loader import load_ai_detection_dataset
from features import extract_features
from model import MnistSvmModel
from evaluation import evaluate_on_test

# Page Configuration 
st.set_page_config(page_title="AI Image Detector", page_icon="[DEMO]", layout="wide")

# File where the trained model will be stored
MODEL_FILE = "ai_detection_model.joblib"

#Session State Initialization 
def initialize_state():
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
    if 'accuracy' not in st.session_state:
        st.session_state.accuracy = None
    if 'cm' not in st.session_state:
        st.session_state.cm = None
    if 'target_size' not in st.session_state:
        st.session_state.target_size = 64
    if 'history' not in st.session_state:
        st.session_state.history = []

initialize_state()

# Auto-load existing model
@st.cache_resource
def load_stored_model():
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            return None
    return None

if st.session_state.trained_model is None:
    st.session_state.trained_model = load_stored_model()

st.title("AI-Generated Image Detection System")
st.markdown("""
This system is meant to distinguish between authentic photography and AI-generated content. 
Inspired by how people be using AI to impersonate or make illegal images. 
""")

#Sidebar: Configuration & Training  
with st.sidebar:
    st.header("Model Management")
    default_path = os.path.join(os.getcwd(), "Data")
    data_path = st.text_input("Dataset Directory", default_path)
    
    st.subheader("Training Parameters")
    target_size = st.slider("Image Resize Dimension", 32, 128, st.session_state.target_size)
    pca_comps = st.slider("PCA Components", 10, 100, 50)
    
    if st.button("Train and Save Model"):
        start_time = time.time()
        progress_bar = st.progress(0)
        
        with st.status("Training System...", expanded=True) as status:
            X_train, y_train, X_test, y_test = load_ai_detection_dataset(data_path, target_size=(target_size, target_size))
            
            if X_train is None or X_train.size == 0:
                st.error("Dataset not found at specified path.")
            else:
                progress_bar.progress(20)
                st.write("Extracting HOG Features...")
                X_train_f = extract_features(X_train)
                X_test_f = extract_features(X_test)
                progress_bar.progress(50)
                
                st.write("Fitting SVM Classifier...")
                # Optimization: Note kernel is set here
                model = MnistSvmModel(pca_components=pca_comps, kernel="linear")
                model.fit(X_train_f, y_train)
                joblib.dump(model, MODEL_FILE)
                progress_bar.progress(80)
                
                st.write("Performing Evaluation...")
                results = evaluate_on_test(model, X_test_f, y_test)
                st.session_state.trained_model = model
                st.session_state.accuracy = results['accuracy']
                st.session_state.cm = results['confusion_matrix']
                st.session_state.target_size = target_size
                
                progress_bar.progress(100)
                status.update(label=f"Training Complete! ({time.time()-start_time:.1f}s)", state="complete")

    st.divider()
    st.subheader("System Information")
    st.text(f"Model Status: {'Loaded' if st.session_state.trained_model else 'Not Found'}")
    if st.session_state.trained_model:
        st.text(f"Resolution: {st.session_state.target_size}x{st.session_state.target_size}")

#Utility Functions for Main UI  
@st.cache_data
def get_hog_viz(image_np):
    """Generates a visualization of the HOG descriptors."""
    fd, hog_image = hog(image_np, orientations=8, pixels_per_cell=(8, 8),
                        cells_per_block=(1, 1), visualize=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return hog_image_rescaled

#Main Interface  
tab1, tab2, tab3 = st.tabs(["Batch Detection", "Model Performance", "Technical Analysis"])

with tab1:
    st.header("Upload Analysis Subject")
    uploaded_files = st.file_uploader("Select images for analysis", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if st.session_state.trained_model is None:
        st.warning("The detection engine is not ready. Please train the model via the sidebar.")
    elif uploaded_files:
        st.subheader(f"Analyzing {len(uploaded_files)} Subjects")
        
        batch_results = []
        for i, file in enumerate(uploaded_files):
            # Load and Preprocess
            image = Image.open(file)
            img_format = image.format
            img_mode = image.mode
            img_size = image.size
            
            size = st.session_state.target_size
            test_img = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
            test_arr = np.array(test_img).astype(np.float32) / 255.0
            test_arr_expanded = np.expand_dims(test_arr, axis=0)
            
            # Feature Extraction and Prediction
            feat = extract_features(test_arr_expanded, use_extra_stats=True)
            prediction = st.session_state.trained_model.predict(feat)[0]
            
            # Estimate confidence using decision function
            try:
                # Transformed feature for SVM
                processed_feat = st.session_state.trained_model.pca.transform(
                    st.session_state.trained_model.scaler.transform(feat)
                )
                dist = st.session_state.trained_model.svm.decision_function(processed_feat)
                # Sigmoid-like normalization for confidence
                confidence = 100 / (1 + np.exp(-abs(dist[0])))
            except:
                confidence = 100.0

            label = "AI-GENERATED" if prediction == 1 else "REAL PHOTOGRAPH"
            batch_results.append({
                "Filename": file.name,
                "Prediction": label,
                "Confidence": f"{confidence:.2f}%",
                "Format": img_format,
                "Dimensions": f"{img_size[0]}x{img_size[1]}"
            })
            
            # UI Layout for results
            with st.expander(f"Analysis: {file.name} - {label}"):
                col_img, col_hog, col_meta = st.columns([2, 2, 2])
                with col_img:
                    st.image(image, caption="Original Subject", use_container_width=True)
                with col_hog:
                    st.image(get_hog_viz(test_arr), caption="Gradient Orientation Map (HOG)", use_container_width=True)
                with col_meta:
                    st.write("**Metadata**")
                    st.write(f"Format: {img_format}")
                    st.write(f"Mode: {img_mode}")
                    st.write(f"Original: {img_size[0]}x{img_size[1]}")
                    st.write(f"Analyzed: {size}x{size}")
                    st.divider()
                    if label == "AI-GENERATED":
                        st.error(f"Classification: {label}")
                    else:
                        st.success(f"Classification: {label}")
                    st.progress(confidence / 100, text=f"Confidence Score: {confidence:.1f}%")

        # Summary Metrics and CSV Export
        if batch_results:
            st.divider()
            df = pd.DataFrame(batch_results)
            st.subheader("Batch Summary")
            st.dataframe(df, use_container_width=True)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Analysis Report (CSV)", data=csv, file_name="detection_report.csv", mime="text/csv")

with tab2:
    if st.session_state.accuracy is not None:
        st.header("Quantitative Evaluation")
        
        m1, m2 = st.columns(2)
        accuracy_pct = st.session_state.accuracy * 100
        m1.metric("Overall System Accuracy", f"{accuracy_pct:.2f}%")
        m2.metric("Classification Type", "Binary (SVM-Linear)")
        
        st.divider()
        
        if st.session_state.cm is not None:
            st.subheader("Confusion Matrix Analysis")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(st.session_state.cm, annot=True, fmt='d', cmap='Greens', 
                        xticklabels=['REAL', 'AI'], yticklabels=['REAL', 'AI'], ax=ax)
            plt.ylabel('Actual Classification')
            plt.xlabel('System Prediction')
            st.pyplot(fig)
            
            st.markdown("""
            **Understanding the results:**
            - **True Positives:** AI images correctly identified as AI.
            - **True Negatives:** Real photos correctly identified as Real.
            - **False Positives:** Real photos mistaken for AI.
            - **False Negatives:** AI images mistaken for Real photos.
            """)
    else:
        st.info("Performance analytics will appear here after the model has been trained.")

with tab3:
    st.header("Methodology")
    st.write("This project works using these methods")
    
    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    
    with col_a:
        st.subheader("Feature Extraction")
        st.info("""
        **HOG:** Looks at the edges and shapes. 
        AI images often leave watermarks and regularity in gradients that are invisible to the eye but clear to HOG.
        """)

    with col_b:
        st.subheader("Dimension Reduction")
        #Fetching current pca_comps from state
        current_pca = pca_comps if 'pca_comps' not in locals() else pca_comps
        st.info(f"""
        **PCA:**
        The raw HOG output is huge. We use PCA to reduce data down to the top {current_pca} components. 
        This removes noise and helps the SVM find the boundary.
        """)
        
    with col_c:
        st.subheader("Texture Analysis")
        st.info("""
        **LBP:**
        Scans pixel-level textures. AI imagery frequently has uncanny texture compared to natural photographic grain. 
        """)

    with col_d:
        st.subheader("Spatial Domain")
        st.info("""
        **GLCM Analysis:**
        Examines pixel-to-pixel relationships. AI struggles to replicate real-world light scattering. 
        """)

    with col_e:
        st.subheader("Classification")
        svm_c_val = locals().get('SVM_C', 1.0)
        st.info(f"""
        **SVM (Linear/RBF):**
        Using a penalty C-value ({svm_c_val}), it maps artifacts into a clear 'Real' or 'Fake' classification.
        """)
    
    st.divider()
    st.markdown("""
    <div style="text-align: left;">
        <p>Source code and technical documentation are available on our official repository:</p>
        <a href="https://github.com/letsdothis2003/AI-Generated-Image-Detector" target="_blank">View on GitHub Repository</a>
        <p style="font-size: 0.8em; color: gray; margin-top: 10px;">Contains documentation and thought process that went into this</p>
    </div>
    """, unsafe_allow_html=True)

