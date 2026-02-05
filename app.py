import streamlit as st
import numpy as np
from PIL import Image
import joblib
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt

# Import your backend scripts
from data_loader import load_ai_detection_dataset
from features import extract_features
from model import MnistSvmModel
from evaluation import evaluate_on_test, plot_confusion_matrix

# --- Page Configuration ---
st.set_page_config(page_title="AI Image Detector", page_icon="🔍", layout="wide")

# File where the trained model will be stored
MODEL_FILE = "ai_detection_model.joblib"

# Initialize session state for the model and metrics
if 'trained_model' not in st.session_state:
    if os.path.exists(MODEL_FILE):
        try:
            st.session_state.trained_model = joblib.load(MODEL_FILE)
            st.session_state.target_size = 64
            st.session_state.accuracy = None
            st.session_state.cm = None
        except Exception as e:
            st.session_state.trained_model = None
    else:
        st.session_state.trained_model = None

st.title("🔍 AI-Generated Image Detection System")
st.markdown("""
This application detects artifacts in AI-generated imagery using **HOG (Histogram of Oriented Gradients)** and **SVM**.
""")

# --- Sidebar: Configuration & Training ---
with st.sidebar:
    st.header("🛠️ Model Management")
    
    # Use a relative path by default
    default_path = os.path.join(os.getcwd(), "Data")
    data_path = st.text_input("Dataset Directory", default_path)
    
    st.subheader("Training Parameters")
    target_size = st.slider("Image Resize Dimension", 32, 128, 64)
    pca_comps = st.slider("PCA Components", 10, 100, 50)
    
    if st.button("🚀 Train & Save Model"):
        start_time = time.time()
        progress_bar = st.progress(0)
        timer_text = st.empty()
        
        with st.status("Training in progress...", expanded=True) as status:
            # Step 1: Loading
            st.write("1. Loading dataset images...")
            X_train, y_train, X_test, y_test = load_ai_detection_dataset(data_path, target_size=(target_size, target_size))
            
            if X_train is None or X_train.size == 0:
                st.error("Error: Dataset folder empty or path incorrect.")
            else:
                elapsed = time.time() - start_time
                timer_text.markdown(f"**Elapsed Time:** {elapsed:.2f}s")
                progress_bar.progress(25)

                # Step 2: Features
                st.write(f"2. Extracting features from {len(X_train)} images...")
                X_train_f = extract_features(X_train)
                X_test_f = extract_features(X_test)
                
                elapsed = time.time() - start_time
                timer_text.markdown(f"**Elapsed Time:** {elapsed:.2f}s")
                progress_bar.progress(60)

                # Step 3: Model Fitting
                st.write("3. Training SVM classifier...")
                model = MnistSvmModel(pca_components=pca_comps, kernel="linear")
                model.fit(X_train_f, y_train)
                
                elapsed = time.time() - start_time
                timer_text.markdown(f"**Elapsed Time:** {elapsed:.2f}s")
                progress_bar.progress(85)

                # Step 4: Saving
                st.write("4. Saving model to local disk...")
                joblib.dump(model, MODEL_FILE)
                
                # Step 5: Evaluation
                st.write("5. Final evaluation...")
                results = evaluate_on_test(model, X_test_f, y_test)
                
                # Update Session State
                st.session_state.trained_model = model
                st.session_state.accuracy = results['accuracy']
                st.session_state.cm = results['confusion_matrix']
                st.session_state.target_size = target_size
                
                total_time = time.time() - start_time
                progress_bar.progress(100)
                timer_text.markdown(f"**Total Training Time:** {total_time:.2f}s")
                
                status.update(label=f"Training Complete in {total_time:.2f}s!", state="complete", expanded=False)
                st.success(f"Model saved successfully to `{MODEL_FILE}`")

# --- Main Interface ---
tab1, tab2 = st.tabs(["🚀 Live Detection", "📊 Model Performance"])

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader("Drop an image here...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.header("📊 Detection Result")
        if st.session_state.trained_model is None:
            st.info("The model hasn't been trained yet. Please use the sidebar to start the training process.")
        elif uploaded_file:
            with st.spinner("Analyzing image gradients..."):
                size = st.session_state.get('target_size', 64)
                test_img = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
                test_arr = np.array(test_img).astype(np.float32) / 255.0
                test_arr = np.expand_dims(test_arr, axis=0)
                
                feat = extract_features(test_arr, use_extra_stats=True)
                prediction = st.session_state.trained_model.predict(feat)[0]
                
                if prediction == 1:
                    st.error("### 🤖 Result: AI GENERATED")
                    st.write("Found statistical anomalies in gradient distribution.")
                else:
                    st.success("### 📸 Result: REAL PHOTOGRAPH")
                    st.write("Gradients match natural optical signatures.")
                    
                if st.session_state.accuracy:
                    st.metric("Model Accuracy (on test set)", f"{st.session_state.accuracy:.2%}")
        else:
            st.write("Upload an image on the left to begin analysis.")

with tab2:
    st.header("Model Evaluation Metrics")
    if st.session_state.cm is not None:
        st.subheader("Confusion Matrix")
        st.write("This matrix shows how many images were correctly or incorrectly classified.")
        
        # Display the Confusion Matrix using the helper function
        fig, ax = plt.subplots(figsize=(6, 4))
        # Note: We can reuse the plot logic or just build a quick seaborn heatmap here
        import seaborn as sns
        sns.heatmap(st.session_state.cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'], ax=ax)
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        st.pyplot(fig)
        
        if st.session_state.accuracy:
            st.write(f"**Final Accuracy:** {st.session_state.accuracy:.4f}")
    else:
        st.info("Training stats will appear here after the model is trained.")

# Footer
st.divider()
st.caption("Fahim Tanvir & Ahmed Ali | CSCI-367 | AI Detection Project")