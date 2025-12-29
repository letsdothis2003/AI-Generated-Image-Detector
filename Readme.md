# Haven't thought of a creative name for this yet! 

Developers: Fahim Tanvir and Ahmed Ali

During covid, I got interested into deepfakes or images manipulated by artificial intelligence as they looked uncanny. Around fall of 2025, this interest got further developed as I learned more about digital image processing and also
other security issues regarding ai-generated content.

## Overview

This project explores how machine learning models learn to interpret images. The goal is to build a system that can detect and classify different types of visual data.
So far, we used HOG(Histogram of Oriented Gradients) Features, PCA(Principal Component Analysis) , and SVM(Support Vector Machine).

---




##  Phase 1 — MNIST Digit Recognition

The first phase uses MNIST dataset, a classic benchmark of  handwritten digits (0–9). We used this dataset: https://www.kaggle.com/datasets/hojjatk/mnist-dataset/data

### Objectives
- Preprocess and normalize image data  
- Train a neural network to classify digits  
- Evaluate accuracy and performance until its like 90-ish percent. 


### Results
96 percent accuracy baby! 

---

##  Phase 2 — Letter Recognition(In Progress)

This next phase repeats phase 1's directives, just with letters. We are currently are trying to find a good dataset for letters and other characters. 


### Objectives(EXACTLY AS PHASE ONE) 
- Preprocess and normalize image data 
- Train a neural network to classify digits  
- Evaluate accuracy and performance until its like 90-ish percent.

  

##  Phase 3 — Face Detection (Future)

The next phase expands the project into peoples faces(we are considering animals as well), which introduces more complexity such as higher-resolution images, lighting variation, and facial feature differences.

### Objectives
- Detect faces in static images  
- Explore real-time detection via webcam  
- Compare different model architectures  
- Build a unified interface supporting both digit and face detection



##  Phase 4 — Uncaniness Assessment(Future)

You know how Ai for some reason gives people extra fingers or weird skin textures, this is just that.


### Objectives
- Use a non-Ai picture to assess an Ai generated one. 
- Detects if it looks weird or uncanny. 

