# Haven't thought of a creative name for this yet! 

Developers: Fahim Tanvir and Ahmed Ali

During covid, deepfakes or images manipulated by artificial intelligence became popular as they looked uncanny. Around fall of 2025, this interest got further developed as we learned more about digital image processing and also
other security issues regarding ai-generated content(imagine the nefarious things done by GROK).

## Overview

This project explores how machine learning models learn to interpret images. The goal is to build a system that can detect and classify different types of visual data.
So far, we used HOG(Histogram of Oriented Gradients) Features, PCA(Principal Component Analysis) , and SVM(Support Vector Machine).

### Objectives
- Detect faces in static images  
- Explore real-time detection via webcam  
- Compare different model architectures  
- Build a unified interface supporting both digit and face detection
-You know how Ai for some reason gives weird skin textures cause it can't create realistic lighting or skin, this is just that.


We used datasets from these sources:

https://www.kaggle.com/datasets/kaustubhdhote/human-faces-dataset

https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images



## CURRENT AND FUTURE PLANS 

Currently, we managed to get it running as it properly trains with the images as accuracy is 96%. We did do some testing here and there by inputting a random image as a test case,
it actually can detect ai-generated images fine but it seems to fail if the image is super bright or saturated. Definetly need to iron that one out.

We would like to create a runnable version of it through applit api, and make optimizations here and there as training is very slow(thats another drawback of our program at the moment).

Besides that, here are some future plans:
-A user-friendly interface
- After getting it successfully working on images of people or art, we move onto to text to combat Google's AI models.
- Apply the same logic into videos using video frames.
- Modify model so it looks at possible metadata of images



  




### Objectives
- Use a non-Ai picture to assess an Ai generated one. 
- Detects if it looks weird or uncanny. 

