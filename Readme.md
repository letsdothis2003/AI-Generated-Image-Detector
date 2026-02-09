# AI Image Detector
URL: https://ai-generated-image-detector-letsdothis2003.streamlit.app


Developers: Fahim Tanvir and Ahmed Ali

During 2020, deepfakes or images manipulated by artificial intelligence became popular as they looked uncanny. Around fall of 2025, this interest got further developed as we learned more about digital image processing and also
other security issues regarding ai-generated content(imagine the nefarious things done by GROK).

## Overview

This project explores how machine learning models learn to interpret images. The goal is to build a system that can detect and classify different types of visual data.
So far, we used HOG(Histogram of Oriented Gradients) Features, PCA(Principal Component Analysis) , and SVM(Support Vector Machine).

### Objectives
- Detect faces in static images  
- Train model using image data from out datasets
- Use our algorithms to compare what an AI-generated image and a real image looks like and separates  them.
- Make sure it can accurately distinguish between the 2 through various testing. 


We used datasets from these sources:

https://www.kaggle.com/datasets/kaustubhdhote/human-faces-dataset


https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images

We used 2 so we could expirement with some non-humanoid images. It is primarily meant for images of people though, atleast in this developement stage. 




## How it works:

1.)When you go on the applicaiton, YOU MUST TRAIN THE MODEL FIRST. This is to ensure proper initialization.
2.)You can mess around with the size settings. Just a warning, the higher these parameters are, the slower the model training  is:
<img width="1484" height="751" alt="image" src="https://github.com/user-attachments/assets/97c7bade-4f87-4019-b68d-fb985ce0c7d8" />

3.)You can test it out with an image you have.
<img width="1485" height="831" alt="image" src="https://github.com/user-attachments/assets/9dea91ed-5661-4463-97fd-e1a9a453ee41" />

## Stuff we got to iron out:
We are satisfied with the demo of our program. Some issues we need to work on is our model doesn't do great with images that contain multiple(around 6 to 8) people or object, often flagging it as AI. If its a very saturated image, it might also trigger it as AI. Besides that, increasing the PVC parameters does increase the training time a bit so we would like to optimize this in later builds. 





## CURRENT AND FUTURE PLANS 
- A bit more unique and user-friendly interface
- Experiment with it so it can work with live images
- After getting it successfully working on images of people or art, we move onto to text to combat Google's AI models.
- Apply the same logic into videos using video frames.
- Modify model so it looks at possible metadata of images
-Frequent updates to the dataset to ensure improvements in accuracy(you might expect an increase from 8000 training images and 2000 test ones to increase significantly one day). 

