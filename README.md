# Real-Time Emotion Detection System

## Overview

This project is a Computer Vision and Deep Learning based Emotion Detection System developed using CNN, OpenCV, and TensorFlow. The model detects human emotions from facial expressions using grayscale facial images from the FER2013 dataset.

The system performs face detection, image preprocessing, emotion classification, and real-time emotion prediction from facial images.

---

## Features

* Facial emotion recognition using CNN
* Face detection using OpenCV Haar Cascade
* Image preprocessing and normalization
* Data augmentation for improved training
* Real-time emotion prediction
* Accuracy and loss visualization
* Confusion matrix and classification report
* Trained model saving using `.h5` format

---

## Technologies Used

* Python
* TensorFlow
* Keras
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* PIL
* Jupyter Notebook

---

## Dataset

Dataset used: FER2013 Facial Expression Dataset

Emotions classified:

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

A reduced subset of the dataset was used for faster execution and efficient model training.

---

## Project Workflow

1. Data Collection
2. Data Preprocessing
3. Data Augmentation
4. CNN Model Building
5. Model Training and Validation
6. Model Evaluation
7. Face Detection using OpenCV
8. Emotion Prediction
9. Model Saving

---

## Model Architecture

The CNN model includes:

* Conv2D Layers
* MaxPooling Layers
* Batch Normalization
* Dropout Layers
* Dense Layers
* Softmax Output Layer

---

## Evaluation Metrics

* Accuracy Score
* Validation Accuracy
* Loss Graph
* Confusion Matrix
* Classification Report

---

## Project Structure

```bash
Emotion_Detection/
│
├── Emotion_Detection.ipynb
├── emotion_model.h5
├── haarcascade_frontalface_default.xml
├── images.jpg
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone <repository_link>
```

Install required libraries:

```bash
pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn pillow
```

---

## Run the Project

Open the notebook:

```bash
jupyter notebook
```

Run all cells in `Emotion_Detection.ipynb`.

---

## Future Improvements

* Real-time webcam emotion detection
* Streamlit deployment
* Transfer Learning implementation
* Mobile application integration

---

## Conclusion

This project demonstrates the practical implementation of Computer Vision and Deep Learning techniques for facial emotion recognition. The developed system successfully classifies emotions from facial expressions and serves as a lightweight AI-based emotion detection application.

