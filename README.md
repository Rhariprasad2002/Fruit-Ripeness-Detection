# 🍎 Real-Time Fruit Ripeness Detection using Deep Learning

An AI-powered computer vision project that detects fruit ripeness in **real-time** using Deep Learning and Transfer Learning.  
The system classifies fruits into **Unripe, Ripe, and Overripe** stages using images or webcam.

---

## 📌 Project Overview

Fruit quality inspection is usually done manually, which is time-consuming and inconsistent.  
This project automates the process using a **Convolutional Neural Network (CNN)** with **Transfer Learning (MobileNetV2)** and deploys the model as a **Streamlit web application**.

The system can:
- Detect fruit ripeness from uploaded images 📷  
- Detect fruit ripeness in real-time using webcam 🎥  

---

## 🎯 Features

- Deep Learning Image Classification  
- Transfer Learning (MobileNetV2)  
- Handles dataset imbalance using class weights  
- Real-time webcam detection  
- Image upload prediction  
- Streamlit Web App deployment  

---

## 🧠 Model Details

| Item | Details |
|---|---|
| Algorithm | CNN + Transfer Learning |
| Base Model | MobileNetV2 |
| Classes | Overripe, Ripe, Unripe |
| Accuracy | ~95% validation accuracy |
| Framework | TensorFlow / Keras |

---

## 📂 Project Structure

Fruit-Ripeness-Detection/
│
├── train_model.py # Model training script
├── app.py # Streamlit web app
├── labels.json # Class label order
├── fruit_model.h5 # Trained model
├── requirements.txt # Dependencies
└── README.md

## Train the Model


dataset/
   Train/
      Overripe/
      Ripe/
      Unripe/
   Test/
      Overripe/
      Ripe/
      Unripe/

Run training:

python train_model.py

This will generate:

fruit_model.h5
labels.json

🌐 Run the Web Application
streamlit run app.py
