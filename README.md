# Hand Bone Fracture Detection System

## Overview

The **Hand Bone Fracture Detection System** is a deep learning-based project designed to analyze hand X-ray images and identify suspected bone fractures.

The project uses image preprocessing and deep learning techniques to classify and detect fracture-related patterns in hand X-ray images. It includes model training, testing, and a web-based interface for using the trained model.

> **Note:** This project is a prototype for educational/research purposes and is not intended to replace professional medical diagnosis.

## Features

- Hand X-ray image analysis
- Fracture detection using deep learning
- Image preprocessing using OpenCV and Albumentations
- Trained PyTorch models
- Model testing and evaluation
- Streamlit-based application interface
- Visual representation of model/system architecture

## Technologies Used

- **Python**
- **PyTorch**
- **Torchvision**
- **OpenCV**
- **NumPy**
- **Pillow**
- **Albumentations**
- **Matplotlib**
- **Streamlit**

## Project Structure

```text
Hand_Project/
│
├── images/
│   ├── model_architecture.png
│   └── system_architecture.png
│
├── model/
│   ├── hand_xray_model.pth
│   └── Resnet.pt
│
├── app.py
├── Model_Fracture_Detection.py
├── Model_Object_Detection.py
├── train_model.py
├── test.py
├── requirements.txt
├── run.bat
├── .gitignore
└── .gitattributes