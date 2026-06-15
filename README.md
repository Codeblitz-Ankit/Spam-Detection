# Federated Learning Spam Detection System

## Overview

A privacy-focused spam detection system built using TensorFlow Lite, FastAPI, Android, and a real-time monitoring dashboard.

The project demonstrates how mobile devices can participate in a collaborative spam detection workflow while minimizing the transfer of sensitive SMS data. Android clients perform local inference using a TensorFlow Lite model and communicate with a centralized FastAPI server that manages model updates, evaluation, and deployment.

---

## Features

* Android SMS Spam Detection Client
* TensorFlow Lite On-Device Inference
* FastAPI Backend Server
* Real-Time Monitoring Dashboard
* Model Versioning and Round Tracking
* Accuracy Monitoring
* Azure VM Deployment
* Privacy-Oriented Architecture

---

## System Architecture

```text
Android Client
      │
      │ Prediction / Training Data
      ▼
FastAPI Server
      │
      ├── Model Management
      ├── Accuracy Evaluation
      ├── Round Tracking
      └── Model Export
      │
      ▼
TensorFlow Lite Model
      │
      ▼
Redistributed To Clients

      ▲
      │
Streamlit Dashboard
(Real-Time Monitoring)
```

---

## Repository Structure

```text
Spam-Detection/
│
├── android/
│   └── SpamDetector/
│
├── dashboard/
│   └── dashboard.py
│
├── server/
│   └── main.py
│
├── data/
│   └── SMSSpamCollection
│
├── models/
│   ├── round_0.tflite
│   ├── round_0.weights.h5
│   ├── accuracy.json
│   ├── current_round.json
│   └── vocab.json
│
├── build_trainable_model.py
├── train.py
├── requirements.txt
└── README.md
```

---

## Dataset

Dataset Used:

* SMS Spam Collection Dataset

Classes:

* Ham (0)
* Spam (1)

---

## Model Architecture

```text
Embedding
      ↓
GlobalAveragePooling1D
      ↓
Dense (ReLU)
      ↓
Dropout
      ↓
Dense (Sigmoid)
```

---

## Training Configuration

| Parameter           | Value |
| ------------------- | ----- |
| Vocabulary Size     | 8000  |
| Sequence Length     | 100   |
| Embedding Dimension | 32    |
| Batch Size          | 32    |
| Epochs              | 10    |

---

## Android Client

The Android application performs on-device spam detection using TensorFlow Lite.

### Included Assets

```text
round_0.tflite
vocab.json
```

### Technologies

* Kotlin
* TensorFlow Lite
* Android SDK
* Material Components

---

## Backend Server

The backend is implemented using FastAPI.

### Main Responsibilities

* Serve latest TensorFlow Lite model
* Accept client uploads
* Track training rounds
* Evaluate model accuracy
* Export updated models
* Provide dashboard metrics

### Endpoints

```http
GET  /get_model
POST /upload_weights
GET  /metrics
WS   /ws/dashboard
```

---

## Dashboard

The Streamlit dashboard provides:

* Current training round
* Accuracy tracking
* Upload statistics
* Client activity monitoring
* Real-time server events

### Start Dashboard

```bash
streamlit run dashboard/dashboard.py --server.port 8501
```

---

## Local Setup

### Clone Repository

```bash
git clone https://github.com/Codeblitz-Ankit/Spam-Detection.git
cd Spam-Detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Initial Model

```bash
python train.py
```

### Start Backend

```bash
cd server
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## Production Deployment

### Development Environment

Lightning AI

Used for:

* Model training
* Experimentation
* Development

### Production Environment

Azure Virtual Machine

Used for:

* FastAPI hosting
* Dashboard hosting
* Model distribution
* Monitoring

---

## Monitoring

Current Round:

```bash
cat models/current_round.json
```

Accuracy:

```bash
cat models/accuracy.json
```

Generated Models:

```bash
ls models/
```

---

## Future Improvements

* Differential Privacy
* Secure Aggregation
* Federated Averaging with True Weight Aggregation
* Multi-Client Simulation
* Transformer-Based Models
* Docker Deployment
* Kubernetes Deployment

---

## Author

Ankit Koli

Federated Learning • Android Development • FastAPI • TensorFlow Lite • Machine Learning
