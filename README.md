# Federated Learning Spam Detection System

## Overview

A privacy-preserving spam detection system built using Federated Learning, TensorFlow Lite, FastAPI, and Android.

Instead of sending user SMS messages to a central server, Android clients train locally and upload only model weight updates. The server aggregates updates using Federated Averaging (FedAvg) to improve the global model while preserving user privacy.

### Key Features

* Federated Learning (FedAvg)
* Android-based local training
* TensorFlow Lite deployment
* FastAPI aggregation server
* Real-time monitoring dashboard
* Spam/Ham binary classification
* Privacy-preserving architecture

---

## Architecture

```text
Android Clients
       │
       │ Local Training
       ▼
Weight Updates Only
       │
       ▼
FastAPI Aggregation Server
       │
       │ FedAvg
       ▼
Updated Global Model
       │
       ▼
TensorFlow Lite Model
       │
       ▼
Redistributed To Clients
```

---

## Project Structure

```text
Spam-Detection/
│
├── train.py
├── build_trainable_model.py
│
├── data/
│   └── SMSSpamCollection
│
├── models/
│   ├── round_0.tflite
│   ├── round_0.weights.h5
│   ├── vocab.json
│   ├── accuracy.json
│   └── current_round.json
│
├── server/
│   └── main.py
│
└── checkpoints/
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
Embedding Layer
       ↓
GlobalAveragePooling1D
       ↓
Dense (24 ReLU)
       ↓
Dropout (0.2)
       ↓
Dense (1 Sigmoid)
```

---

## Training Configuration

| Parameter           | Value |
| ------------------- | ----- |
| Vocabulary Size     | 8000  |
| Max Sequence Length | 100   |
| Embedding Dimension | 32    |
| Batch Size          | 32    |
| Epochs              | 10    |
| Random Seed         | 42    |

---

## Performance

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | ~98.8% |
| Precision | ~97.0% |
| Recall    | ~94.8% |
| F1 Score  | ~95.9% |

---

# Local Setup

## Clone Repository

```bash
git clone https://github.com/Codeblitz-Ankit/Spam-Detection.git
cd Spam-Detection
```

## Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Training The Model

Run:

```bash
python train.py
```

Generated files:

```text
models/
├── round_0.tflite
├── round_0.weights.h5
└── vocab.json
```

---

# Starting The Federated Learning Server

Navigate to:

```bash
cd server
```

Run:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Server Endpoints:

### Download Current Model

```http
GET /get_model
```

### Upload Client Weights

```http
POST /upload_weights
```

### Metrics

```http
GET /metrics
```

### Dashboard WebSocket

```http
/ws/dashboard
```

---

# Testing Server Health

Open:

```text
http://localhost:8000/metrics
```

Expected response:

```json
{
  "round": 0,
  "pending_uploads": 0
}
```

---

# Federated Learning Workflow

1. Client downloads latest model
2. Client trains locally
3. Client uploads weight updates
4. Server waits for uploads
5. FedAvg aggregation runs
6. New global model generated
7. Updated model distributed

---

# Production Deployment

## Training Environment

Lightning.ai

```text
/teamspace/studios/this_studio/
```

Used for:

* Training
* Experimentation
* Model generation

---

## Production Environment

Azure VM

Used for:

* FastAPI server
* FedAvg aggregation
* Dashboard hosting
* Model distribution

---

# Azure Commands

## SSH Into VM

```bash
ssh -i ~/.ssh/spam-fl-key.pem azureuser@YOUR_VM_IP
```

## Check Running Services

```bash
tmux ls
```

## Start FastAPI Server

```bash
tmux new -s server

cd ~/server

python3 -m uvicorn main:app \
--host 0.0.0.0 \
--port 8000
```

Detach:

```text
Ctrl+B
D
```

---

# Dashboard

## Start Dashboard

```bash
streamlit run dashboard.py \
--server.headless true \
--server.port 8501
```

## Open Dashboard

```text
http://YOUR_VM_IP:8501
```

---

# Monitoring

Check current round:

```bash
cat models/current_round.json
```

Check accuracy:

```bash
cat models/accuracy.json
```

List generated models:

```bash
ls -la models/
```

---

# Future Improvements

* Differential Privacy
* Secure Aggregation
* Multi-client simulations
* Transformer-based architecture
* Real-device benchmarking
* Docker deployment
* Kubernetes orchestration

---

# Author

Ankit Koli

Federated Learning • Android • FastAPI • TensorFlow Lite • Privacy-Preserving AI
