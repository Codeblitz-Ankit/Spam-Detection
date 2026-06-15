import os, json, threading
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import tensorflow as tf

app = FastAPI()

MODELS_DIR  = '/home/azureuser/models'
ROUND_FILE  = '/home/azureuser/models/current_round.json'
MIN_CLIENTS = 3

VOCAB_SIZE  = 8000
MAX_LEN     = 100
EMBED_DIM   = 32

pending_uploads      = []
clients_lock         = threading.Lock()
connected_dashboards = []

X_test = np.load(f'{MODELS_DIR}/X_test.npy')
y_test = np.load(f'{MODELS_DIR}/y_test.npy')
print(f"Test data loaded: {X_test.shape}")

def get_current_round():
    if os.path.exists(ROUND_FILE):
        return json.load(open(ROUND_FILE))['round']
    return 0

def save_round(n):
    json.dump({'round': n}, open(ROUND_FILE, 'w'))

def get_accuracy():
    path = f'{MODELS_DIR}/accuracy.json'
    if os.path.exists(path):
        return json.load(open(path))
    return {'round': 0, 'accuracy': 0}

def load_history():
    path = f'{MODELS_DIR}/history.json'
    if os.path.exists(path):
        return json.load(open(path))
    return []

def save_history(entry):
    history = load_history()
    history.append(entry)
    json.dump(history, open(f'{MODELS_DIR}/history.json', 'w'))

def build_model():
    inputs  = tf.keras.Input(shape=(MAX_LEN,), name='input_ids')
    x       = tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM, name='embedding')(inputs)
    x       = tf.keras.layers.GlobalAveragePooling1D(name='pooling')(x)
    x       = tf.keras.layers.Dense(24, activation='relu', name='dense_head')(x)
    x       = tf.keras.layers.Dropout(0.2, name='dropout')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)
    model   = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

class ClientUpload(BaseModel):
    tokens:      List[float]
    label:       int
    num_samples: int
    client_id:   str

@app.get('/get_model')
def get_model():
    r    = get_current_round()
    path = f'{MODELS_DIR}/round_{r}.tflite'
    if not os.path.exists(path):
        path = f'{MODELS_DIR}/round_0.tflite'
    return FileResponse(path, media_type='application/octet-stream',
                        filename='model.tflite')

@app.post('/upload_weights')
def upload_weights(data: ClientUpload):
    with clients_lock:
        pending_uploads.append(data)
        count = len(pending_uploads)

    print(f"[FL] Client {data.client_id[:8]} uploaded | {count}/{MIN_CLIENTS}")
    broadcast({'event': 'upload', 'client_id': data.client_id[:8],
               'total_uploads': count, 'needed': MIN_CLIENTS})

    if count >= MIN_CLIENTS:
        threading.Thread(target=run_fedavg).start()
        return {'status': 'fedavg_triggered', 'uploads': count}
    return {'status': 'received', 'uploads': count, 'needed': MIN_CLIENTS - count}

@app.get('/metrics')
def metrics():
    acc = get_accuracy()
    return {
        'round':           get_current_round(),
        'pending_uploads': len(pending_uploads),
        'accuracy':        acc.get('accuracy', 0),
        'history':         load_history()
    }

@app.websocket('/ws/dashboard')
async def dashboard_ws(websocket: WebSocket):
    await websocket.accept()
    connected_dashboards.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_dashboards.remove(websocket)

def run_fedavg():
    global pending_uploads
    with clients_lock:
        uploads = pending_uploads.copy()
        pending_uploads = []

    print(f"\n{'='*50}")
    print(f"[FedAvg] Round starting | Clients: {len(uploads)}")
    broadcast({'event': 'fedavg_started', 'num_clients': len(uploads)})

    current_round = get_current_round()

    # Load current global model
    model = build_model()
    weights_path = f'{MODELS_DIR}/round_{current_round}.weights.h5'
    model.load_weights(weights_path)

    # Train on each client's data locally (simulates what phone would do)
    X_new = np.array([u.tokens for u in uploads], dtype=np.float32)
    y_new = np.array([u.label  for u in uploads], dtype=np.float32)

    # Fine-tune only head layers
    model.get_layer('embedding').trainable = False
    model.get_layer('pooling').trainable   = False
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_new, y_new, epochs=3, verbose=1)

    # Evaluate
    old_acc  = get_accuracy().get('accuracy', 0)
    y_pred   = (model.predict(X_test, verbose=0) >= 0.5).astype(int).flatten()
    new_acc  = round(float(np.mean(y_pred == y_test)) * 100, 4)
    delta    = round(new_acc - old_acc, 4)

    new_round = current_round + 1
    save_round(new_round)

    # Save weights
    model.save_weights(f'{MODELS_DIR}/round_{new_round}.weights.h5')

    # Export new tflite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(f'{MODELS_DIR}/round_{new_round}.tflite', 'wb') as f:
        f.write(tflite_model)

    # Save accuracy + history
    json.dump({'round': new_round, 'accuracy': new_acc},
              open(f'{MODELS_DIR}/accuracy.json', 'w'))

    entry = {
        'round':        new_round,
        'old_accuracy': old_acc,
        'new_accuracy': new_acc,
        'delta':        delta,
        'num_clients':  len(uploads),
        'samples':      len(uploads)
    }
    save_history(entry)

    print(f"[FedAvg] Round {new_round} complete")
    print(f"[FedAvg] Old accuracy : {old_acc}%")
    print(f"[FedAvg] New accuracy : {new_acc}%")
    print(f"[FedAvg] Delta        : {delta:+.4f}%")
    print(f"[FedAvg] Saved        : round_{new_round}.tflite")
    print(f"{'='*50}\n")

    broadcast({
        'event':        'fedavg_done',
        'round':        new_round,
        'old_accuracy': old_acc,
        'new_accuracy': new_acc,
        'delta':        delta,
        'num_clients':  len(uploads)
    })

def broadcast(data: dict):
    import asyncio
    msg = json.dumps(data)
    for ws in connected_dashboards.copy():
        try:
            asyncio.run(ws.send_text(msg))
        except:
            pass
