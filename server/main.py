import os, json, threading, numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import tensorflow as tf

app = FastAPI()

# ── Config ──────────────────────────────────────────
MODELS_DIR   = '/teamspace/studios/this_studio/models'
ROUND_FILE   = '/teamspace/studios/this_studio/models/current_round.json'
MIN_CLIENTS  = 1  # FedAvg triggers after this many uploads

# ── State ───────────────────────────────────────────
pending_weights = []
clients_lock    = threading.Lock()
connected_dashboards = []

def get_current_round():
    if os.path.exists(ROUND_FILE):
        return json.load(open(ROUND_FILE))['round']
    return 0

def save_round(n):
    json.dump({'round': n}, open(ROUND_FILE, 'w'))

# ── Schemas ─────────────────────────────────────────
class WeightUpload(BaseModel):
    weights: List[List[float]]
    num_samples: int
    client_id: str

# ── Endpoints ───────────────────────────────────────
@app.get('/get_model')
def get_model():
    r = get_current_round()
    path = f'{MODELS_DIR}/round_{r}.tflite'
    if not os.path.exists(path):
        path = f'{MODELS_DIR}/round_0.tflite'
    return FileResponse(path, media_type='application/octet-stream', filename='model.tflite')

@app.post('/upload_weights')
def upload_weights(data: WeightUpload):
    with clients_lock:
        pending_weights.append({
            'weights': data.weights,
            'num_samples': data.num_samples,
            'client_id': data.client_id
        })
        count = len(pending_weights)

    broadcast({'event': 'upload', 'client_id': data.client_id, 'total_uploads': count})

    if count >= MIN_CLIENTS:
        threading.Thread(target=run_fedavg).start()
        return {'status': 'fedavg_triggered', 'uploads': count}
    return {'status': 'received', 'uploads': count, 'needed': MIN_CLIENTS - count}

@app.get('/metrics')
def metrics():
    return {
        'round': get_current_round(),
        'pending_uploads': len(pending_weights)
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

# ── FedAvg ───────────────────────────────────────────
def run_fedavg():
    global pending_weights
    with clients_lock:
        uploads = pending_weights.copy()
        pending_weights = []

    broadcast({'event': 'fedavg_started', 'num_clients': len(uploads)})

    total_samples = sum(u['num_samples'] for u in uploads)
    avg_weights = []
    for i in range(len(uploads[0]['weights'])):
        layer = sum(
            np.array(u['weights'][i]) * (u['num_samples'] / total_samples)
            for u in uploads
        )
        avg_weights.append(layer.tolist())

    new_round = get_current_round() + 1
    save_round(new_round)

    # Save new weights
    weights_path = f'{MODELS_DIR}/round_{new_round}.weights.h5'
    np.save(weights_path, np.array(avg_weights, dtype=object))

    broadcast({
        'event': 'fedavg_done',
        'round': new_round,
        'num_clients': len(uploads),
        'total_samples': total_samples
    })

# ── Broadcast helper ─────────────────────────────────
def broadcast(data: dict):
    import asyncio
    msg = json.dumps(data)
    for ws in connected_dashboards.copy():
        try:
            asyncio.run(ws.send_text(msg))
        except:
            pass