import os, re, json, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf

# ── Config ──────────────────────────────────────────
DATA_PATH  = 'data/SMSSpamCollection'
MODEL_DIR  = 'models'
VOCAB_SIZE = 8000
MAX_LEN    = 100
EMBED_DIM  = 32
SEED       = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Load & preprocess ────────────────────────────────
print("Loading data...")
texts, labels = [], []
with open(DATA_PATH, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            labels.append(1 if parts[0] == 'spam' else 0)
            texts.append(parts[1])

def clean(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

texts = [clean(t) for t in texts]

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=SEED, stratify=labels)

# Load existing tokenizer
with open(f'{MODEL_DIR}/vocab.json', 'r') as f:
    word_index = json.load(f)

def encode(texts_list):
    result = []
    for text in texts_list:
        seq = [word_index.get(w, 1) for w in text.split()]
        seq = [min(idx, VOCAB_SIZE-1) for idx in seq]
        seq = seq[:MAX_LEN] + [0] * max(0, MAX_LEN - len(seq))
        result.append(seq)
    return np.array(result, dtype=np.float32)

X_train_enc = encode(X_train)
X_test_enc  = encode(X_test)
y_train = np.array(y_train, dtype=np.float32)
y_test  = np.array(y_test,  dtype=np.float32)

print(f"Train: {len(X_train_enc)} | Test: {len(X_test_enc)}")

# ── Build model with named layers ────────────────────
print("Building model...")

# Base (frozen during FL)
inputs = tf.keras.Input(shape=(MAX_LEN,), name='input_ids')
x = tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM, name='embedding')(inputs)
x = tf.keras.layers.GlobalAveragePooling1D(name='pooling')(x)

# Head (trainable during FL)
x = tf.keras.layers.Dense(24, activation='relu', name='dense_head')(x)
x = tf.keras.layers.Dropout(0.2, name='dropout')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)

model = tf.keras.Model(inputs, outputs, name='SpamDetector')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ── Train ────────────────────────────────────────────
neg, pos = np.bincount(y_train.astype(int))
class_weight = {0: 1.0, 1: neg/pos}

print("\nTraining...")
model.fit(
    X_train_enc, y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=32,
    class_weight=class_weight,
    verbose=1
)

# ── Evaluate ─────────────────────────────────────────
y_pred = (model.predict(X_test_enc, verbose=0) >= 0.5).astype(int).flatten()
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc*100:.2f}%")

# Save accuracy for server reference
with open(f'{MODEL_DIR}/accuracy.json', 'w') as f:
    json.dump({'round': 0, 'accuracy': round(acc*100, 4)}, f)

# ── Save full weights ─────────────────────────────────
model.save_weights(f'{MODEL_DIR}/round_0.weights.h5')
print("Full weights saved.")

# ── Export inference TFLite ───────────────────────────
print("Exporting inference.tflite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open(f'{MODEL_DIR}/round_0.tflite', 'wb') as f:
    f.write(tflite_model)
print(f"inference.tflite: {len(tflite_model)/1024:.1f} KB")

# ── Export trainable head TFLite ──────────────────────
print("Exporting trainable head...")

# Head only model: takes pooling output, returns updated head weights
pooling_output = model.get_layer('pooling').output
head_input = tf.keras.Input(shape=(EMBED_DIM,), name='head_input')
hx = model.get_layer('dense_head')(head_input)
hx = model.get_layer('output')(hx)
head_model = tf.keras.Model(head_input, hx, name='HeadModel')
head_model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                   loss='binary_crossentropy')

# Save head weights separately
head_weights = {
    'dense_head': model.get_layer('dense_head').get_weights(),
    'output': model.get_layer('output').get_weights()
}
np.save(f'{MODEL_DIR}/head_weights_round_0.npy',
        np.array(head_weights, dtype=object))

print("Head weights saved.")
print(f"\nDense head shape: {model.get_layer('dense_head').get_weights()[0].shape}")
print(f"Output shape:     {model.get_layer('output').get_weights()[0].shape}")

# ── Save test data for server evaluation ──────────────
np.save(f'{MODEL_DIR}/X_test.npy', X_test_enc)
np.save(f'{MODEL_DIR}/y_test.npy', y_test)
print("Test data saved for server-side evaluation.")

print("\n✅ Done. Files in models/:")
print(os.listdir(MODEL_DIR))
