import os, re, json, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── Config ──────────────────────────────────────────
DATA_PATH   = 'data/SMSSpamCollection'
CKPT_DIR    = 'checkpoints'
MODEL_DIR   = 'models'
VOCAB_SIZE  = 8000
MAX_LEN     = 100
EMBED_DIM   = 32
EPOCHS      = 10
BATCH_SIZE  = 32
SEED        = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Load data ───────────────────────────────────────
print("Loading data...")
texts, labels = [], []
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            labels.append(1 if parts[0] == 'spam' else 0)
            texts.append(parts[1])
print(f"Total: {len(texts)} | Spam: {sum(labels)} | Ham: {len(labels)-sum(labels)}")

# ── Preprocess ──────────────────────────────────────
def clean(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

texts = [clean(t) for t in texts]

# ── Split ───────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=SEED, stratify=labels)

# ── Tokenize ────────────────────────────────────────
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN, padding='post')
X_test  = pad_sequences(tokenizer.texts_to_sequences(X_test),  maxlen=MAX_LEN, padding='post')
y_train, y_test = np.array(y_train), np.array(y_test)

# Save tokenizer
with open(f'{MODEL_DIR}/vocab.json', 'w') as f:
    json.dump(tokenizer.word_index, f)
print("Tokenizer saved.")

# ── Model ───────────────────────────────────────────
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_model()
model.summary()

# ── Checkpoint callback (keep last 2 only) ──────────
class KeepLast2(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        path = f'{CKPT_DIR}/epoch_{epoch+1:02d}.weights.h5'
        self.model.save_weights(path)
        print(f"\nSaved: {path}")
        files = sorted([f for f in os.listdir(CKPT_DIR) if f.endswith('.weights.h5')])
        for old in files[:-2]:
            os.remove(f'{CKPT_DIR}/{old}')
            print(f"Deleted: {old}")

# ── Train ───────────────────────────────────────────
neg, pos = np.bincount(y_train)
class_weight = {0: 1.0, 1: neg/pos}

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    callbacks=[KeepLast2()],
    verbose=1
)

# ── Evaluate ─────────────────────────────────────────
y_pred = (model.predict(X_test, verbose=0) >= 0.5).astype(int)
print("\n" + classification_report(y_test, y_pred, target_names=['Ham','Spam']))

# ── Export TFLite ────────────────────────────────────
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open(f'{MODEL_DIR}/round_0.tflite', 'wb') as f:
    f.write(tflite_model)

model.save_weights(f'{MODEL_DIR}/round_0.weights.h5')
print(f"\nDone. Files in models/:")
print(os.listdir(MODEL_DIR))
