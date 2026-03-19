"""
model.py — LSTM model builder, trainer, and predictor
"""

import numpy as np
import pickle
import os
import re
from collections import Counter

# ── Try importing TensorFlow; fall back gracefully ──
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ──────────────────────────────────────────────────────────────
# TEXT PREPROCESSING
# ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lowercase and remove special characters."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s'.,!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_vocab(words: list, max_vocab: int = 5000):
    """Build word → index and index → word mappings."""
    freq = Counter(words)
    vocab = ["<PAD>", "<UNK>"] + [w for w, _ in freq.most_common(max_vocab - 2)]
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word, vocab


def create_sequences(words: list, word2idx: dict, seq_len: int = 10):
    """Sliding window sequences for next-word prediction."""
    sequences = []
    for i in range(seq_len, len(words)):
        seq = words[i - seq_len: i + 1]
        encoded = [word2idx.get(w, 1) for w in seq]
        sequences.append(encoded)
    return np.array(sequences)


# ──────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE
# ──────────────────────────────────────────────────────────────

def build_lstm_model(vocab_size: int, embed_dim: int = 64, lstm_units: int = 128, seq_len: int = 10):
    """Build and compile the LSTM model."""
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not installed.")

    model = Sequential([
        Embedding(vocab_size, embed_dim, input_length=seq_len),
        LSTM(lstm_units, return_sequences=True),
        Dropout(0.2),
        LSTM(lstm_units // 2),
        Dropout(0.2),
        Dense(vocab_size, activation="softmax"),
    ])
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


# ──────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────

def train_model(
    text: str,
    seq_len: int = 10,
    epochs: int = 30,
    batch_size: int = 64,
    max_vocab: int = 3000,
    lstm_units: int = 128,
    embed_dim: int = 64,
    save_dir: str = "saved_model",
):
    """
    Full pipeline: preprocess → build vocab → create sequences → train → save.
    Returns: (model, word2idx, idx2word, history_dict, stats_dict)
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not available.")

    # 1. Clean & tokenise
    cleaned = clean_text(text)
    words   = cleaned.split()

    if len(words) < seq_len + 1:
        raise ValueError(f"Text too short. Need at least {seq_len + 1} words.")

    # 2. Vocabulary
    word2idx, idx2word, vocab = build_vocab(words, max_vocab)
    vocab_size = len(vocab)

    # 3. Sequences
    sequences = create_sequences(words, word2idx, seq_len)
    X = sequences[:, :-1]   # all but last word
    y = sequences[:, -1]    # last word (target)

    # 4. Model
    model = build_lstm_model(vocab_size, embed_dim, lstm_units, seq_len)

    # 5. Train
    es = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0,
    )

    # 6. Save artefacts
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, "lstm_model.h5"))
    with open(os.path.join(save_dir, "vocab.pkl"), "wb") as f:
        pickle.dump({"word2idx": word2idx, "idx2word": idx2word, "seq_len": seq_len}, f)

    stats = {
        "vocab_size":     vocab_size,
        "total_words":    len(words),
        "unique_words":   len(set(words)),
        "sequences":      len(sequences),
        "epochs_trained": len(history.history["loss"]),
        "final_loss":     round(history.history["loss"][-1], 4),
        "final_acc":      round(history.history["accuracy"][-1], 4),
    }

    return model, word2idx, idx2word, history.history, stats


# ──────────────────────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────────────────────

def predict_next_words(
    seed_text: str,
    model,
    word2idx: dict,
    idx2word: dict,
    seq_len: int = 10,
    top_k: int = 5,
    temperature: float = 1.0,
):
    """
    Predict top-k next words for a given seed text.
    Returns list of (word, probability) tuples.
    """
    cleaned = clean_text(seed_text)
    words   = cleaned.split()

    # Pad or trim to seq_len
    if len(words) < seq_len:
        words = ["<PAD>"] * (seq_len - len(words)) + words
    else:
        words = words[-seq_len:]

    encoded = np.array([[word2idx.get(w, 1) for w in words]])

    probs = model.predict(encoded, verbose=0)[0]

    # Apply temperature
    probs = np.log(probs + 1e-10) / temperature
    probs = np.exp(probs)
    probs = probs / probs.sum()

    top_indices = np.argsort(probs)[::-1][:top_k]
    return [(idx2word.get(i, "<UNK>"), float(probs[i])) for i in top_indices]


def generate_text(
    seed_text: str,
    model,
    word2idx: dict,
    idx2word: dict,
    seq_len: int = 10,
    num_words: int = 20,
    temperature: float = 0.8,
):
    """Generate a sequence of words auto-regressively."""
    cleaned = clean_text(seed_text)
    result  = cleaned.split()

    for _ in range(num_words):
        preds = predict_next_words(
            " ".join(result), model, word2idx, idx2word,
            seq_len=seq_len, top_k=1, temperature=temperature
        )
        next_word = preds[0][0]
        if next_word in ("<PAD>", "<UNK>"):
            break
        result.append(next_word)

    return " ".join(result)


# ──────────────────────────────────────────────────────────────
# LOAD SAVED MODEL
# ──────────────────────────────────────────────────────────────

def load_saved_model(save_dir: str = "saved_model"):
    """Load a previously trained model and vocab from disk."""
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not available.")

    model_path = os.path.join(save_dir, "lstm_model.h5")
    vocab_path  = os.path.join(save_dir, "vocab.pkl")

    if not os.path.exists(model_path) or not os.path.exists(vocab_path):
        return None, None, None, None

    model = load_model(model_path)
    with open(vocab_path, "rb") as f:
        vocab_data = pickle.load(f)

    return model, vocab_data["word2idx"], vocab_data["idx2word"], vocab_data["seq_len"]
