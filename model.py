"""
model.py
Pure NumPy LSTM implementation for next-word prediction.
No TensorFlow / PyTorch dependency — runs on any Python version.
"""

import numpy as np
import pickle
import os
import re
from collections import Counter


# ──────────────────────────────────────────────────────────────
# TEXT PREPROCESSING
# ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s'.,!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_vocab(words: list, max_vocab: int = 3000):
    freq  = Counter(words)
    vocab = ["<PAD>", "<UNK>"] + [w for w, _ in freq.most_common(max_vocab - 2)]
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word, vocab


def create_sequences(words: list, word2idx: dict, seq_len: int = 10):
    sequences = []
    for i in range(seq_len, len(words)):
        seq = words[i - seq_len: i + 1]
        encoded = [word2idx.get(w, 1) for w in seq]
        sequences.append(encoded)
    return np.array(sequences, dtype=np.int32)


# ──────────────────────────────────────────────────────────────
# NUMPY ACTIVATIONS / HELPERS
# ──────────────────────────────────────────────────────────────

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x):
    return np.tanh(np.clip(x, -500, 500))


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-10)


def cross_entropy(probs, target):
    return -np.log(probs[target] + 1e-10)


# ──────────────────────────────────────────────────────────────
# EMBEDDING LAYER
# ──────────────────────────────────────────────────────────────

class Embedding:
    def __init__(self, vocab_size, embed_dim):
        self.W = np.random.randn(vocab_size, embed_dim) * 0.01
        self.dW = np.zeros_like(self.W)
        self.indices = None

    def forward(self, indices):
        self.indices = indices
        return self.W[indices]          # shape: (seq_len, embed_dim)

    def backward(self, dout):
        self.dW = np.zeros_like(self.W)
        np.add.at(self.dW, self.indices, dout)


# ──────────────────────────────────────────────────────────────
# LSTM CELL (single layer, manual forward + BPTT)
# ──────────────────────────────────────────────────────────────

class LSTMCell:
    def __init__(self, input_dim, hidden_dim):
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))
        # Gates: input, forget, cell, output  (concatenated weight matrix)
        self.Wf = np.random.randn(hidden_dim, input_dim + hidden_dim) * scale
        self.Wi = np.random.randn(hidden_dim, input_dim + hidden_dim) * scale
        self.Wc = np.random.randn(hidden_dim, input_dim + hidden_dim) * scale
        self.Wo = np.random.randn(hidden_dim, input_dim + hidden_dim) * scale
        self.bf = np.zeros(hidden_dim)
        self.bi = np.zeros(hidden_dim)
        self.bc = np.zeros(hidden_dim)
        self.bo = np.zeros(hidden_dim)

        self.hidden_dim  = hidden_dim
        self.cache       = []

    def forward_sequence(self, X, h0=None, c0=None):
        """X: (seq_len, input_dim)"""
        T   = X.shape[0]
        H   = self.hidden_dim
        h   = np.zeros(H) if h0 is None else h0
        c   = np.zeros(H) if c0 is None else c0
        self.cache = []

        hs = np.zeros((T, H))
        for t in range(T):
            x    = X[t]
            xh   = np.concatenate([x, h])

            f    = sigmoid(self.Wf @ xh + self.bf)
            i    = sigmoid(self.Wi @ xh + self.bi)
            g    = tanh   (self.Wc @ xh + self.bc)
            o    = sigmoid(self.Wo @ xh + self.bo)
            c    = f * c + i * g
            h    = o * tanh(c)

            self.cache.append((x, h, c, f, i, g, o, xh))
            hs[t] = h

        self.h_last = h
        self.c_last = c
        return hs, h, c

    def backward_sequence(self, dhs, lr=0.001):
        """Simple truncated BPTT returning gradient w.r.t. inputs."""
        T, H    = dhs.shape
        input_dim = self.cache[0][0].shape[0]

        dWf = np.zeros_like(self.Wf); dbf = np.zeros_like(self.bf)
        dWi = np.zeros_like(self.Wi); dbi = np.zeros_like(self.bi)
        dWc = np.zeros_like(self.Wc); dbc = np.zeros_like(self.bc)
        dWo = np.zeros_like(self.Wo); dbo = np.zeros_like(self.bo)
        dX  = np.zeros((T, input_dim))

        dh_next = np.zeros(H)
        dc_next = np.zeros(H)

        for t in reversed(range(T)):
            x, h, c, f, i, g, o, xh = self.cache[t]
            c_prev = self.cache[t - 1][2] if t > 0 else np.zeros(H)

            dh = dhs[t] + dh_next
            do = dh * tanh(c)
            dc = dh * o * (1 - tanh(c) ** 2) + dc_next
            df = dc * c_prev
            di = dc * g
            dg = dc * i

            d_o_pre = do * o * (1 - o)
            d_f_pre = df * f * (1 - f)
            d_i_pre = di * i * (1 - i)
            d_g_pre = dg * (1 - g ** 2)

            dWo += np.outer(d_o_pre, xh); dbo += d_o_pre
            dWf += np.outer(d_f_pre, xh); dbf += d_f_pre
            dWi += np.outer(d_i_pre, xh); dbi += d_i_pre
            dWc += np.outer(d_g_pre, xh); dbc += d_g_pre

            dxh = (self.Wo.T @ d_o_pre + self.Wf.T @ d_f_pre +
                   self.Wi.T @ d_i_pre + self.Wc.T @ d_g_pre)
            dX[t]   = dxh[:input_dim]
            dh_next = dxh[input_dim:]
            dc_next = dc * f

        # Gradient clipping
        for grad in [dWf, dWi, dWc, dWo, dbf, dbi, dbc, dbo]:
            np.clip(grad, -5, 5, out=grad)

        # SGD update
        self.Wf -= lr * dWf; self.bf -= lr * dbf
        self.Wi -= lr * dWi; self.bi -= lr * dbi
        self.Wc -= lr * dWc; self.bc -= lr * dbc
        self.Wo -= lr * dWo; self.bo -= lr * dbo

        return dX


# ──────────────────────────────────────────────────────────────
# DENSE OUTPUT LAYER
# ──────────────────────────────────────────────────────────────

class DenseLayer:
    def __init__(self, input_dim, output_dim):
        self.W  = np.random.randn(output_dim, input_dim) * 0.01
        self.b  = np.zeros(output_dim)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.x  = None

    def forward(self, x):
        self.x = x
        return self.W @ x + self.b

    def backward(self, dout, lr=0.001):
        self.dW = np.outer(dout, self.x)
        self.db = dout
        np.clip(self.dW, -5, 5, out=self.dW)
        self.W -= lr * self.dW
        self.b -= lr * self.db
        return self.W.T @ dout


# ──────────────────────────────────────────────────────────────
# FULL LSTM MODEL
# ──────────────────────────────────────────────────────────────

class LSTMModel:
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        self.embedding   = Embedding(vocab_size, embed_dim)
        self.lstm        = LSTMCell(embed_dim, hidden_dim)
        self.dense       = DenseLayer(hidden_dim, vocab_size)
        self.vocab_size  = vocab_size
        self.embed_dim   = embed_dim
        self.hidden_dim  = hidden_dim

    def forward(self, indices):
        """indices: (seq_len,) int array"""
        X          = self.embedding.forward(indices)
        hs, h, c   = self.lstm.forward_sequence(X)
        logits     = self.dense.forward(h)
        probs      = softmax(logits)
        return probs, hs, h

    def loss_and_grad(self, indices, target, lr=0.001):
        probs, hs, h = self.forward(indices)
        loss = cross_entropy(probs, target)

        # Backprop through dense
        dlogits       = probs.copy()
        dlogits[target] -= 1.0
        dh            = self.dense.backward(dlogits, lr)

        # Backprop through LSTM (only last hidden state matters)
        dhs           = np.zeros_like(hs)
        dhs[-1]       = dh
        dX            = self.lstm.backward_sequence(dhs, lr)

        # Backprop through embedding
        self.embedding.backward(dX)
        # SGD embedding update
        np.add.at(self.embedding.W, self.embedding.indices,
                  -lr * self.embedding.dW[self.embedding.indices])

        return loss, probs

    def predict(self, indices, temperature=1.0, top_k=5):
        probs, _, _ = self.forward(indices)
        log_probs   = np.log(probs + 1e-10) / temperature
        probs_t     = np.exp(log_probs - np.max(log_probs))
        probs_t    /= probs_t.sum()
        top_idx     = np.argsort(probs_t)[::-1][:top_k]
        return top_idx, probs_t[top_idx]

    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "embed_W":  self.embedding.W,
                "lstm_Wf":  self.lstm.Wf, "lstm_bf": self.lstm.bf,
                "lstm_Wi":  self.lstm.Wi, "lstm_bi": self.lstm.bi,
                "lstm_Wc":  self.lstm.Wc, "lstm_bc": self.lstm.bc,
                "lstm_Wo":  self.lstm.Wo, "lstm_bo": self.lstm.bo,
                "dense_W":  self.dense.W, "dense_b":  self.dense.b,
                "vocab_size": self.vocab_size,
                "embed_dim":  self.embed_dim,
                "hidden_dim": self.hidden_dim,
            }, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        m = cls(d["vocab_size"], d["embed_dim"], d["hidden_dim"])
        m.embedding.W = d["embed_W"]
        m.lstm.Wf = d["lstm_Wf"]; m.lstm.bf = d["lstm_bf"]
        m.lstm.Wi = d["lstm_Wi"]; m.lstm.bi = d["lstm_bi"]
        m.lstm.Wc = d["lstm_Wc"]; m.lstm.bc = d["lstm_bc"]
        m.lstm.Wo = d["lstm_Wo"]; m.lstm.bo = d["lstm_bo"]
        m.dense.W = d["dense_W"]; m.dense.b  = d["dense_b"]
        return m


# ──────────────────────────────────────────────────────────────
# HIGH-LEVEL TRAINING FUNCTION
# ──────────────────────────────────────────────────────────────

def train_model(
    text: str,
    seq_len: int     = 8,
    epochs: int      = 15,
    lr: float        = 0.005,
    max_vocab: int   = 1500,
    embed_dim: int   = 32,
    hidden_dim: int  = 64,
    max_sequences: int = 800,
    progress_cb=None,          # optional callable(epoch, total, loss, acc)
):
    """
    Train LSTM on text. Returns (model, word2idx, idx2word, history, stats).
    Uses pure NumPy — no TensorFlow required.
    """
    cleaned   = clean_text(text)
    words     = cleaned.split()

    if len(words) < seq_len + 1:
        raise ValueError(f"Text too short. Need at least {seq_len + 1} words.")

    word2idx, idx2word, vocab = build_vocab(words, max_vocab)
    vocab_size = len(vocab)

    sequences  = create_sequences(words, word2idx, seq_len)
    # Cap sequences for speed on cloud
    if len(sequences) > max_sequences:
        idx = np.random.choice(len(sequences), max_sequences, replace=False)
        sequences = sequences[idx]

    model = LSTMModel(vocab_size, embed_dim, hidden_dim)

    history = {"loss": [], "accuracy": []}

    for epoch in range(epochs):
        np.random.shuffle(sequences)
        total_loss = 0.0
        correct    = 0

        for seq in sequences:
            x_idx  = seq[:-1]
            target = int(seq[-1])
            loss, probs = model.loss_and_grad(x_idx, target, lr)
            total_loss += loss
            if np.argmax(probs) == target:
                correct += 1

        avg_loss = total_loss / len(sequences)
        acc      = correct   / len(sequences)
        history["loss"].append(round(avg_loss, 4))
        history["accuracy"].append(round(acc, 4))

        if progress_cb:
            progress_cb(epoch + 1, epochs, avg_loss, acc)

    stats = {
        "vocab_size":     vocab_size,
        "total_words":    len(words),
        "unique_words":   len(set(words)),
        "sequences":      len(sequences),
        "epochs_trained": epochs,
        "final_loss":     round(history["loss"][-1], 4),
        "final_acc":      round(history["accuracy"][-1], 4),
    }

    return model, word2idx, idx2word, history, stats


# ──────────────────────────────────────────────────────────────
# INFERENCE HELPERS
# ──────────────────────────────────────────────────────────────

def predict_next_words(seed_text, model, word2idx, idx2word,
                       seq_len=8, top_k=5, temperature=1.0):
    cleaned = clean_text(seed_text).split()
    if len(cleaned) < seq_len:
        cleaned = ["<PAD>"] * (seq_len - len(cleaned)) + cleaned
    else:
        cleaned = cleaned[-seq_len:]

    indices   = np.array([word2idx.get(w, 1) for w in cleaned], dtype=np.int32)
    top_idx, top_probs = model.predict(indices, temperature=temperature, top_k=top_k)
    return [(idx2word.get(i, "<UNK>"), float(p)) for i, p in zip(top_idx, top_probs)]


def generate_text(seed_text, model, word2idx, idx2word,
                  seq_len=8, num_words=20, temperature=0.8):
    cleaned = clean_text(seed_text).split()
    result  = cleaned.copy()

    for _ in range(num_words):
        window = result[-seq_len:] if len(result) >= seq_len else ["<PAD>"] * (seq_len - len(result)) + result
        indices = np.array([word2idx.get(w, 1) for w in window], dtype=np.int32)
        top_idx, _ = model.predict(indices, temperature=temperature, top_k=3)
        next_word   = idx2word.get(int(top_idx[0]), "<UNK>")
        if next_word in ("<PAD>", "<UNK>"):
            next_word = idx2word.get(int(top_idx[1]) if len(top_idx) > 1 else 2, "the")
        result.append(next_word)

    return " ".join(result)
