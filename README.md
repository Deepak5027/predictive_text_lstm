# 🧠 LSTMind — LSTM Predictive Text Engine

A complete **Machine Learning portfolio project** built with Streamlit that demonstrates LSTM-based next-word prediction and text generation.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

---

## 🚀 Features

| Feature | Description |
|---|---|
| 📊 **Data Explorer** | Corpus analysis — word frequency, lexical density, word length distribution |
| ⚙️ **Model Training** | Configure LSTM hyperparameters and train on any text corpus |
| 🔮 **Next Word Prediction** | Top-K predictions with probability scores and temperature sampling |
| ✍️ **Text Generation** | Auto-regressive text generation with highlighted seed vs generated tokens |
| 📈 **Model Analytics** | Training curves, perplexity, vocabulary analysis |

---

## 🧬 Model Architecture

```
Input (Seed Text)
     ↓
Embedding Layer  (64 dim)
     ↓
LSTM Layer 1     (128 units)
     ↓
Dropout (0.2)
     ↓
LSTM Layer 2     (64 units)
     ↓
Dropout (0.2)
     ↓
Dense + Softmax  (vocab_size)
     ↓
Top-K Word Predictions
```

---

## 📁 Project Structure

```
lstm_project/
├── app.py           # Main Streamlit application
├── model.py         # LSTM model — build, train, predict, generate
├── data_utils.py    # Sample corpora, text stats, word frequency helpers
├── requirements.txt # Python dependencies
└── README.md        # This file
```

---

## 🛠️ Tech Stack

- **Framework**: Streamlit
- **Deep Learning**: TensorFlow / Keras
- **ML Utilities**: Scikit-learn, NumPy
- **Visualization**: Plotly
- **Data Processing**: Pandas, NLTK

---

## ⚡ Quick Start (Local)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/lstm_project.git
cd lstm_project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## ☁️ Deploy to Streamlit Cloud

1. Push this folder to a **public GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set `app.py` as main file
4. Click **Deploy** ✅

> **Note**: Streamlit Cloud free tier has memory limits. Use `tensorflow-cpu` in requirements.txt (already set).

---

## 🎛️ Hyperparameters Explained

| Parameter | Default | What it does |
|---|---|---|
| Sequence Length | 10 | How many previous words the model sees |
| Embedding Dim | 64 | Size of word vector representations |
| LSTM Units | 128 | Hidden state size per LSTM layer |
| Max Vocabulary | 2000 | Cap on unique words the model learns |
| Temperature | 1.0 | Low = confident, High = creative/random |
| Batch Size | 64 | Samples per gradient update |

---

## 📊 Key Concepts

**LSTM (Long Short-Term Memory)** — A type of RNN that solves the vanishing gradient problem using three gates (forget, input, output) to remember relevant context across long sequences.

**Perplexity** — `exp(loss)`. Lower is better. Measures how "surprised" the model is by the test data.

**Temperature Sampling** — Controls randomness: `prob = softmax(logits / temperature)`. Lower temperature = greedier predictions.

---

## 📝 License

MIT License — free to use, modify, and distribute.

---

*Built as a Machine Learning portfolio project — LSTM · NLP · Deep Learning · Streamlit*
