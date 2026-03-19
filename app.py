"""
app.py — LSTM Predictive Text Analyzer
A complete ML project dashboard built with Streamlit.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import time
import re
import warnings
warnings.filterwarnings("ignore")

# ── Local modules ──
from data_utils import (
    SAMPLE_CORPORA, get_corpus_names, get_corpus_text,
    get_corpus_stats, get_top_words,
)

# ── TensorFlow availability check ──
try:
    import tensorflow as tf
    from model import (
        train_model, predict_next_words,
        generate_text, load_saved_model, clean_text,
    )
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="LSTM Predictive Text",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

.stApp { background: #080812; color: #dcdcf0; }

header[data-testid="stHeader"] { display: none; }

section[data-testid="stSidebar"] {
    background: #0e0e1c !important;
    border-right: 1px solid #1e1e3a;
}
section[data-testid="stSidebar"] * { color: #b0b0d0 !important; }

/* Cards */
.card {
    background: #10101e;
    border: 1px solid #1e1e3a;
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 0.8rem;
    position: relative;
    overflow: hidden;
}
.card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: var(--accent, #00e5ff);
}
.card-label {
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #5a5a80;
    margin-bottom: 0.35rem;
}
.card-value {
    font-size: 1.9rem;
    font-weight: 800;
    color: var(--accent, #00e5ff);
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.1;
}
.card-sub { font-size: 0.76rem; color: #4a4a70; margin-top: 0.25rem; }

/* Page title */
.page-title {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.page-sub { font-size: 0.85rem; color: #4a4a70; letter-spacing: 0.06em; }

/* Section headers */
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #dcdcf0;
    margin: 1.5rem 0 0.8rem;
    padding-left: 0.7rem;
    border-left: 3px solid #00e5ff;
}

/* Prediction chips */
.pred-chip {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 30px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    font-weight: 500;
    margin: 4px;
    cursor: default;
    transition: all 0.2s;
}

/* Word highlight */
.word-highlight {
    display: inline-block;
    padding: 2px 8px;
    background: #00e5ff18;
    border: 1px solid #00e5ff44;
    border-radius: 6px;
    color: #00e5ff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.88rem;
    margin: 2px;
}

/* Generated text box */
.gen-text-box {
    background: #10101e;
    border: 1px solid #1e1e3a;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.88rem;
    color: #c0c0e0;
    line-height: 1.8;
    min-height: 80px;
}

/* Buttons */
.stButton > button {
    background: #00e5ff !important;
    color: #080812 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s !important;
    letter-spacing: 0.04em !important;
}
.stButton > button:hover {
    background: #00cfea !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(0,229,255,0.3) !important;
}

/* Textarea */
textarea {
    background: #10101e !important;
    color: #dcdcf0 !important;
    border: 1px solid #2a2a4a !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88rem !important;
}
textarea:focus {
    border-color: #00e5ff !important;
    box-shadow: 0 0 0 2px rgba(0,229,255,0.15) !important;
}

/* Slider */
div[data-testid="stSlider"] > div { color: #00e5ff !important; }

/* Select */
div[data-testid="stSelectbox"] > div > div {
    background: #10101e !important;
    color: #dcdcf0 !important;
    border-color: #2a2a4a !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #080812; }
::-webkit-scrollbar-thumb { background: #1e1e3a; border-radius: 3px; }

/* Info/warning */
div[data-testid="stInfo"], div[data-testid="stWarning"],
div[data-testid="stSuccess"], div[data-testid="stError"] {
    background: #10101e !important;
    border: 1px solid #1e1e3a !important;
    border-radius: 10px !important;
}

hr { border-color: #1e1e3a !important; }

/* Progress bar */
div[data-testid="stProgress"] > div > div {
    background: #00e5ff !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PLOTLY THEME
# ══════════════════════════════════════════════════════════════
PLOT_CFG = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Syne", color="#7070a0"),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="#151528", linecolor="#151528", zerolinecolor="#151528"),
    yaxis=dict(gridcolor="#151528", linecolor="#151528", zerolinecolor="#151528"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#b0b0d0")),
)

CYAN   = "#00e5ff"
PURPLE = "#a855f7"
GREEN  = "#22d3a5"
ORANGE = "#f97316"


# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
defaults = {
    "model":         None,
    "word2idx":      None,
    "idx2word":      None,
    "seq_len":       10,
    "train_history": None,
    "train_stats":   None,
    "trained":       False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem;'>
        <div style='font-size:1.3rem;font-weight:800;color:#dcdcf0;letter-spacing:-0.5px;'>
            🧠 LSTMind
        </div>
        <div style='font-size:0.68rem;color:#4a4a70;letter-spacing:0.14em;
                    text-transform:uppercase;margin-top:2px;'>
            Predictive Text Engine
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    nav = st.radio(
        "Navigation",
        ["🏠 Home", "📊 Data Explorer", "⚙️ Train Model",
         "🔮 Predict & Generate", "📈 Model Analytics"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    if st.session_state.trained:
        s = st.session_state.train_stats or {}
        st.markdown(f"""
        <div style='background:#0a0a18;border:1px solid #1e1e3a;border-radius:10px;
                    padding:0.9rem;font-size:0.75rem;line-height:1.8;'>
            <div style='color:#00e5ff;font-weight:700;margin-bottom:0.4rem;'>✅ Model Trained</div>
            <div style='color:#5a5a80;'>Vocab size &nbsp;<b style="color:#dcdcf0">{s.get("vocab_size","—")}</b></div>
            <div style='color:#5a5a80;'>Sequences &nbsp;&nbsp;<b style="color:#dcdcf0">{s.get("sequences","—")}</b></div>
            <div style='color:#5a5a80;'>Accuracy &nbsp;&nbsp;&nbsp;<b style="color:#22d3a5">{s.get("final_acc","—")}</b></div>
            <div style='color:#5a5a80;'>Loss &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b style="color:#f97316">{s.get("final_loss","—")}</b></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#0a0a18;border:1px dashed #1e1e3a;border-radius:10px;
                    padding:0.9rem;font-size:0.75rem;color:#4a4a70;text-align:center;'>
            No model trained yet.<br>Go to ⚙️ Train Model.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    if not TF_AVAILABLE:
        st.warning("⚠️ TensorFlow not detected. Training disabled. Predictions use demo mode.")

page = nav.split(" ", 1)[1]


# ══════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════
if page == "Home":
    st.markdown("""
    <div style='margin-bottom:2rem;'>
        <div class='page-title'>LSTM <span style='color:#00e5ff;'>Predictive</span><br>Text Engine</div>
        <div class='page-sub'>MACHINE LEARNING · DEEP LEARNING · NLP · SEQUENCE MODELING</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class='card' style='--accent:#00e5ff;'>
            <div class='card-label'>Model Type</div>
            <div class='card-value' style='font-size:1.2rem;'>LSTM</div>
            <div class='card-sub'>Long Short-Term Memory</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='card' style='--accent:#a855f7;'>
            <div class='card-label'>Task</div>
            <div class='card-value' style='font-size:1.2rem;'>Next Word</div>
            <div class='card-sub'>Sequence prediction</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class='card' style='--accent:#22d3a5;'>
            <div class='card-label'>Corpora</div>
            <div class='card-value'>4</div>
            <div class='card-sub'>Built-in datasets</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class='card' style='--accent:#f97316;'>
            <div class='card-label'>Framework</div>
            <div class='card-value' style='font-size:1.2rem;'>Keras</div>
            <div class='card-sub'>TensorFlow backend</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Architecture diagram using plotly
    st.markdown("<div class='section-title'>LSTM Architecture</div>", unsafe_allow_html=True)

    layers = ["Input\n(Seed Text)", "Embedding\nLayer", "LSTM\nLayer 1", "Dropout\n(0.2)",
              "LSTM\nLayer 2", "Dropout\n(0.2)", "Dense\nSoftmax", "Output\n(Top-K Words)"]
    colors = [CYAN, PURPLE, CYAN, "#555577", CYAN, "#555577", GREEN, ORANGE]

    fig_arch = go.Figure()
    for i, (layer, color) in enumerate(zip(layers, colors)):
        fig_arch.add_trace(go.Scatter(
            x=[i], y=[0],
            mode="markers+text",
            marker=dict(size=55, color=color, opacity=0.85,
                        line=dict(color="#080812", width=3)),
            text=[layer],
            textposition="top center",
            textfont=dict(size=9, color="#dcdcf0"),
            showlegend=False,
            hoverinfo="skip",
        ))
        if i < len(layers) - 1:
            fig_arch.add_annotation(
                x=i + 0.5, y=0,
                ax=i, ay=0,
                xref="x", yref="y", axref="x", ayref="y",
                arrowhead=2, arrowsize=1.2,
                arrowcolor="#2a2a4a", arrowwidth=2,
            )
    fig_arch.update_layout(
        **PLOT_CFG,
        height=200,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.5, 7.5]),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.8, 0.8]),
    )
    st.plotly_chart(fig_arch, use_container_width=True, config={"displayModeBar": False})

    # How it works
    st.markdown("<div class='section-title'>How It Works</div>", unsafe_allow_html=True)
    steps = [
        ("01", "Choose a corpus", "Select from built-in datasets or paste your own text."),
        ("02", "Explore the data", "Analyse word frequency, lexical density, and text stats."),
        ("03", "Train the model", "Configure LSTM hyperparameters and train on your corpus."),
        ("04", "Predict & generate", "Type a seed phrase and get top-K next word predictions."),
        ("05", "Analyse results", "Study training curves, perplexity, and model behaviour."),
    ]
    cols = st.columns(5)
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class='card' style='--accent:#00e5ff;text-align:center;'>
                <div style='font-family:"JetBrains Mono",monospace;font-size:1.6rem;
                            font-weight:800;color:#1e1e3a;'>{num}</div>
                <div style='font-size:0.8rem;font-weight:700;color:#dcdcf0;margin:0.3rem 0;'>{title}</div>
                <div style='font-size:0.72rem;color:#4a4a70;line-height:1.5;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: DATA EXPLORER
# ══════════════════════════════════════════════════════════════
elif page == "Data Explorer":
    st.markdown("<div class='page-title'>Data <span style='color:#00e5ff;'>Explorer</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>CORPUS ANALYSIS · VOCABULARY · WORD FREQUENCY</div><br>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📚 Sample Corpora", "✏️ Custom Text"])

    with tab1:
        corpus_choice = st.selectbox("Select a corpus", get_corpus_names())
        text = get_corpus_text(corpus_choice)

    with tab2:
        custom_text = st.text_area(
            "Paste your own text here",
            placeholder="Paste any text — articles, books, chat logs, code comments...",
            height=180,
            key="custom_corpus"
        )
        text = custom_text if custom_text.strip() else text

    # Save selected text to session
    st.session_state["selected_text"] = text

    if text.strip():
        stats = get_corpus_stats(text)

        s1, s2, s3, s4, s5 = st.columns(5)
        with s1:
            st.markdown(f"""<div class='card' style='--accent:{CYAN};'>
                <div class='card-label'>Total Words</div>
                <div class='card-value'>{stats["total_words"]:,}</div>
            </div>""", unsafe_allow_html=True)
        with s2:
            st.markdown(f"""<div class='card' style='--accent:{PURPLE};'>
                <div class='card-label'>Unique Words</div>
                <div class='card-value'>{stats["unique_words"]:,}</div>
            </div>""", unsafe_allow_html=True)
        with s3:
            st.markdown(f"""<div class='card' style='--accent:{GREEN};'>
                <div class='card-label'>Sentences</div>
                <div class='card-value'>{stats["sentences"]}</div>
            </div>""", unsafe_allow_html=True)
        with s4:
            st.markdown(f"""<div class='card' style='--accent:{ORANGE};'>
                <div class='card-label'>Avg Word Len</div>
                <div class='card-value'>{stats["avg_word_len"]}</div>
            </div>""", unsafe_allow_html=True)
        with s5:
            st.markdown(f"""<div class='card' style='--accent:#ec4899;'>
                <div class='card-label'>Lexical Density</div>
                <div class='card-value'>{stats["lexical_density"]}%</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        left, right = st.columns([3, 2])

        with left:
            st.markdown("<div class='section-title'>Top 20 Keywords</div>", unsafe_allow_html=True)
            top_words = get_top_words(text, 20)
            words_df  = pd.DataFrame(top_words, columns=["Word", "Count"])

            fig_bar = go.Figure(go.Bar(
                x=words_df["Count"],
                y=words_df["Word"],
                orientation="h",
                marker=dict(
                    color=words_df["Count"],
                    colorscale=[[0, "#1e1e3a"], [1, CYAN]],
                    line=dict(color="#080812", width=0.5),
                ),
                text=words_df["Count"],
                textposition="outside",
                textfont=dict(color="#7070a0", size=10),
            ))
            fig_bar.update_layout(
                **PLOT_CFG, height=420,
                yaxis=dict(autorange="reversed", gridcolor="#0d0d20"),
                xaxis=dict(gridcolor="#0d0d20"),
                title=dict(text="Word Frequency (stopwords removed)",
                           font=dict(color="#dcdcf0", size=13)),
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

        with right:
            st.markdown("<div class='section-title'>Word Length Distribution</div>", unsafe_allow_html=True)
            all_words = re.findall(r"[a-z']+", text.lower())
            lengths   = [len(w) for w in all_words if len(w) > 1]
            len_count = Counter(lengths)
            len_df    = pd.DataFrame(sorted(len_count.items()), columns=["Length", "Count"])

            fig_len = go.Figure(go.Bar(
                x=len_df["Length"], y=len_df["Count"],
                marker=dict(color=PURPLE, opacity=0.85, line=dict(color="#080812", width=0.5)),
            ))
            fig_len.update_layout(
                **PLOT_CFG, height=250,
                title=dict(text="Word Length Distribution",
                           font=dict(color="#dcdcf0", size=13)),
                xaxis_title="Characters", yaxis_title="Count",
            )
            st.plotly_chart(fig_len, use_container_width=True, config={"displayModeBar": False})

            st.markdown("<div class='section-title'>Corpus Preview</div>", unsafe_allow_html=True)
            preview = text[:600] + ("..." if len(text) > 600 else "")
            st.markdown(
                f"<div class='gen-text-box' style='font-size:0.78rem;'>{preview}</div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════
# PAGE: TRAIN MODEL
# ══════════════════════════════════════════════════════════════
elif page == "Train Model":
    st.markdown("<div class='page-title'>Train <span style='color:#00e5ff;'>Model</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>HYPERPARAMETER CONFIGURATION · LSTM TRAINING</div><br>", unsafe_allow_html=True)

    if not TF_AVAILABLE:
        st.error("TensorFlow is not available in this environment. Training is disabled. "
                 "You can still explore the app in demo mode.")
        st.stop()

    # ── Corpus selection ──
    st.markdown("<div class='section-title'>1. Select Training Corpus</div>", unsafe_allow_html=True)
    col_src1, col_src2 = st.columns([1, 1])
    with col_src1:
        corpus_src = st.radio("Source", ["Use sample corpus", "Use custom text"],
                              horizontal=True, label_visibility="collapsed")
    with col_src2:
        corpus_name = st.selectbox("Corpus", get_corpus_names(), label_visibility="collapsed")

    if corpus_src == "Use sample corpus":
        train_text = get_corpus_text(corpus_name)
    else:
        train_text = st.session_state.get("selected_text", get_corpus_text(corpus_name))

    stats = get_corpus_stats(train_text)
    st.markdown(
        f"<div style='font-size:0.78rem;color:#4a4a70;margin-bottom:1rem;'>"
        f"Corpus: <b style='color:#dcdcf0;'>{stats['total_words']} words</b> · "
        f"<b style='color:#dcdcf0;'>{stats['unique_words']} unique</b></div>",
        unsafe_allow_html=True,
    )

    # ── Hyperparameters ──
    st.markdown("<div class='section-title'>2. Configure Hyperparameters</div>", unsafe_allow_html=True)

    h1, h2, h3 = st.columns(3)
    with h1:
        seq_len    = st.slider("Sequence Length",    5, 20, 10,
                               help="How many previous words the model looks at")
        embed_dim  = st.select_slider("Embedding Dim", [32, 64, 128, 256], value=64,
                                      help="Word vector dimensions")
    with h2:
        lstm_units = st.select_slider("LSTM Units",    [64, 128, 256], value=128,
                                      help="Hidden units in each LSTM layer")
        max_vocab  = st.slider("Max Vocabulary",   500, 5000, 2000, step=500,
                               help="Maximum number of unique words to learn")
    with h3:
        epochs     = st.slider("Max Epochs",       5, 50, 20,
                               help="Training stops early if loss stops improving")
        batch_size = st.select_slider("Batch Size",    [32, 64, 128], value=64)

    # Estimated complexity
    est_seqs = max(0, stats["total_words"] - seq_len)
    st.markdown(
        f"<div style='background:#0a0a18;border:1px solid #1e1e3a;border-radius:8px;"
        f"padding:0.6rem 1rem;font-size:0.78rem;color:#5a5a80;margin-bottom:1rem;'>"
        f"Estimated sequences: <b style='color:{CYAN};'>{est_seqs:,}</b> · "
        f"Parameters ≈ <b style='color:{PURPLE};'>"
        f"{min(max_vocab, stats['unique_words']) * embed_dim + lstm_units * (embed_dim + lstm_units) * 4:,}</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Train button ──
    st.markdown("<div class='section-title'>3. Train</div>", unsafe_allow_html=True)
    if st.button("🚀 Start Training", use_container_width=False):
        progress_bar  = st.progress(0)
        status_text   = st.empty()
        status_text.markdown(
            "<div style='color:#00e5ff;font-size:0.85rem;'>Initialising...</div>",
            unsafe_allow_html=True,
        )
        try:
            with st.spinner(""):
                start = time.time()
                status_text.markdown(
                    "<div style='color:#00e5ff;font-size:0.85rem;'>Building vocabulary & sequences...</div>",
                    unsafe_allow_html=True,
                )
                progress_bar.progress(15)

                model, word2idx, idx2word, history, stats_out = train_model(
                    text=train_text,
                    seq_len=seq_len,
                    epochs=epochs,
                    batch_size=batch_size,
                    max_vocab=max_vocab,
                    lstm_units=lstm_units,
                    embed_dim=embed_dim,
                )
                progress_bar.progress(100)
                elapsed = round(time.time() - start, 1)

            st.session_state.model         = model
            st.session_state.word2idx      = word2idx
            st.session_state.idx2word      = idx2word
            st.session_state.seq_len       = seq_len
            st.session_state.train_history = history
            st.session_state.train_stats   = stats_out
            st.session_state.trained       = True

            status_text.empty()
            progress_bar.empty()

            st.success(f"✅ Training complete in {elapsed}s!")

            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.markdown(f"""<div class='card' style='--accent:{CYAN};'>
                    <div class='card-label'>Vocab Size</div>
                    <div class='card-value'>{stats_out['vocab_size']:,}</div>
                </div>""", unsafe_allow_html=True)
            with r2:
                st.markdown(f"""<div class='card' style='--accent:{PURPLE};'>
                    <div class='card-label'>Sequences</div>
                    <div class='card-value'>{stats_out['sequences']:,}</div>
                </div>""", unsafe_allow_html=True)
            with r3:
                st.markdown(f"""<div class='card' style='--accent:{GREEN};'>
                    <div class='card-label'>Final Accuracy</div>
                    <div class='card-value'>{stats_out['final_acc']:.1%}</div>
                </div>""", unsafe_allow_html=True)
            with r4:
                st.markdown(f"""<div class='card' style='--accent:{ORANGE};'>
                    <div class='card-label'>Final Loss</div>
                    <div class='card-value'>{stats_out['final_loss']}</div>
                </div>""", unsafe_allow_html=True)

        except Exception as e:
            progress_bar.empty()
            st.error(f"Training failed: {e}")


# ══════════════════════════════════════════════════════════════
# PAGE: PREDICT & GENERATE
# ══════════════════════════════════════════════════════════════
elif page == "Predict & Generate":
    st.markdown("<div class='page-title'>Predict <span style='color:#00e5ff;'>&</span> Generate</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>NEXT WORD PREDICTION · TEXT GENERATION · TEMPERATURE SAMPLING</div><br>", unsafe_allow_html=True)

    # ── Demo mode when TF not available ──
    def demo_predict(seed, top_k=5, temperature=1.0):
        demo_words  = ["the", "machine", "learning", "model", "data",
                       "neural", "network", "language", "text", "deep",
                       "training", "prediction", "sequence", "word", "input"]
        random_probs = np.random.dirichlet(np.ones(top_k) * (2.0 / temperature))
        chosen = np.random.choice(demo_words, size=min(top_k, len(demo_words)), replace=False)
        return list(zip(chosen, sorted(random_probs, reverse=True)))

    def demo_generate(seed, num_words=20, temperature=1.0):
        demo_words = ["the", "machine", "learning", "model", "predicts",
                      "next", "word", "based", "on", "training", "data",
                      "neural", "networks", "understand", "language"]
        result = seed.lower().split()
        for _ in range(num_words):
            result.append(np.random.choice(demo_words))
        return " ".join(result)

    model_ready = st.session_state.trained and st.session_state.model is not None

    if not model_ready and not TF_AVAILABLE:
        st.info("🔮 Running in **Demo Mode** — predictions are illustrative. "
                "Train a model for real results.")

    # ── Layout ──
    pred_col, gen_col = st.columns([1, 1], gap="large")

    # ── NEXT WORD PREDICTION ──
    with pred_col:
        st.markdown("<div class='section-title'>🔮 Next Word Prediction</div>", unsafe_allow_html=True)

        seed_input = st.text_area(
            "Seed Text",
            placeholder="Type a phrase, e.g. 'the machine learning model'",
            height=100,
            key="seed_pred",
        )

        p1, p2 = st.columns([1, 1])
        with p1:
            top_k = st.slider("Top K predictions", 3, 10, 5)
        with p2:
            temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1,
                                    help="Low=confident, High=creative")

        predict_btn = st.button("🔍 Predict Next Word", use_container_width=True)

        if predict_btn:
            if not seed_input.strip():
                st.error("Please enter some seed text.")
            else:
                with st.spinner("Predicting..."):
                    if model_ready:
                        preds = predict_next_words(
                            seed_input,
                            st.session_state.model,
                            st.session_state.word2idx,
                            st.session_state.idx2word,
                            seq_len=st.session_state.seq_len,
                            top_k=top_k,
                            temperature=temperature,
                        )
                    else:
                        preds = demo_predict(seed_input, top_k, temperature)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<div style='font-size:0.72rem;color:#4a4a70;text-transform:uppercase;"
                            "letter-spacing:0.12em;margin-bottom:0.6rem;'>Predictions</div>",
                            unsafe_allow_html=True)

                # Chip display
                max_prob = preds[0][1] if preds else 1
                chips_html = ""
                for word, prob in preds:
                    intensity = int(30 + (prob / max_prob) * 70)
                    chips_html += (
                        f"<span class='pred-chip' "
                        f"style='background:rgba(0,229,255,{prob/max_prob*0.25:.2f});"
                        f"border:1px solid rgba(0,229,255,{prob/max_prob*0.5:.2f});"
                        f"color:#00e5ff;'>"
                        f"{word} <span style='opacity:0.5;font-size:0.72rem;'>"
                        f"{prob:.1%}</span></span>"
                    )
                st.markdown(chips_html, unsafe_allow_html=True)

                # Bar chart of probabilities
                words_list = [p[0] for p in preds]
                probs_list  = [p[1] for p in preds]

                fig_pred = go.Figure(go.Bar(
                    x=probs_list, y=words_list, orientation="h",
                    marker=dict(
                        color=probs_list,
                        colorscale=[[0, "#1e1e3a"], [1, CYAN]],
                        line=dict(color="#080812", width=0.5),
                    ),
                    text=[f"{p:.1%}" for p in probs_list],
                    textposition="outside",
                    textfont=dict(color="#7070a0", size=10),
                ))
                fig_pred.update_layout(
                    **PLOT_CFG, height=260,
                    yaxis=dict(autorange="reversed", gridcolor="#0d0d20"),
                    xaxis=dict(gridcolor="#0d0d20", tickformat=".0%"),
                    title=dict(text="Prediction Probabilities",
                               font=dict(color="#dcdcf0", size=12)),
                )
                st.plotly_chart(fig_pred, use_container_width=True,
                                config={"displayModeBar": False})

    # ── TEXT GENERATION ──
    with gen_col:
        st.markdown("<div class='section-title'>✍️ Text Generation</div>", unsafe_allow_html=True)

        seed_gen = st.text_area(
            "Generation Seed",
            placeholder="e.g. 'artificial intelligence is'",
            height=100,
            key="seed_gen",
        )

        g1, g2 = st.columns([1, 1])
        with g1:
            num_words   = st.slider("Words to generate", 10, 100, 30)
        with g2:
            temp_gen    = st.slider("Temperature ", 0.1, 2.0, 0.8, 0.1,
                                    key="temp_gen",
                                    help="Low=predictable, High=creative/random")

        gen_btn = st.button("✨ Generate Text", use_container_width=True)

        if gen_btn:
            if not seed_gen.strip():
                st.error("Please enter a seed phrase.")
            else:
                with st.spinner("Generating..."):
                    if model_ready:
                        generated = generate_text(
                            seed_gen,
                            st.session_state.model,
                            st.session_state.word2idx,
                            st.session_state.idx2word,
                            seq_len=st.session_state.seq_len,
                            num_words=num_words,
                            temperature=temp_gen,
                        )
                    else:
                        generated = demo_generate(seed_gen, num_words, temp_gen)

                # Highlight seed words vs generated
                seed_words = seed_gen.lower().split()
                gen_words  = generated.split()

                highlighted = ""
                for i, word in enumerate(gen_words):
                    if i < len(seed_words):
                        highlighted += (
                            f"<span style='color:{CYAN};font-weight:700;'>{word}</span> "
                        )
                    else:
                        highlighted += f"{word} "

                st.markdown(
                    f"<div class='gen-text-box'>{highlighted.strip()}</div>",
                    unsafe_allow_html=True,
                )

                # Token timeline
                st.markdown("<br>", unsafe_allow_html=True)
                gen_only = gen_words[len(seed_words):]
                if gen_only:
                    fig_tok = go.Figure(go.Scatter(
                        x=list(range(len(gen_only))),
                        y=[len(w) for w in gen_only],
                        mode="lines+markers+text",
                        text=gen_only,
                        textposition="top center",
                        textfont=dict(size=9, color="#7070a0"),
                        line=dict(color=PURPLE, width=2),
                        marker=dict(color=PURPLE, size=8,
                                    line=dict(color="#080812", width=1.5)),
                        fill="tozeroy", fillcolor="rgba(168,85,247,0.06)",
                    ))
                    fig_tok.update_layout(
                        **PLOT_CFG, height=200,
                        title=dict(text="Generated Token Timeline (word length)",
                                   font=dict(color="#dcdcf0", size=12)),
                        xaxis_title="Token position",
                        yaxis_title="Char length",
                    )
                    st.plotly_chart(fig_tok, use_container_width=True,
                                    config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════
# PAGE: MODEL ANALYTICS
# ══════════════════════════════════════════════════════════════
elif page == "Model Analytics":
    st.markdown("<div class='page-title'>Model <span style='color:#00e5ff;'>Analytics</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>TRAINING CURVES · PERPLEXITY · VOCABULARY ANALYSIS</div><br>", unsafe_allow_html=True)

    if not st.session_state.trained or st.session_state.train_history is None:
        st.info("No model trained yet. Go to ⚙️ Train Model first.")
        st.stop()

    history = st.session_state.train_history
    stats   = st.session_state.train_stats
    epochs_run = list(range(1, len(history["loss"]) + 1))

    # ── KPI row ──
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f"""<div class='card' style='--accent:{CYAN};'>
            <div class='card-label'>Epochs Run</div>
            <div class='card-value'>{stats["epochs_trained"]}</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class='card' style='--accent:{GREEN};'>
            <div class='card-label'>Final Accuracy</div>
            <div class='card-value'>{stats["final_acc"]:.1%}</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class='card' style='--accent:{ORANGE};'>
            <div class='card-label'>Final Loss</div>
            <div class='card-value'>{stats["final_loss"]}</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        perplexity = round(np.exp(stats["final_loss"]), 2)
        st.markdown(f"""<div class='card' style='--accent:{PURPLE};'>
            <div class='card-label'>Perplexity</div>
            <div class='card-value'>{perplexity}</div>
            <div class='card-sub'>exp(loss)</div>
        </div>""", unsafe_allow_html=True)
    with k5:
        st.markdown(f"""<div class='card' style='--accent:#ec4899;'>
            <div class='card-label'>Vocab Size</div>
            <div class='card-value'>{stats["vocab_size"]:,}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Training curves ──
    left, right = st.columns([3, 2])

    with left:
        fig_curves = go.Figure()
        fig_curves.add_trace(go.Scatter(
            x=epochs_run, y=history["loss"],
            name="Loss", mode="lines+markers",
            line=dict(color=ORANGE, width=2.5),
            marker=dict(size=7, color=ORANGE, line=dict(color="#080812", width=1.5)),
            yaxis="y",
        ))
        fig_curves.add_trace(go.Scatter(
            x=epochs_run, y=history["accuracy"],
            name="Accuracy", mode="lines+markers",
            line=dict(color=GREEN, width=2.5),
            marker=dict(size=7, color=GREEN, line=dict(color="#080812", width=1.5)),
            yaxis="y2",
        ))
        fig_curves.update_layout(
            **PLOT_CFG, height=340,
            title=dict(text="Training Loss & Accuracy per Epoch",
                       font=dict(color="#dcdcf0", size=14)),
            yaxis=dict(title="Loss", titlefont=dict(color=ORANGE),
                       tickfont=dict(color=ORANGE), gridcolor="#0d0d20"),
            yaxis2=dict(title="Accuracy", titlefont=dict(color=GREEN),
                        tickfont=dict(color=GREEN), overlaying="y",
                        side="right", gridcolor="#0d0d20", tickformat=".0%"),
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig_curves, use_container_width=True,
                        config={"displayModeBar": False})

    with right:
        # Perplexity curve
        perplexities = [round(np.exp(l), 2) for l in history["loss"]]
        fig_perp = go.Figure(go.Scatter(
            x=epochs_run, y=perplexities,
            mode="lines+markers",
            line=dict(color=CYAN, width=2.5),
            marker=dict(size=7, color=CYAN, line=dict(color="#080812", width=1.5)),
            fill="tozeroy", fillcolor=f"rgba(0,229,255,0.06)",
            name="Perplexity",
        ))
        fig_perp.update_layout(
            **PLOT_CFG, height=200,
            title=dict(text="Perplexity over Epochs",
                       font=dict(color="#dcdcf0", size=13)),
            yaxis_title="Perplexity",
        )
        st.plotly_chart(fig_perp, use_container_width=True,
                        config={"displayModeBar": False})

        # Training summary table
        summary = pd.DataFrame({
            "Epoch":    epochs_run,
            "Loss":     [round(l, 4) for l in history["loss"]],
            "Accuracy": [f"{a:.1%}" for a in history["accuracy"]],
            "Perplexity": perplexities,
        })
        st.dataframe(
            summary.tail(10).reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

    # ── Vocab analysis ──
    if st.session_state.word2idx:
        st.markdown("<div class='section-title'>Vocabulary Distribution</div>",
                    unsafe_allow_html=True)

        w2i   = st.session_state.word2idx
        words = [w for w in w2i.keys() if w not in ("<PAD>", "<UNK>")]
        lengths = [len(w) for w in words]
        lcount  = Counter(lengths)
        ldf     = pd.DataFrame(sorted(lcount.items()), columns=["Length", "Words"])

        fig_voc = go.Figure(go.Bar(
            x=ldf["Length"], y=ldf["Words"],
            marker=dict(color=PURPLE, opacity=0.85,
                        line=dict(color="#080812", width=0.5)),
        ))
        fig_voc.update_layout(
            **PLOT_CFG, height=240,
            title=dict(text="Vocabulary Word Length Distribution",
                       font=dict(color="#dcdcf0", size=13)),
            xaxis_title="Word Length (chars)",
            yaxis_title="Number of Words",
        )
        st.plotly_chart(fig_voc, use_container_width=True,
                        config={"displayModeBar": False})


# ── FOOTER ──
st.markdown("""
<div style='text-align:center;color:#1e1e3a;font-size:0.72rem;margin-top:3rem;
            padding-top:1rem;border-top:1px solid #10101e;'>
    LSTMind &nbsp;·&nbsp; LSTM Predictive Text Engine &nbsp;·&nbsp;
    Built with Streamlit &amp; TensorFlow &nbsp;·&nbsp;
    <span style='color:#2a2a4a;'>Machine Learning Portfolio Project</span>
</div>
""", unsafe_allow_html=True)
