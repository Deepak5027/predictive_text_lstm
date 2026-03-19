"""
app.py — LSTM Predictive Text Analyzer
Pure NumPy LSTM — no TensorFlow, works on Python 3.14+
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import re
import time
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

from data_utils import (
    SAMPLE_CORPORA, get_corpus_names, get_corpus_text,
    get_corpus_stats, get_top_words,
)
from model import (
    train_model, predict_next_words,
    generate_text, clean_text,
)

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

.page-title {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.page-sub { font-size: 0.82rem; color: #4a4a70; letter-spacing: 0.06em; }

.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #dcdcf0;
    margin: 1.5rem 0 0.8rem;
    padding-left: 0.7rem;
    border-left: 3px solid #00e5ff;
}

.gen-text-box {
    background: #10101e;
    border: 1px solid #1e1e3a;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.86rem;
    color: #c0c0e0;
    line-height: 1.9;
    min-height: 80px;
    word-break: break-word;
}

.stButton > button {
    background: #00e5ff !important;
    color: #080812 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #00cfea !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(0,229,255,0.3) !important;
}

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

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #080812; }
::-webkit-scrollbar-thumb { background: #1e1e3a; border-radius: 3px; }
hr { border-color: #1e1e3a !important; }

div[data-testid="stInfo"], div[data-testid="stWarning"],
div[data-testid="stSuccess"], div[data-testid="stError"] {
    background: #10101e !important;
    border: 1px solid #1e1e3a !important;
    border-radius: 10px !important;
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
for k, v in {
    "model": None, "word2idx": None, "idx2word": None,
    "seq_len": 8, "train_history": None, "train_stats": None, "trained": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem;'>
        <div style='font-size:1.3rem;font-weight:800;color:#dcdcf0;letter-spacing:-0.5px;'>🧠 LSTMind</div>
        <div style='font-size:0.68rem;color:#4a4a70;letter-spacing:0.14em;text-transform:uppercase;margin-top:2px;'>
            Predictive Text Engine
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    nav = st.radio("Navigation", [
        "🏠 Home", "📊 Data Explorer", "⚙️ Train Model",
        "🔮 Predict & Generate", "📈 Model Analytics"
    ], label_visibility="collapsed")

    st.markdown("---")
    if st.session_state.trained:
        s = st.session_state.train_stats or {}
        st.markdown(f"""
        <div style='background:#0a0a18;border:1px solid #1e1e3a;border-radius:10px;
                    padding:0.9rem;font-size:0.75rem;line-height:1.9;'>
            <div style='color:#00e5ff;font-weight:700;margin-bottom:0.3rem;'>✅ Model Trained</div>
            <div style='color:#5a5a80;'>Vocab &nbsp;&nbsp;&nbsp;<b style="color:#dcdcf0">{s.get("vocab_size","—")}</b></div>
            <div style='color:#5a5a80;'>Sequences &nbsp;<b style="color:#dcdcf0">{s.get("sequences","—")}</b></div>
            <div style='color:#5a5a80;'>Accuracy &nbsp;&nbsp;<b style="color:#22d3a5">{s.get("final_acc","—")}</b></div>
            <div style='color:#5a5a80;'>Loss &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b style="color:#f97316">{s.get("final_loss","—")}</b></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#0a0a18;border:1px dashed #1e1e3a;border-radius:10px;
                    padding:0.9rem;font-size:0.75rem;color:#4a4a70;text-align:center;line-height:1.6;'>
            No model trained yet.<br>Go to ⚙️ Train Model.
        </div>
        """, unsafe_allow_html=True)

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
    cards = [
        (c1, CYAN,   "Model Type",  "LSTM",     "Long Short-Term Memory"),
        (c2, PURPLE, "Task",        "Next Word", "Sequence prediction"),
        (c3, GREEN,  "Corpora",     "4",         "Built-in datasets"),
        (c4, ORANGE, "Framework",   "NumPy",     "No TensorFlow needed"),
    ]
    for col, accent, label, val, sub in cards:
        with col:
            st.markdown(f"""<div class='card' style='--accent:{accent};'>
                <div class='card-label'>{label}</div>
                <div class='card-value' style='font-size:1.2rem;'>{val}</div>
                <div class='card-sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Architecture diagram
    st.markdown("<div class='section-title'>LSTM Architecture</div>", unsafe_allow_html=True)
    layers = ["Input\nSeed Text", "Embedding\nLayer", "LSTM\nCell", "Hidden\nState h", "Dense\nLayer", "Softmax", "Top-K\nPredictions"]
    colors = [CYAN, PURPLE, CYAN, GREEN, CYAN, ORANGE, GREEN]

    fig_arch = go.Figure()
    for i, (layer, color) in enumerate(zip(layers, colors)):
        fig_arch.add_trace(go.Scatter(
            x=[i], y=[0], mode="markers+text",
            marker=dict(size=54, color=color, opacity=0.8, line=dict(color="#080812", width=3)),
            text=[layer], textposition="top center",
            textfont=dict(size=9, color="#dcdcf0"),
            showlegend=False, hoverinfo="skip",
        ))
        if i < len(layers) - 1:
            fig_arch.add_annotation(
                x=i+0.5, y=0, ax=i, ay=0,
                xref="x", yref="y", axref="x", ayref="y",
                arrowhead=2, arrowsize=1.2, arrowcolor="#2a2a4a", arrowwidth=2,
            )
    arch_layout = {k: v for k, v in PLOT_CFG.items() if k not in ("xaxis", "yaxis")}
    arch_layout["height"] = 200
    fig_arch.update_layout(**arch_layout)
    fig_arch.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, range=[-0.5, 6.5])
    fig_arch.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, range=[-0.8, 0.8])
    st.plotly_chart(fig_arch, use_container_width=True, config={"displayModeBar": False})

    # Steps
    st.markdown("<div class='section-title'>How It Works</div>", unsafe_allow_html=True)
    steps = [
        ("01", "Choose Corpus",  "Pick a built-in dataset or paste your own text."),
        ("02", "Explore Data",   "Analyse word frequency, lexical density, text stats."),
        ("03", "Train LSTM",     "Set hyperparameters and train the NumPy LSTM model."),
        ("04", "Predict",        "Enter a seed phrase and get top-K next word predictions."),
        ("05", "Analyse",        "Study training loss, accuracy, perplexity curves."),
    ]
    cols = st.columns(5)
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""<div class='card' style='--accent:{CYAN};text-align:center;'>
                <div style='font-family:"JetBrains Mono",monospace;font-size:1.5rem;font-weight:800;color:#1a1a35;'>{num}</div>
                <div style='font-size:0.8rem;font-weight:700;color:#dcdcf0;margin:0.3rem 0;'>{title}</div>
                <div style='font-size:0.72rem;color:#4a4a70;line-height:1.5;'>{desc}</div>
            </div>""", unsafe_allow_html=True)


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
        custom_text = st.text_area("Paste your own text", height=180, key="custom_corpus",
                                   placeholder="Paste any text — articles, books, chat logs...")
        text = custom_text.strip() if custom_text.strip() else text

    st.session_state["selected_text"] = text

    if text.strip():
        stats = get_corpus_stats(text)
        cols  = st.columns(5)
        kpis  = [
            ("Total Words",     stats["total_words"],     CYAN),
            ("Unique Words",    stats["unique_words"],    PURPLE),
            ("Sentences",       stats["sentences"],       GREEN),
            ("Avg Word Len",    stats["avg_word_len"],    ORANGE),
            ("Lexical Density", f"{stats['lexical_density']}%", "#ec4899"),
        ]
        for col, (label, val, accent) in zip(cols, kpis):
            with col:
                st.markdown(f"""<div class='card' style='--accent:{accent};'>
                    <div class='card-label'>{label}</div>
                    <div class='card-value'>{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        left, right = st.columns([3, 2])

        with left:
            st.markdown("<div class='section-title'>Top 20 Keywords</div>", unsafe_allow_html=True)
            top_words = get_top_words(text, 20)
            words_df  = pd.DataFrame(top_words, columns=["Word", "Count"])
            fig_bar = go.Figure(go.Bar(
                x=words_df["Count"], y=words_df["Word"], orientation="h",
                marker=dict(color=words_df["Count"],
                            colorscale=[[0, "#1e1e3a"], [1, CYAN]],
                            line=dict(color="#080812", width=0.5)),
                text=words_df["Count"], textposition="outside",
                textfont=dict(color="#7070a0", size=10),
            ))
            fig_bar.update_layout(
                **PLOT_CFG, height=420,
                yaxis=dict(autorange="reversed", gridcolor="#0d0d20"),
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
                **PLOT_CFG, height=240,
                title=dict(text="Word Length Distribution", font=dict(color="#dcdcf0", size=13)),
                xaxis_title="Characters", yaxis_title="Count",
            )
            st.plotly_chart(fig_len, use_container_width=True, config={"displayModeBar": False})

            st.markdown("<div class='section-title'>Corpus Preview</div>", unsafe_allow_html=True)
            preview = text[:500] + ("..." if len(text) > 500 else "")
            st.markdown(f"<div class='gen-text-box' style='font-size:0.76rem;'>{preview}</div>",
                        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: TRAIN MODEL
# ══════════════════════════════════════════════════════════════
elif page == "Train Model":
    st.markdown("<div class='page-title'>Train <span style='color:#00e5ff;'>Model</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>HYPERPARAMETER CONFIGURATION · NUMPY LSTM TRAINING</div><br>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>1. Select Training Corpus</div>", unsafe_allow_html=True)
    src_col, name_col = st.columns([1, 1])
    with src_col:
        corpus_src = st.radio("Source", ["Sample corpus", "Custom text"],
                              horizontal=True, label_visibility="collapsed")
    with name_col:
        corpus_name = st.selectbox("Corpus", get_corpus_names(), label_visibility="collapsed")

    train_text = (get_corpus_text(corpus_name) if corpus_src == "Sample corpus"
                  else st.session_state.get("selected_text", get_corpus_text(corpus_name)))
    cstats = get_corpus_stats(train_text)
    st.markdown(
        f"<div style='font-size:0.78rem;color:#4a4a70;margin-bottom:1rem;'>"
        f"Corpus: <b style='color:#dcdcf0;'>{cstats['total_words']} words</b> · "
        f"<b style='color:#dcdcf0;'>{cstats['unique_words']} unique</b></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-title'>2. Configure Hyperparameters</div>", unsafe_allow_html=True)
    h1, h2, h3 = st.columns(3)
    with h1:
        seq_len    = st.slider("Sequence Length",  4, 15, 8,
                               help="How many previous words the model sees at once")
        embed_dim  = st.select_slider("Embedding Dim", [16, 32, 64], value=32,
                                      help="Word vector size — larger = richer but slower")
    with h2:
        hidden_dim = st.select_slider("LSTM Hidden Units", [32, 64, 128], value=64,
                                      help="LSTM memory size")
        max_vocab  = st.slider("Max Vocabulary", 300, 2000, 1000, step=100,
                               help="Max unique words the model learns")
    with h3:
        epochs     = st.slider("Epochs",         5, 30, 15,
                               help="Full passes through the training data")
        lr         = st.select_slider("Learning Rate", [0.001, 0.003, 0.005, 0.01], value=0.005)
        max_seq    = st.slider("Max Sequences",  200, 1000, 600, step=100,
                               help="Cap training sequences for speed")

    st.markdown(
        f"<div style='background:#0a0a18;border:1px solid #1e1e3a;border-radius:8px;"
        f"padding:0.6rem 1rem;font-size:0.78rem;color:#5a5a80;margin-bottom:1rem;'>"
        f"Estimated sequences: <b style='color:{CYAN};'>"
        f"{min(max_seq, max(0, cstats['total_words'] - seq_len)):,}</b> &nbsp;·&nbsp; "
        f"No TensorFlow needed — trains with pure NumPy ✅</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-title'>3. Train</div>", unsafe_allow_html=True)
    if st.button("🚀 Start Training", use_container_width=False):
        progress_bar = st.progress(0)
        status_box   = st.empty()
        epoch_log    = st.empty()

        def on_progress(epoch, total, loss, acc):
            pct = int(epoch / total * 100)
            progress_bar.progress(pct)
            status_box.markdown(
                f"<div style='font-size:0.82rem;color:{CYAN};'>"
                f"Epoch {epoch}/{total} &nbsp;·&nbsp; "
                f"Loss: <b>{loss:.4f}</b> &nbsp;·&nbsp; "
                f"Acc: <b>{acc:.1%}</b></div>",
                unsafe_allow_html=True,
            )

        try:
            start = time.time()
            model, word2idx, idx2word, history, stats_out = train_model(
                text=train_text,
                seq_len=seq_len,
                epochs=epochs,
                lr=lr,
                max_vocab=max_vocab,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                max_sequences=max_seq,
                progress_cb=on_progress,
            )
            elapsed = round(time.time() - start, 1)
            progress_bar.progress(100)
            status_box.empty()
            epoch_log.empty()

            st.session_state.model         = model
            st.session_state.word2idx      = word2idx
            st.session_state.idx2word      = idx2word
            st.session_state.seq_len       = seq_len
            st.session_state.train_history = history
            st.session_state.train_stats   = stats_out
            st.session_state.trained       = True

            st.success(f"✅ Training complete in {elapsed}s!")

            r1, r2, r3, r4 = st.columns(4)
            result_cards = [
                (r1, CYAN,   "Vocab Size",      f"{stats_out['vocab_size']:,}"),
                (r2, PURPLE, "Sequences Used",  f"{stats_out['sequences']:,}"),
                (r3, GREEN,  "Final Accuracy",  f"{stats_out['final_acc']:.1%}"),
                (r4, ORANGE, "Final Loss",      f"{stats_out['final_loss']}"),
            ]
            for col, accent, label, val in result_cards:
                with col:
                    st.markdown(f"""<div class='card' style='--accent:{accent};'>
                        <div class='card-label'>{label}</div>
                        <div class='card-value'>{val}</div>
                    </div>""", unsafe_allow_html=True)

        except Exception as e:
            progress_bar.empty()
            status_box.empty()
            st.error(f"Training error: {e}")


# ══════════════════════════════════════════════════════════════
# PAGE: PREDICT & GENERATE
# ══════════════════════════════════════════════════════════════
elif page == "Predict & Generate":
    st.markdown("<div class='page-title'>Predict <span style='color:#00e5ff;'>&</span> Generate</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>NEXT WORD PREDICTION · TEXT GENERATION · TEMPERATURE SAMPLING</div><br>", unsafe_allow_html=True)

    if not st.session_state.trained:
        st.warning("⚠️ No model trained yet. Please go to ⚙️ Train Model first.")
        st.stop()

    pred_col, gen_col = st.columns([1, 1], gap="large")

    # ── PREDICTION ──
    with pred_col:
        st.markdown("<div class='section-title'>🔮 Next Word Prediction</div>", unsafe_allow_html=True)

        seed_input = st.text_area("Seed Text",
                                  placeholder="e.g. 'machine learning is'",
                                  height=100, key="seed_pred")
        p1, p2 = st.columns(2)
        with p1:
            top_k       = st.slider("Top K predictions", 3, 10, 5)
        with p2:
            temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1,
                                    help="Low = confident, High = creative")

        if st.button("🔍 Predict Next Word", use_container_width=True):
            if not seed_input.strip():
                st.error("Please enter some seed text.")
            else:
                with st.spinner("Predicting..."):
                    preds = predict_next_words(
                        seed_input,
                        st.session_state.model,
                        st.session_state.word2idx,
                        st.session_state.idx2word,
                        seq_len=st.session_state.seq_len,
                        top_k=top_k,
                        temperature=temperature,
                    )

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    "<div style='font-size:0.7rem;color:#4a4a70;text-transform:uppercase;"
                    "letter-spacing:0.12em;margin-bottom:0.6rem;'>Top Predictions</div>",
                    unsafe_allow_html=True,
                )

                max_prob = preds[0][1] if preds else 1
                chips    = ""
                for word, prob in preds:
                    alpha = prob / max_prob
                    chips += (
                        f"<span style='display:inline-block;padding:5px 14px;border-radius:25px;"
                        f"background:rgba(0,229,255,{alpha*0.2:.2f});"
                        f"border:1px solid rgba(0,229,255,{alpha*0.5:.2f});"
                        f"color:#00e5ff;font-family:JetBrains Mono,monospace;"
                        f"font-size:0.82rem;margin:3px;'>"
                        f"{word} <span style='opacity:0.5;'>{prob:.1%}</span></span>"
                    )
                st.markdown(chips, unsafe_allow_html=True)

                words_l = [p[0] for p in preds]
                probs_l = [p[1] for p in preds]
                fig_pred = go.Figure(go.Bar(
                    x=probs_l, y=words_l, orientation="h",
                    marker=dict(color=probs_l,
                                colorscale=[[0, "#1e1e3a"], [1, CYAN]],
                                line=dict(color="#080812", width=0.5)),
                    text=[f"{p:.1%}" for p in probs_l], textposition="outside",
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

    # ── GENERATION ──
    with gen_col:
        st.markdown("<div class='section-title'>✍️ Text Generation</div>", unsafe_allow_html=True)

        seed_gen = st.text_area("Generation Seed",
                                placeholder="e.g. 'artificial intelligence is'",
                                height=100, key="seed_gen")
        g1, g2 = st.columns(2)
        with g1:
            num_words = st.slider("Words to generate", 10, 80, 25)
        with g2:
            temp_gen  = st.slider("Temperature ", 0.1, 2.0, 0.8, 0.1, key="temp_gen")

        if st.button("✨ Generate Text", use_container_width=True):
            if not seed_gen.strip():
                st.error("Please enter a seed phrase.")
            else:
                with st.spinner("Generating..."):
                    generated = generate_text(
                        seed_gen,
                        st.session_state.model,
                        st.session_state.word2idx,
                        st.session_state.idx2word,
                        seq_len=st.session_state.seq_len,
                        num_words=num_words,
                        temperature=temp_gen,
                    )

                seed_words = seed_gen.lower().split()
                gen_words  = generated.split()
                highlighted = " ".join([
                    f"<span style='color:{CYAN};font-weight:700;'>{w}</span>"
                    if i < len(seed_words) else w
                    for i, w in enumerate(gen_words)
                ])
                st.markdown(f"<div class='gen-text-box'>{highlighted}</div>",
                            unsafe_allow_html=True)

                # Token chart
                gen_only = gen_words[len(seed_words):]
                if gen_only:
                    fig_tok = go.Figure(go.Scatter(
                        x=list(range(len(gen_only))),
                        y=[len(w) for w in gen_only],
                        mode="lines+markers+text",
                        text=gen_only,
                        textposition="top center",
                        textfont=dict(size=8, color="#7070a0"),
                        line=dict(color=PURPLE, width=2),
                        marker=dict(color=PURPLE, size=7,
                                    line=dict(color="#080812", width=1.5)),
                        fill="tozeroy", fillcolor="rgba(168,85,247,0.06)",
                    ))
                    fig_tok.update_layout(
                        **PLOT_CFG, height=200,
                        title=dict(text="Generated Token Timeline",
                                   font=dict(color="#dcdcf0", size=12)),
                        xaxis_title="Token position",
                        yaxis_title="Word length",
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

    history    = st.session_state.train_history
    stats      = st.session_state.train_stats
    epochs_run = list(range(1, len(history["loss"]) + 1))

    k1, k2, k3, k4, k5 = st.columns(5)
    perplexity = round(float(np.exp(stats["final_loss"])), 2)
    kpis = [
        (k1, CYAN,   "Epochs",        stats["epochs_trained"]),
        (k2, GREEN,  "Final Accuracy", f"{stats['final_acc']:.1%}"),
        (k3, ORANGE, "Final Loss",     stats["final_loss"]),
        (k4, PURPLE, "Perplexity",    perplexity),
        (k5, "#ec4899", "Vocab Size", f"{stats['vocab_size']:,}"),
    ]
    for col, accent, label, val in kpis:
        with col:
            st.markdown(f"""<div class='card' style='--accent:{accent};'>
                <div class='card-label'>{label}</div>
                <div class='card-value'>{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
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
        st.plotly_chart(fig_curves, use_container_width=True, config={"displayModeBar": False})

    with right:
        perplexities = [round(float(np.exp(l)), 2) for l in history["loss"]]
        fig_perp = go.Figure(go.Scatter(
            x=epochs_run, y=perplexities,
            mode="lines+markers",
            line=dict(color=CYAN, width=2.5),
            marker=dict(size=7, color=CYAN, line=dict(color="#080812", width=1.5)),
            fill="tozeroy", fillcolor="rgba(0,229,255,0.06)",
        ))
        fig_perp.update_layout(
            **PLOT_CFG, height=200,
            title=dict(text="Perplexity over Epochs",
                       font=dict(color="#dcdcf0", size=13)),
            yaxis_title="Perplexity",
        )
        st.plotly_chart(fig_perp, use_container_width=True, config={"displayModeBar": False})

        summary = pd.DataFrame({
            "Epoch":      epochs_run,
            "Loss":       [round(l, 4) for l in history["loss"]],
            "Accuracy":   [f"{a:.1%}" for a in history["accuracy"]],
            "Perplexity": perplexities,
        })
        st.dataframe(summary.tail(10).reset_index(drop=True),
                     use_container_width=True, hide_index=True)

    if st.session_state.word2idx:
        st.markdown("<div class='section-title'>Vocabulary Distribution</div>",
                    unsafe_allow_html=True)
        words   = [w for w in st.session_state.word2idx.keys()
                   if w not in ("<PAD>", "<UNK>")]
        lcount  = Counter(len(w) for w in words)
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
            xaxis_title="Word Length (chars)", yaxis_title="Number of Words",
        )
        st.plotly_chart(fig_voc, use_container_width=True, config={"displayModeBar": False})


# ── FOOTER ──
st.markdown("""
<div style='text-align:center;color:#1a1a30;font-size:0.72rem;margin-top:3rem;
            padding-top:1rem;border-top:1px solid #10101e;'>
    LSTMind &nbsp;·&nbsp; Pure NumPy LSTM &nbsp;·&nbsp; No TensorFlow Required &nbsp;·&nbsp;
    Built with Streamlit &amp; Plotly
</div>
""", unsafe_allow_html=True)
