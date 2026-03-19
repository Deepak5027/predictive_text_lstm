"""
app.py — LSTM Predictive Text Analyzer
Pure NumPy LSTM — no TensorFlow, works on Python 3.14+
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import time
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

from data_utils import (
    SAMPLE_CORPORA, get_corpus_names, get_corpus_text,
    get_corpus_stats, get_top_words,
)
from model import train_model, predict_next_words, generate_text, clean_text

# ══════════════════════════════════════════════════════════════
st.set_page_config(page_title="LSTM Predictive Text", page_icon="🧠",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #080812; color: #dcdcf0; }
header[data-testid="stHeader"] { display: none; }
section[data-testid="stSidebar"] { background: #0e0e1c !important; border-right: 1px solid #1e1e3a; }
section[data-testid="stSidebar"] * { color: #b0b0d0 !important; }
.card { background: #10101e; border: 1px solid #1e1e3a; border-radius: 14px; padding: 1.3rem 1.5rem; margin-bottom: 0.8rem; position: relative; overflow: hidden; }
.card::after { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: var(--accent, #00e5ff); }
.card-label { font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase; color: #5a5a80; margin-bottom: 0.35rem; }
.card-value { font-size: 1.9rem; font-weight: 800; color: var(--accent, #00e5ff); font-family: 'JetBrains Mono', monospace; line-height: 1.1; }
.card-sub { font-size: 0.76rem; color: #4a4a70; margin-top: 0.25rem; }
.page-title { font-size: 2.4rem; font-weight: 800; letter-spacing: -1px; line-height: 1.1; margin-bottom: 0.2rem; }
.page-sub { font-size: 0.82rem; color: #4a4a70; letter-spacing: 0.06em; }
.section-title { font-size: 1.05rem; font-weight: 700; color: #dcdcf0; margin: 1.5rem 0 0.8rem; padding-left: 0.7rem; border-left: 3px solid #00e5ff; }
.gen-text-box { background: #10101e; border: 1px solid #1e1e3a; border-radius: 12px; padding: 1.2rem 1.4rem; font-family: 'JetBrains Mono', monospace; font-size: 0.86rem; color: #c0c0e0; line-height: 1.9; min-height: 80px; word-break: break-word; }
.stButton > button { background: #00e5ff !important; color: #080812 !important; border: none !important; border-radius: 10px !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; padding: 0.55rem 1.4rem !important; transition: all 0.2s !important; }
.stButton > button:hover { background: #00cfea !important; transform: translateY(-1px) !important; box-shadow: 0 6px 24px rgba(0,229,255,0.3) !important; }
textarea { background: #10101e !important; color: #dcdcf0 !important; border: 1px solid #2a2a4a !important; border-radius: 10px !important; font-family: 'JetBrains Mono', monospace !important; font-size: 0.88rem !important; }
textarea:focus { border-color: #00e5ff !important; box-shadow: 0 0 0 2px rgba(0,229,255,0.15) !important; }
::-webkit-scrollbar { width: 5px; } ::-webkit-scrollbar-track { background: #080812; } ::-webkit-scrollbar-thumb { background: #1e1e3a; border-radius: 3px; }
hr { border-color: #1e1e3a !important; }
div[data-testid="stInfo"], div[data-testid="stWarning"], div[data-testid="stSuccess"], div[data-testid="stError"] { background: #10101e !important; border: 1px solid #1e1e3a !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

CYAN   = "#00e5ff"
PURPLE = "#a855f7"
GREEN  = "#22d3a5"
ORANGE = "#f97316"
BG     = "rgba(0,0,0,0)"
GRID   = "#151528"

def base_layout(height=350, title_text="", title_size=13):
    """Clean base layout — no xaxis/yaxis so callers can set freely."""
    return dict(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="Syne", color="#7070a0"),
        height=height,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(bgcolor=BG, font=dict(color="#b0b0d0")),
        title=dict(text=title_text, font=dict(color="#dcdcf0", size=title_size)),
    )

def axis_style(**kwargs):
    return dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID, **kwargs)

# ══════════════════════════════════════════════════════════════
for k, v in {"model": None, "word2idx": None, "idx2word": None,
             "seq_len": 6, "train_history": None, "train_stats": None, "trained": False}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem;'>
        <div style='font-size:1.3rem;font-weight:800;color:#dcdcf0;'>🧠 LSTMind</div>
        <div style='font-size:0.68rem;color:#4a4a70;letter-spacing:0.14em;text-transform:uppercase;'>Predictive Text Engine</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    nav = st.radio("Navigation", ["🏠 Home","📊 Data Explorer","⚙️ Train Model",
                                   "🔮 Predict & Generate","📈 Model Analytics"],
                   label_visibility="collapsed")
    st.markdown("---")
    if st.session_state.trained:
        s = st.session_state.train_stats or {}
        st.markdown(f"""
        <div style='background:#0a0a18;border:1px solid #1e1e3a;border-radius:10px;padding:0.9rem;font-size:0.75rem;line-height:1.9;'>
            <div style='color:#00e5ff;font-weight:700;margin-bottom:0.3rem;'>✅ Model Trained</div>
            <div style='color:#5a5a80;'>Vocab &nbsp;&nbsp;&nbsp;<b style="color:#dcdcf0">{s.get("vocab_size","—")}</b></div>
            <div style='color:#5a5a80;'>Sequences &nbsp;<b style="color:#dcdcf0">{s.get("sequences","—")}</b></div>
            <div style='color:#5a5a80;'>Accuracy &nbsp;&nbsp;<b style="color:#22d3a5">{s.get("final_acc","—")}</b></div>
            <div style='color:#5a5a80;'>Loss &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b style="color:#f97316">{s.get("final_loss","—")}</b></div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div style='background:#0a0a18;border:1px dashed #1e1e3a;border-radius:10px;
                    padding:0.9rem;font-size:0.75rem;color:#4a4a70;text-align:center;line-height:1.6;'>
            No model trained yet.<br>Go to ⚙️ Train Model.</div>""", unsafe_allow_html=True)

page = nav.split(" ", 1)[1]

# ══════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════
if page == "Home":
    st.markdown("""<div style='margin-bottom:2rem;'>
        <div class='page-title'>LSTM <span style='color:#00e5ff;'>Predictive</span><br>Text Engine</div>
        <div class='page-sub'>MACHINE LEARNING · DEEP LEARNING · NLP · SEQUENCE MODELING</div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, acc, lbl, val, sub in [
        (c1, CYAN,   "Model Type",  "LSTM",     "Long Short-Term Memory"),
        (c2, PURPLE, "Task",        "Next Word", "Sequence prediction"),
        (c3, GREEN,  "Corpora",     "4",         "Built-in datasets"),
        (c4, ORANGE, "Framework",   "NumPy",     "No TensorFlow needed"),
    ]:
        with col:
            st.markdown(f"""<div class='card' style='--accent:{acc};'>
                <div class='card-label'>{lbl}</div>
                <div class='card-value' style='font-size:1.2rem;'>{val}</div>
                <div class='card-sub'>{sub}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>LSTM Architecture</div>", unsafe_allow_html=True)

    layers = ["Input\nSeed Text","Embedding\nLayer","LSTM\nCell","Hidden\nState h","Dense\nLayer","Softmax","Top-K\nPredictions"]
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
            fig_arch.add_annotation(x=i+0.5, y=0, ax=i, ay=0,
                xref="x", yref="y", axref="x", ayref="y",
                arrowhead=2, arrowsize=1.2, arrowcolor="#2a2a4a", arrowwidth=2)
    fig_arch.update_layout(**base_layout(200))
    fig_arch.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, range=[-0.5, 6.5])
    fig_arch.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, range=[-0.8, 0.8])
    st.plotly_chart(fig_arch, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div class='section-title'>How It Works</div>", unsafe_allow_html=True)
    cols = st.columns(5)
    for col, (num, title, desc) in zip(cols, [
        ("01","Choose Corpus","Pick a built-in dataset or paste your own text."),
        ("02","Explore Data","Analyse word frequency, lexical density, text stats."),
        ("03","Train LSTM","Set hyperparameters and train the NumPy LSTM model."),
        ("04","Predict","Enter a seed phrase and get top-K next word predictions."),
        ("05","Analyse","Study training loss, accuracy, perplexity curves."),
    ]):
        with col:
            st.markdown(f"""<div class='card' style='--accent:{CYAN};text-align:center;'>
                <div style='font-family:"JetBrains Mono",monospace;font-size:1.5rem;font-weight:800;color:#1a1a35;'>{num}</div>
                <div style='font-size:0.8rem;font-weight:700;color:#dcdcf0;margin:0.3rem 0;'>{title}</div>
                <div style='font-size:0.72rem;color:#4a4a70;line-height:1.5;'>{desc}</div></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# DATA EXPLORER
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
                                   placeholder="Paste any text...")
        text = custom_text.strip() if custom_text.strip() else text
    st.session_state["selected_text"] = text

    if text.strip():
        stats = get_corpus_stats(text)
        for col, (lbl, val, acc) in zip(st.columns(5), [
            ("Total Words",     stats["total_words"],            CYAN),
            ("Unique Words",    stats["unique_words"],           PURPLE),
            ("Sentences",       stats["sentences"],              GREEN),
            ("Avg Word Len",    stats["avg_word_len"],           ORANGE),
            ("Lexical Density", f"{stats['lexical_density']}%",  "#ec4899"),
        ]):
            with col:
                st.markdown(f"""<div class='card' style='--accent:{acc};'>
                    <div class='card-label'>{lbl}</div>
                    <div class='card-value'>{val}</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        left, right = st.columns([3, 2])

        with left:
            st.markdown("<div class='section-title'>Top 20 Keywords</div>", unsafe_allow_html=True)
            top_words = get_top_words(text, 20)
            words_df  = pd.DataFrame(top_words, columns=["Word", "Count"])
            fig_bar = go.Figure(go.Bar(
                x=words_df["Count"], y=words_df["Word"], orientation="h",
                marker=dict(color=words_df["Count"], colorscale=[[0,"#1e1e3a"],[1,CYAN]],
                            line=dict(color="#080812", width=0.5)),
                text=words_df["Count"], textposition="outside",
                textfont=dict(color="#7070a0", size=10),
            ))
            fig_bar.update_layout(**base_layout(420, "Word Frequency (stopwords removed)"))
            fig_bar.update_xaxes(**axis_style())
            fig_bar.update_yaxes(**axis_style(autorange="reversed"))
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

        with right:
            st.markdown("<div class='section-title'>Word Length Distribution</div>", unsafe_allow_html=True)
            all_words = re.findall(r"[a-z']+", text.lower())
            lengths   = [len(w) for w in all_words if len(w) > 1]
            len_df    = pd.DataFrame(sorted(Counter(lengths).items()), columns=["Length","Count"])
            fig_len = go.Figure(go.Bar(
                x=len_df["Length"], y=len_df["Count"],
                marker=dict(color=PURPLE, opacity=0.85, line=dict(color="#080812", width=0.5)),
            ))
            fig_len.update_layout(**base_layout(240, "Word Length Distribution"))
            fig_len.update_xaxes(**axis_style(title_text="Characters"))
            fig_len.update_yaxes(**axis_style(title_text="Count"))
            st.plotly_chart(fig_len, use_container_width=True, config={"displayModeBar": False})

            st.markdown("<div class='section-title'>Corpus Preview</div>", unsafe_allow_html=True)
            preview = text[:500] + ("..." if len(text) > 500 else "")
            st.markdown(f"<div class='gen-text-box' style='font-size:0.76rem;'>{preview}</div>",
                        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TRAIN MODEL
# ══════════════════════════════════════════════════════════════
elif page == "Train Model":
    st.markdown("<div class='page-title'>Train <span style='color:#00e5ff;'>Model</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>HYPERPARAMETER CONFIGURATION · NUMPY LSTM TRAINING</div><br>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>1. Select Training Corpus</div>", unsafe_allow_html=True)
    src_col, name_col = st.columns(2)
    with src_col:
        corpus_src = st.radio("Source", ["Sample corpus","Custom text"],
                              horizontal=True, label_visibility="collapsed")
    with name_col:
        corpus_name = st.selectbox("Corpus", get_corpus_names(), label_visibility="collapsed")

    train_text = (get_corpus_text(corpus_name) if corpus_src == "Sample corpus"
                  else st.session_state.get("selected_text", get_corpus_text(corpus_name)))
    cstats = get_corpus_stats(train_text)
    st.markdown(f"<div style='font-size:0.78rem;color:#4a4a70;margin-bottom:1rem;'>"
                f"Corpus: <b style='color:#dcdcf0;'>{cstats['total_words']} words</b> · "
                f"<b style='color:#dcdcf0;'>{cstats['unique_words']} unique</b></div>",
                unsafe_allow_html=True)

    st.markdown("<div class='section-title'>2. Configure Hyperparameters</div>", unsafe_allow_html=True)
    h1, h2, h3 = st.columns(3)
    with h1:
        seq_len   = st.slider("Sequence Length", 4, 15, 6,
                              help="How many previous words the model sees at once")
        embed_dim = st.select_slider("Embedding Dim", [16, 32, 64], value=64,
                                     help="Word vector size")
    with h2:
        hidden_dim = st.select_slider("LSTM Hidden Units", [32, 64, 128], value=128,
                                      help="LSTM memory size")
        max_vocab  = st.slider("Max Vocabulary", 300, 2000, 1500, step=100,
                               help="Max unique words the model learns")
    with h3:
        epochs  = st.slider("Epochs", 5, 30, 25, help="Full passes through training data")
        lr      = st.select_slider("Learning Rate", [0.001, 0.003, 0.005, 0.01], value=0.005)
        max_seq = st.slider("Max Sequences", 200, 1000, 800, step=100,
                            help="Cap training sequences for speed")

    st.markdown(f"<div style='background:#0a0a18;border:1px solid #1e1e3a;border-radius:8px;"
                f"padding:0.6rem 1rem;font-size:0.78rem;color:#5a5a80;margin-bottom:1rem;'>"
                f"Estimated sequences: <b style='color:{CYAN};'>"
                f"{min(max_seq, max(0, cstats['total_words'] - seq_len)):,}</b> · "
                f"Pure NumPy — no TensorFlow ✅</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>3. Train</div>", unsafe_allow_html=True)
    if st.button("🚀 Start Training"):
        progress_bar = st.progress(0)
        status_box   = st.empty()

        def on_progress(epoch, total, loss, acc):
            progress_bar.progress(int(epoch / total * 100))
            status_box.markdown(
                f"<div style='font-size:0.82rem;color:{CYAN};'>"
                f"Epoch {epoch}/{total} · Loss: <b>{loss:.4f}</b> · Acc: <b>{acc:.1%}</b></div>",
                unsafe_allow_html=True)

        try:
            start = time.time()
            model, word2idx, idx2word, history, stats_out = train_model(
                text=train_text, seq_len=seq_len, epochs=epochs, lr=lr,
                max_vocab=max_vocab, embed_dim=embed_dim, hidden_dim=hidden_dim,
                max_sequences=max_seq, progress_cb=on_progress,
            )
            elapsed = round(time.time() - start, 1)
            progress_bar.progress(100)
            status_box.empty()

            st.session_state.model         = model
            st.session_state.word2idx      = word2idx
            st.session_state.idx2word      = idx2word
            st.session_state.seq_len       = seq_len
            st.session_state.train_history = history
            st.session_state.train_stats   = stats_out
            st.session_state.trained       = True

            st.success(f"✅ Training complete in {elapsed}s!")
            for col, (acc, lbl, val) in zip(st.columns(4), [
                (CYAN,   "Vocab Size",     f"{stats_out['vocab_size']:,}"),
                (PURPLE, "Sequences Used", f"{stats_out['sequences']:,}"),
                (GREEN,  "Final Accuracy", f"{stats_out['final_acc']:.1%}"),
                (ORANGE, "Final Loss",     f"{stats_out['final_loss']}"),
            ]):
                with col:
                    st.markdown(f"""<div class='card' style='--accent:{acc};'>
                        <div class='card-label'>{lbl}</div>
                        <div class='card-value'>{val}</div></div>""", unsafe_allow_html=True)
        except Exception as e:
            progress_bar.empty()
            status_box.empty()
            st.error(f"Training error: {e}")

# ══════════════════════════════════════════════════════════════
# PREDICT & GENERATE
# ══════════════════════════════════════════════════════════════
elif page == "Predict & Generate":
    st.markdown("<div class='page-title'>Predict <span style='color:#00e5ff;'>&</span> Generate</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>NEXT WORD PREDICTION · TEXT GENERATION · TEMPERATURE SAMPLING</div><br>", unsafe_allow_html=True)

    if not st.session_state.trained:
        st.warning("⚠️ No model trained yet. Please go to ⚙️ Train Model first.")
        st.stop()

    pred_col, gen_col = st.columns(2, gap="large")

    with pred_col:
        st.markdown("<div class='section-title'>🔮 Next Word Prediction</div>", unsafe_allow_html=True)
        seed_input = st.text_area("Seed Text", placeholder="e.g. 'machine learning is'",
                                  height=100, key="seed_pred")
        p1, p2 = st.columns(2)
        with p1:
            top_k = st.slider("Top K predictions", 3, 10, 5)
        with p2:
            temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1,
                                    help="Low=confident, High=creative")

        if st.button("🔍 Predict Next Word", use_container_width=True):
            if not seed_input.strip():
                st.error("Please enter some seed text.")
            else:
                with st.spinner("Predicting..."):
                    preds = predict_next_words(
                        seed_input, st.session_state.model,
                        st.session_state.word2idx, st.session_state.idx2word,
                        seq_len=st.session_state.seq_len,
                        top_k=top_k, temperature=temperature,
                    )
                st.markdown("<br><div style='font-size:0.7rem;color:#4a4a70;text-transform:uppercase;"
                            "letter-spacing:0.12em;margin-bottom:0.6rem;'>Top Predictions</div>",
                            unsafe_allow_html=True)
                max_p  = preds[0][1] if preds else 1
                chips  = "".join([
                    f"<span style='display:inline-block;padding:5px 14px;border-radius:25px;"
                    f"background:rgba(0,229,255,{p/max_p*0.2:.2f});"
                    f"border:1px solid rgba(0,229,255,{p/max_p*0.5:.2f});"
                    f"color:#00e5ff;font-family:JetBrains Mono,monospace;font-size:0.82rem;margin:3px;'>"
                    f"{w} <span style='opacity:0.5;'>{p:.1%}</span></span>"
                    for w, p in preds
                ])
                st.markdown(chips, unsafe_allow_html=True)

                fig_pred = go.Figure(go.Bar(
                    x=[p[1] for p in preds], y=[p[0] for p in preds], orientation="h",
                    marker=dict(color=[p[1] for p in preds],
                                colorscale=[[0,"#1e1e3a"],[1,CYAN]],
                                line=dict(color="#080812", width=0.5)),
                    text=[f"{p[1]:.1%}" for p in preds], textposition="outside",
                    textfont=dict(color="#7070a0", size=10),
                ))
                fig_pred.update_layout(**base_layout(260, "Prediction Probabilities"))
                fig_pred.update_xaxes(**axis_style(tickformat=".0%"))
                fig_pred.update_yaxes(**axis_style(autorange="reversed"))
                st.plotly_chart(fig_pred, use_container_width=True, config={"displayModeBar": False})

    with gen_col:
        st.markdown("<div class='section-title'>✍️ Text Generation</div>", unsafe_allow_html=True)
        seed_gen = st.text_area("Generation Seed",
                                placeholder="e.g. 'artificial intelligence is'",
                                height=100, key="seed_gen")
        g1, g2 = st.columns(2)
        with g1:
            num_words = st.slider("Words to generate", 10, 80, 25)
        with g2:
            temp_gen = st.slider("Temperature ", 0.1, 2.0, 0.8, 0.1, key="temp_gen")

        if st.button("✨ Generate Text", use_container_width=True):
            if not seed_gen.strip():
                st.error("Please enter a seed phrase.")
            else:
                with st.spinner("Generating..."):
                    generated = generate_text(
                        seed_gen, st.session_state.model,
                        st.session_state.word2idx, st.session_state.idx2word,
                        seq_len=st.session_state.seq_len,
                        num_words=num_words, temperature=temp_gen,
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

                gen_only = gen_words[len(seed_words):]
                if gen_only:
                    fig_tok = go.Figure(go.Scatter(
                        x=list(range(len(gen_only))),
                        y=[len(w) for w in gen_only],
                        mode="lines+markers+text",
                        text=gen_only, textposition="top center",
                        textfont=dict(size=8, color="#7070a0"),
                        line=dict(color=PURPLE, width=2),
                        marker=dict(color=PURPLE, size=7, line=dict(color="#080812", width=1.5)),
                        fill="tozeroy", fillcolor="rgba(168,85,247,0.06)",
                    ))
                    fig_tok.update_layout(**base_layout(200, "Generated Token Timeline"))
                    fig_tok.update_xaxes(**axis_style(title_text="Token position"))
                    fig_tok.update_yaxes(**axis_style(title_text="Word length"))
                    st.plotly_chart(fig_tok, use_container_width=True, config={"displayModeBar": False})

# ══════════════════════════════════════════════════════════════
# MODEL ANALYTICS
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
    perplexity = round(float(np.exp(stats["final_loss"])), 2)

    for col, (acc, lbl, val) in zip(st.columns(5), [
        (CYAN,     "Epochs",         stats["epochs_trained"]),
        (GREEN,    "Final Accuracy", f"{stats['final_acc']:.1%}"),
        (ORANGE,   "Final Loss",     stats["final_loss"]),
        (PURPLE,   "Perplexity",     perplexity),
        ("#ec4899","Vocab Size",     f"{stats['vocab_size']:,}"),
    ]):
        with col:
            st.markdown(f"""<div class='card' style='--accent:{acc};'>
                <div class='card-label'>{lbl}</div>
                <div class='card-value'>{val}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([3, 2])

    with left:
        # Use make_subplots for dual-axis — works in both Plotly 5 and 6
        fig_curves = make_subplots(specs=[[{"secondary_y": True}]])
        fig_curves.add_trace(go.Scatter(
            x=epochs_run, y=history["loss"], name="Loss", mode="lines+markers",
            line=dict(color=ORANGE, width=2.5),
            marker=dict(size=7, color=ORANGE, line=dict(color="#080812", width=1.5)),
        ), secondary_y=False)
        fig_curves.add_trace(go.Scatter(
            x=epochs_run, y=history["accuracy"], name="Accuracy", mode="lines+markers",
            line=dict(color=GREEN, width=2.5),
            marker=dict(size=7, color=GREEN, line=dict(color="#080812", width=1.5)),
        ), secondary_y=True)
        fig_curves.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(family="Syne", color="#7070a0"),
            height=340, margin=dict(l=40, r=60, t=40, b=40),
            title=dict(text="Training Loss & Accuracy per Epoch",
                       font=dict(color="#dcdcf0", size=14)),
            legend=dict(bgcolor=BG, font=dict(color="#b0b0d0"), orientation="h", y=1.08),
        )
        fig_curves.update_xaxes(gridcolor=GRID, linecolor=GRID)
        fig_curves.update_yaxes(title_text="Loss", title_font=dict(color=ORANGE),
                                tickfont=dict(color=ORANGE), gridcolor=GRID,
                                secondary_y=False)
        fig_curves.update_yaxes(title_text="Accuracy", title_font=dict(color=GREEN),
                                tickfont=dict(color=GREEN), tickformat=".0%",
                                gridcolor=GRID, secondary_y=True)
        st.plotly_chart(fig_curves, use_container_width=True, config={"displayModeBar": False})

    with right:
        perplexities = [round(float(np.exp(l)), 2) for l in history["loss"]]
        fig_perp = go.Figure(go.Scatter(
            x=epochs_run, y=perplexities, mode="lines+markers",
            line=dict(color=CYAN, width=2.5),
            marker=dict(size=7, color=CYAN, line=dict(color="#080812", width=1.5)),
            fill="tozeroy", fillcolor="rgba(0,229,255,0.06)",
        ))
        fig_perp.update_layout(**base_layout(200, "Perplexity over Epochs"))
        fig_perp.update_xaxes(**axis_style())
        fig_perp.update_yaxes(**axis_style(title_text="Perplexity"))
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
        words  = [w for w in st.session_state.word2idx if w not in ("<PAD>","<UNK>")]
        lcount = Counter(len(w) for w in words)
        ldf    = pd.DataFrame(sorted(lcount.items()), columns=["Length","Words"])
        fig_voc = go.Figure(go.Bar(
            x=ldf["Length"], y=ldf["Words"],
            marker=dict(color=PURPLE, opacity=0.85, line=dict(color="#080812", width=0.5)),
        ))
        fig_voc.update_layout(**base_layout(240, "Vocabulary Word Length Distribution"))
        fig_voc.update_xaxes(**axis_style(title_text="Word Length (chars)"))
        fig_voc.update_yaxes(**axis_style(title_text="Number of Words"))
        st.plotly_chart(fig_voc, use_container_width=True, config={"displayModeBar": False})

st.markdown("""<div style='text-align:center;color:#1a1a30;font-size:0.72rem;margin-top:3rem;
            padding-top:1rem;border-top:1px solid #10101e;'>
    LSTMind · Pure NumPy LSTM · No TensorFlow Required · Built with Streamlit & Plotly
</div>""", unsafe_allow_html=True)
