# app.py  –  run with:  streamlit run app.py
import time
import csv
import io
import numpy as np
import streamlit as st
from PIL import Image
from datetime import datetime, timedelta
import tensorflow as tf
import json

# ---------- config ----------
MODEL_PATH = "cnn_v4_saved_model"
CLASSES = ["New", "Worn", "Damaged"]
HISTORY_PATH = "history.json"
IMG_SIZE = (224, 224)
CONFIDENCE_WARN = 60
CONFIDENCE_LOW  = 40

# ---------- page setup ----------
st.set_page_config(page_title="Classify ✦", page_icon="✦", layout="centered")

# ---------- inject custom theme ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background: linear-gradient(160deg, #0f0c29 0%, #1a1a40 40%, #302b63 100%);
}
#MainMenu, footer, header { visibility: hidden; }

.hero { text-align: center; padding: 2rem 1rem 0.5rem; }
.hero h1 {
    font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(135deg, #f7971e, #ffd200, #ff6fd8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.hero p { color: #a8a4ce; font-size: 1.05rem; }

.stTabs [data-baseweb="tab-list"] {
    justify-content: center; gap: 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 0.5rem;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.04); border-radius: 10px;
    padding: 0.55rem 1.3rem; color: #a8a4ce;
    font-weight: 600; font-size: 0.92rem; border: 1px solid transparent;
}
.stTabs [aria-selected="true"] {
    background: rgba(255,215,0,0.1) !important; color: #ffd200 !important;
    border: 1px solid rgba(255,215,0,0.25) !important;
}

[data-testid="stFileUploader"] {
    border: 2px dashed #5c52a0 !important; border-radius: 16px !important;
    padding: 1.2rem !important; transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover { border-color: #ffd200 !important; }

.result-card, .guide-card, .history-card {
    background: rgba(255,255,255,0.06); backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.12); border-radius: 20px;
    padding: 1.8rem 2rem; margin-top: 1.2rem;
    animation: fadeUp 0.5s ease-out;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-card { text-align: center; }
.result-card .label {
    font-size: 1.7rem; font-weight: 800; color: #fff; margin-bottom: 0.25rem;
}
.result-card .conf { font-size: 1.1rem; color: #ffd200; font-weight: 600; }

.warn-box {
    border-radius: 14px; padding: 1rem 1.4rem; margin-top: 1rem;
    display: flex; align-items: flex-start; gap: 0.7rem;
    animation: fadeUp 0.4s ease-out;
}
.warn-yellow {
    background: rgba(255,193,7,0.1); border: 1px solid rgba(255,193,7,0.3);
}
.warn-red {
    background: rgba(255,82,82,0.1); border: 1px solid rgba(255,82,82,0.35);
}
.warn-box .warn-icon { font-size: 1.3rem; flex-shrink: 0; margin-top: 1px; }
.warn-box .warn-text { color: #e0dcff; font-size: 0.9rem; line-height: 1.55; }
.warn-box .warn-text strong { color: #fff; }

.guide-card h3 { color: #ffd200; font-size: 1.15rem; margin-bottom: 0.5rem; }
.guide-card p, .guide-card li {
    color: #c4bfef; font-size: 0.95rem; line-height: 1.7;
}
.guide-card .step-num {
    display: inline-block;
    background: linear-gradient(135deg, #f7971e, #ffd200);
    color: #0f0c29; font-weight: 800;
    width: 28px; height: 28px; line-height: 28px;
    text-align: center; border-radius: 50%;
    margin-right: 0.6rem; font-size: 0.85rem;
}
.guide-card .step-row { display: flex; align-items: flex-start; margin-bottom: 1rem; }
.guide-card .step-text { color: #c4bfef; font-size: 0.95rem; line-height: 1.6; }
.guide-card .tip {
    background: rgba(255,215,0,0.08); border-left: 3px solid #ffd200;
    border-radius: 0 10px 10px 0; padding: 0.8rem 1rem;
    margin-top: 0.8rem; color: #e0dcff; font-size: 0.88rem;
}

.history-card { margin-top: 0.8rem; }
.history-card .entry-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 0.3rem;
}
.history-card .entry-class { font-size: 1.1rem; font-weight: 800; color: #fff; }
.history-card .entry-conf { font-size: 0.9rem; font-weight: 600; }
.history-card .entry-meta { color: #8a84b8; font-size: 0.82rem; margin-bottom: 0.15rem; }
.history-card .entry-note {
    color: #a8a4ce; font-size: 0.88rem; font-style: italic; margin-top: 0.4rem;
}
.history-empty {
    text-align: center; color: #5c52a0; padding: 2rem; font-size: 0.95rem;
}

.prob-row { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.55rem; }
.prob-name {
    min-width: 110px; text-align: right; color: #c4bfef;
    font-size: 0.85rem; font-weight: 600;
}
.prob-track {
    flex: 1; height: 10px; background: rgba(255,255,255,0.07);
    border-radius: 6px; overflow: hidden;
}
.prob-fill {
    height: 100%; border-radius: 6px;
    background: linear-gradient(90deg, #f7971e, #ffd200);
    transition: width 0.8s cubic-bezier(.22,1,.36,1);
}
.prob-fill.top { background: linear-gradient(90deg, #ff6fd8, #ffd200); }
.prob-pct { min-width: 52px; color: #a8a4ce; font-size: 0.82rem; font-weight: 600; }

.filter-label { color: #a8a4ce; font-size: 0.85rem; font-weight: 600; margin-bottom: 0.3rem; }

.badge {
    text-align: center; margin-top: 3rem;
    color: #5c52a0; font-size: 0.78rem; letter-spacing: 0.08em;
}
</style>
""", unsafe_allow_html=True)

# ---------- load model ----------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ---------- history helpers ----------
def load_history() -> list:
    try:
        with open(HISTORY_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_entry(entry: dict):
    history = load_history()
    history.insert(0, entry)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

def clear_history():
    with open(HISTORY_PATH, "w") as f:
        json.dump([], f)

def history_to_csv(history: list) -> str:
    buf = io.StringIO()
    if not history:
        return ""
    fields = ["timestamp", "filename", "prediction", "confidence", "elapsed_sec", "note"]
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for h in history:
        writer.writerow(h)
    return buf.getvalue()

# ---------- preprocessing ----------
def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

# ---------- confidence warning ----------
def render_confidence_warning(confidence_pct: float):
    if confidence_pct < CONFIDENCE_LOW:
        st.markdown(
            '<div class="warn-box warn-red">'
            '  <span class="warn-icon">🚨</span>'
            '  <span class="warn-text">'
            f'    <strong>Very low confidence ({confidence_pct:.1f}%)</strong><br>'
            '    The model is not sure about this prediction. The image may be '
            '    unclear, poorly lit, or show a part the model wasn\'t trained on. '
            '    <strong>Do not rely on this result</strong> — retake the photo or '
            '    inspect the part manually.'
            '  </span>'
            '</div>',
            unsafe_allow_html=True,
        )
    elif confidence_pct < CONFIDENCE_WARN:
        st.markdown(
            '<div class="warn-box warn-yellow">'
            '  <span class="warn-icon">⚠️</span>'
            '  <span class="warn-text">'
            f'    <strong>Low confidence ({confidence_pct:.1f}%)</strong><br>'
            '    The model is only moderately sure. Consider retaking the photo '
            '    with better lighting and a closer angle, or get a second opinion '
            '    before making maintenance decisions.'
            '  </span>'
            '</div>',
            unsafe_allow_html=True,
        )

# ---------- header ----------
st.markdown(
    '<div class="hero">'
    '  <h1>✦ Classify</h1>'
    '  <p>Inspect · Classify · Track</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ---------- tabs ----------
tab_guide, tab_classify, tab_history = st.tabs(["📖  Guide", "🔍  Classify", "📊  History"])

# ========================  TAB 1 — GUIDE  ========================
with tab_guide:
    st.markdown("""
    <div class="guide-card">
        <h3>What does this tool do?</h3>
        <p>
            This app uses a trained image‑classification model to inspect photos of
            components and tell you their current condition — for example
            <strong>Good</strong>, <strong>Worn</strong>, or <strong>Damaged</strong>.
            Every result is saved so you can track how parts degrade over time.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="guide-card">
        <h3>How to use — step by step</h3>
        <div class="step-row">
            <span class="step-num">1</span>
            <span class="step-text">Open the <strong>Classify</strong> tab above.</span>
        </div>
        <div class="step-row">
            <span class="step-num">2</span>
            <span class="step-text">
                Upload a clear photo of the part you want to inspect.
                Use <strong>.jpg</strong>, <strong>.png</strong>, or <strong>.webp</strong>.
            </span>
        </div>
        <div class="step-row">
            <span class="step-num">3</span>
            <span class="step-text">
                The model runs automatically. You'll see the predicted class and a
                confidence percentage within seconds.
            </span>
        </div>
        <div class="step-row">
            <span class="step-num">4</span>
            <span class="step-text">
                <strong>Check the warning banner</strong> — if confidence is below 60%
                the result may be unreliable. Retake the photo or inspect manually.
            </span>
        </div>
        <div class="step-row">
            <span class="step-num">5</span>
            <span class="step-text">
                Optionally add a <strong>note</strong> (part ID, location, age) before
                saving. The result is logged in the <strong>History</strong> tab.
            </span>
        </div>
        <div class="step-row">
            <span class="step-num">6</span>
            <span class="step-text">
                Check <strong>History</strong> any time to filter past inspections,
                spot patterns, and <strong>export to CSV</strong> for your reports.
            </span>
        </div>
        <div class="tip">
            💡 <strong>Tip:</strong> For the best results, photograph the part close‑up
            with even lighting and avoid blurry or dark images.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="guide-card">
        <h3>Reading the results</h3>
        <p>
            <strong>🟢 85 %+</strong> — High confidence. The model is reliable.<br>
            <strong>🟡 60 – 84 %</strong> — Moderate. Result is probably correct but double‑check if critical.<br>
            <strong>🟠 40 – 59 %</strong> — Low. A yellow warning will appear. Retake the photo or verify manually.<br>
            <strong>🔴 Below 40 %</strong> — Very low. A red warning will appear. Do not rely on the result.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ========================  TAB 2 — CLASSIFY  ========================
with tab_classify:
    uploaded = st.file_uploader(
        "Choose an image…", type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        image = Image.open(uploaded)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, use_container_width=True)

        with st.spinner("Thinking ✦"):
            t0 = time.time()
            x = preprocess(image)
            preds = model.predict(x, verbose=0)[0]
            elapsed = round(time.time() - t0, 2)

        top_idx = int(np.argmax(preds))
        confidence = float(preds[top_idx])
        confidence_pct = confidence * 100

        st.markdown(
            f'<div class="result-card">'
            f'  <div class="label">{CLASSES[top_idx]}</div>'
            f'  <div class="conf">{confidence:.1%} confidence &nbsp;·&nbsp; {elapsed}s</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        render_confidence_warning(confidence_pct)

        with st.expander("Full breakdown"):
            ranked = sorted(enumerate(preds), key=lambda p: p[1], reverse=True)
            bars = ""
            for rank, (i, prob) in enumerate(ranked):
                fc = "prob-fill top" if rank == 0 else "prob-fill"
                bars += (
                    f'<div class="prob-row">'
                    f'  <span class="prob-name">{CLASSES[i]}</span>'
                    f'  <div class="prob-track">'
                    f'    <div class="{fc}" style="width:{prob*100:.1f}%"></div>'
                    f'  </div>'
                    f'  <span class="prob-pct">{prob:.2%}</span>'
                    f'</div>'
                )
            st.markdown(bars, unsafe_allow_html=True)

        st.markdown("")
        note = st.text_input(
            "Add a note (part ID, location, age…)",
            placeholder="e.g.  Bearing #12 — 6 months old",
        )
        if st.button("💾  Save to History", use_container_width=True):
            entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "filename": uploaded.name,
                "prediction": CLASSES[top_idx],
                "confidence": round(confidence_pct, 1),
                "elapsed_sec": elapsed,
                "note": note if note else "—",
                "all_probs": {
                    CLASSES[i]: round(float(preds[i]) * 100, 1) for i in range(len(CLASSES))
                },
            }
            save_entry(entry)
            st.success("Saved ✓")

# ========================  TAB 3 — HISTORY  ========================
with tab_history:
    history = load_history()

    if not history:
        st.markdown(
            '<div class="guide-card history-empty">'
            '  No inspections yet — head to <strong>Classify</strong> and save your first result.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        all_predictions = sorted(set(h["prediction"] for h in history))

        f_col1, f_col2, f_col3 = st.columns([2, 2, 2])
        with f_col1:
            filter_class = st.selectbox("Filter by class", options=["All"] + all_predictions, index=0)
        with f_col2:
            date_options = ["All time", "Last 7 days", "Last 30 days", "Last 90 days"]
            filter_date = st.selectbox("Date range", options=date_options, index=0)
        with f_col3:
            min_conf = st.slider("Min. confidence %", 0, 100, 0, step=5)

        filtered = history
        if filter_class != "All":
            filtered = [h for h in filtered if h["prediction"] == filter_class]
        if filter_date != "All time":
            days_map = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90}
            cutoff = datetime.now() - timedelta(days=days_map[filter_date])
            filtered = [
                h for h in filtered
                if datetime.strptime(h["timestamp"], "%Y-%m-%d %H:%M:%S") >= cutoff
            ]
        if min_conf > 0:
            filtered = [h for h in filtered if h["confidence"] >= min_conf]

        if filtered:
            cond_counts = {}
            for h in filtered:
                cond_counts[h["prediction"]] = cond_counts.get(h["prediction"], 0) + 1
            most_common = max(cond_counts, key=cond_counts.get)

            c1, c2, c3 = st.columns(3)
            c1.metric("Showing", f"{len(filtered)} of {len(history)}")
            c2.metric("Most common", most_common)
            c3.metric("Avg. confidence", f"{np.mean([h['confidence'] for h in filtered]):.1f}%")
        else:
            st.info("No results match your filters. Try widening your criteria.")

        st.markdown("")

        exp_col1, exp_col2 = st.columns([1, 1])
        with exp_col1:
            csv_data = history_to_csv(filtered)
            st.download_button(
                label="📥  Export filtered to CSV", data=csv_data,
                file_name=f"inspections_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv", use_container_width=True,
            )
        with exp_col2:
            csv_all = history_to_csv(history)
            st.download_button(
                label="📥  Export all to CSV", data=csv_all,
                file_name=f"inspections_all_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv", use_container_width=True,
            )

        st.markdown("")

        for h in filtered:
            if h["confidence"] >= 85:
                badge_color = "#ffd200"
            elif h["confidence"] >= 60:
                badge_color = "#f7971e"
            else:
                badge_color = "#ff6fd8"

            st.markdown(
                f'<div class="history-card">'
                f'  <div class="entry-header">'
                f'    <span class="entry-class">{h["prediction"]}</span>'
                f'    <span class="entry-conf" style="color:{badge_color}">'
                f'      {h["confidence"]}%</span>'
                f'  </div>'
                f'  <div class="entry-meta">'
                f'    🕒 {h["timestamp"]} &nbsp;·&nbsp; '
                f'    📁 {h["filename"]} &nbsp;·&nbsp; '
                f'    ⚡ {h["elapsed_sec"]}s'
                f'  </div>'
                f'  <div class="entry-note">📝 {h["note"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("")
        if st.button("🗑️  Clear all history", use_container_width=True):
            clear_history()
            st.rerun()

st.markdown('<div class="badge">BUILT BY JONAS (256226)</div>', unsafe_allow_html=True)