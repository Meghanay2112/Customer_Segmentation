import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, roc_auc_score, roc_curve)
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation · Deep Learning",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0d0f1a 0%, #111827 50%, #0d0f1a 100%);
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13172a 0%, #0e1120 100%) !important;
    border-right: 1px solid #2d3a5e;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #7dd3fc !important;
}

/* Hero */
.hero-banner {
    background: linear-gradient(120deg, #1e3a5f 0%, #162547 40%, #1a1f3c 100%);
    border: 1px solid #2d4a7a;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(56,189,248,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.1rem;
    font-weight: 700;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
}
.hero-sub {
    color: #94a3b8;
    font-size: 1rem;
    letter-spacing: 0.02em;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1a2340 0%, #141b2d 100%);
    border: 1px solid #2d3a5e;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover { transform: translateY(-2px); border-color: #38bdf8; }
.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #64748b;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #38bdf8;
}
.metric-delta {
    font-size: 0.75rem;
    color: #4ade80;
    margin-top: 0.2rem;
}

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #38bdf8;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0e1120;
    border-radius: 10px;
    gap: 4px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #94a3b8 !important;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    border-radius: 7px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1e3a5f, #162547) !important;
    color: #38bdf8 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8 0%, #7c3aed 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    padding: 0.6rem 1.4rem;
    transition: opacity 0.2s, transform 0.15s;
    width: 100%;
}
.stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }

/* Sliders */
.stSlider label { color: #94a3b8 !important; font-size: 0.85rem !important; }

/* Selectbox */
.stSelectbox label { color: #94a3b8 !important; font-size: 0.85rem !important; }

/* Progress */
.stProgress > div > div { background: linear-gradient(90deg, #1d4ed8, #7c3aed); }

/* DataFrame */
.dataframe { background: #111827 !important; color: #e2e8f0 !important; }

/* Info / success / warning boxes */
.stAlert { border-radius: 10px !important; }

/* Plot backgrounds */
.plot-container > div { background: transparent !important; }

/* Tag pill */
.tag {
    display: inline-block;
    background: rgba(56,189,248,0.12);
    color: #38bdf8;
    border: 1px solid rgba(56,189,248,0.3);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-right: 4px;
}
</style>
""", unsafe_allow_html=True)

# ─── Data Generation ─────────────────────────────────────────────────────────
@st.cache_data
def generate_dataset(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    age           = rng.normal(42, 13, n_samples).clip(18, 80)
    annual_income = rng.normal(65000, 28000, n_samples).clip(15000, 200000)
    spending_score= rng.normal(50, 22, n_samples).clip(1, 100)
    recency       = rng.exponential(30, n_samples).clip(1, 365)
    frequency     = rng.poisson(8, n_samples).clip(1, 50)
    monetary      = rng.lognormal(7.5, 1.2, n_samples).clip(100, 50000)
    online_ratio  = rng.beta(3, 2, n_samples)
    satisfaction  = rng.integers(1, 6, n_samples).astype(float)
    loyalty_years = rng.exponential(3, n_samples).clip(0, 20)
    products_bought= rng.poisson(5, n_samples).clip(1, 30)

    # Deterministic segment assignment
    seg = np.zeros(n_samples, dtype=int)
    seg[(annual_income > 90000) & (spending_score > 60) & (frequency > 10)] = 0  # Champions
    seg[(annual_income > 70000) & (spending_score > 45) & (loyalty_years > 3) & (seg == 0)] = 1  # Loyal
    seg[(recency < 15) & (frequency > 6) & (monetary > 2000) & (seg == 0)] = 2  # Potential Loyalists
    seg[(recency > 200) & (seg == 0)] = 3  # At Risk
    seg[(recency > 300) & (monetary < 500) & (seg == 0)] = 4  # Lost / Churned
    # Remaining → default Promising (2) or split
    mask = seg == 0
    seg[mask] = rng.choice([1, 2, 3], size=mask.sum())

    noise_mask = rng.random(n_samples) < 0.04
    seg[noise_mask] = rng.integers(0, 5, noise_mask.sum())

    df = pd.DataFrame({
        "age": age, "annual_income": annual_income,
        "spending_score": spending_score, "recency": recency,
        "frequency": frequency, "monetary": monetary,
        "online_ratio": online_ratio, "satisfaction": satisfaction,
        "loyalty_years": loyalty_years, "products_bought": products_bought,
        "segment": seg,
    })
    return df

SEGMENT_NAMES  = ["Champions", "Loyal", "Potential Loyalists", "At Risk", "Lost / Churned"]
SEGMENT_COLORS = ["#4ade80", "#38bdf8", "#a78bfa", "#fb923c", "#f87171"]
FEATURES       = ["age","annual_income","spending_score","recency",
                  "frequency","monetary","online_ratio","satisfaction",
                  "loyalty_years","products_bought"]

# ─── Model ───────────────────────────────────────────────────────────────────
def build_model(input_dim: int, num_classes: int, dropout: float, units: list) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,))
    x = layers.BatchNormalization()(inputs)

    for u in units:
        x = layers.Dense(u, kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("swish")(x)
        x = layers.Dropout(dropout)(x)

    # Residual shortcut if last unit matches first
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, out)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def plot_confusion_matrix(cm, class_names, figsize=(6, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    cmap = plt.cm.Blues
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm[i, j]
            norm = cm_norm[i, j]
            color = "white" if norm < 0.55 else "#0d1117"
            ax.text(j, i, f"{val}\n({norm:.0%})", ha="center", va="center",
                    fontsize=8.5, color=color, fontfamily="monospace", fontweight="bold")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right", color="#94a3b8", fontsize=8)
    ax.set_yticklabels(class_names, color="#94a3b8", fontsize=8)
    ax.set_xlabel("Predicted Label", color="#94a3b8", fontsize=9, labelpad=10)
    ax.set_ylabel("True Label", color="#94a3b8", fontsize=9, labelpad=10)
    ax.set_title("Confusion Matrix", color="#e2e8f0", fontfamily="monospace",
                 fontsize=11, fontweight="bold", pad=14)
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d3a5e")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="#94a3b8")
    cbar.ax.tick_params(labelcolor="#94a3b8", labelsize=8)
    cbar.set_label("Normalized Rate", color="#94a3b8", fontsize=8)

    plt.tight_layout()
    return fig

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.patch.set_facecolor("#0d1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#111827")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2d3a5e")
        ax.tick_params(colors="#94a3b8")

    epochs = range(1, len(history["accuracy"]) + 1)
    ax1.plot(epochs, history["accuracy"],     color="#38bdf8", lw=2, label="Train")
    ax1.plot(epochs, history["val_accuracy"], color="#a78bfa", lw=2, label="Val", linestyle="--")
    ax1.set_title("Accuracy", color="#e2e8f0", fontfamily="monospace", fontsize=10)
    ax1.set_xlabel("Epoch", color="#94a3b8", fontsize=8)
    ax1.set_ylabel("Accuracy", color="#94a3b8", fontsize=8)
    ax1.legend(framealpha=0, labelcolor="#94a3b8", fontsize=8)
    ax1.yaxis.label.set_color("#94a3b8")
    ax1.xaxis.label.set_color("#94a3b8")

    ax2.plot(epochs, history["loss"],     color="#fb923c", lw=2, label="Train")
    ax2.plot(epochs, history["val_loss"], color="#f87171", lw=2, label="Val", linestyle="--")
    ax2.set_title("Loss", color="#e2e8f0", fontfamily="monospace", fontsize=10)
    ax2.set_xlabel("Epoch", color="#94a3b8", fontsize=8)
    ax2.set_ylabel("Loss", color="#94a3b8", fontsize=8)
    ax2.legend(framealpha=0, labelcolor="#94a3b8", fontsize=8)
    ax2.yaxis.label.set_color("#94a3b8")
    ax2.xaxis.label.set_color("#94a3b8")

    plt.tight_layout()
    return fig

def plot_feature_importance(model, feature_names, X_sample):
    baseline = model.predict(X_sample, verbose=0)
    importance = []
    X_perm = X_sample.copy()
    for i in range(X_perm.shape[1]):
        orig = X_perm[:, i].copy()
        np.random.shuffle(X_perm[:, i])
        perm_pred = model.predict(X_perm, verbose=0)
        importance.append(np.mean(np.abs(baseline - perm_pred)))
        X_perm[:, i] = orig

    importance = np.array(importance)
    importance = importance / importance.max()
    idx = np.argsort(importance)

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#111827")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d3a5e")

    colors = plt.cm.cool(np.linspace(0.3, 0.9, len(feature_names)))
    bars = ax.barh([feature_names[i] for i in idx], importance[idx],
                   color=colors, height=0.65, edgecolor="#0d1117")

    for bar, val in zip(bars, importance[idx]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", color="#e2e8f0", fontsize=8, fontfamily="monospace")

    ax.set_title("Feature Importance (Permutation)", color="#e2e8f0",
                 fontfamily="monospace", fontsize=10, fontweight="bold")
    ax.set_xlabel("Normalized Importance", color="#94a3b8", fontsize=8)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.xaxis.label.set_color("#94a3b8")
    plt.tight_layout()
    return fig

def plot_pca_clusters(X_scaled, y, segment_names, segment_colors):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#111827")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d3a5e")

    for i, (name, color) in enumerate(zip(segment_names, segment_colors)):
        mask = y == i
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=color, alpha=0.45, s=18, label=name, edgecolors="none")

    ax.set_title(f"PCA Cluster Visualization\n(Var explained: {pca.explained_variance_ratio_.sum():.1%})",
                 color="#e2e8f0", fontfamily="monospace", fontsize=10, fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", color="#94a3b8", fontsize=8)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", color="#94a3b8", fontsize=8)
    ax.tick_params(colors="#94a3b8", labelsize=7)
    legend = ax.legend(framealpha=0.1, labelcolor="#e2e8f0", fontsize=7.5,
                       frameon=True, facecolor="#1a2340", edgecolor="#2d3a5e")
    plt.tight_layout()
    return fig

def plot_segment_distribution(y, segment_names, segment_colors):
    counts = pd.Series(y).value_counts().sort_index()
    labels = [segment_names[i] for i in counts.index]
    sizes  = counts.values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor("#0d1117")

    # Donut
    ax1.set_facecolor("#0d1117")
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=segment_colors, startangle=140,
        wedgeprops=dict(width=0.55, edgecolor="#0d1117", linewidth=2),
        textprops=dict(color="#e2e8f0", fontsize=8),
    )
    for at in autotexts:
        at.set_color("#0d1117"); at.set_fontsize(7.5); at.set_fontweight("bold")
    ax1.set_title("Segment Distribution", color="#e2e8f0",
                  fontfamily="monospace", fontsize=10, fontweight="bold")

    # Bar
    ax2.set_facecolor("#111827")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#2d3a5e")
    bars = ax2.barh(labels, sizes, color=segment_colors, height=0.6, edgecolor="#0d1117")
    for bar, size in zip(bars, sizes):
        ax2.text(bar.get_width() + 15, bar.get_y() + bar.get_height() / 2,
                 f"{size:,}", va="center", color="#e2e8f0", fontsize=8, fontfamily="monospace")
    ax2.set_title("Count per Segment", color="#e2e8f0",
                  fontfamily="monospace", fontsize=10, fontweight="bold")
    ax2.tick_params(colors="#94a3b8", labelsize=8)
    ax2.set_xlabel("Count", color="#94a3b8", fontsize=8)

    plt.tight_layout()
    return fig

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Configuration")
    st.markdown("---")

    n_samples = st.slider("Dataset Size", 1000, 10000, 5000, 500)
    test_size  = st.slider("Test Split (%)", 10, 40, 20, 5) / 100

    st.markdown("**Architecture**")
    arch_preset = st.selectbox("Network Preset", [
        "Standard (256-128-64)",
        "Deep (512-256-128-64)",
        "Ultra (512-512-256-128-64)",
        "Compact (128-64-32)",
    ])
    arch_map = {
        "Standard (256-128-64)":          [256, 128, 64],
        "Deep (512-256-128-64)":          [512, 256, 128, 64],
        "Ultra (512-512-256-128-64)":     [512, 512, 256, 128, 64],
        "Compact (128-64-32)":            [128, 64, 32],
    }
    units = arch_map[arch_preset]

    dropout = st.slider("Dropout Rate", 0.1, 0.5, 0.25, 0.05)
    epochs  = st.slider("Max Epochs", 20, 200, 80, 10)

    st.markdown("---")
    train_btn = st.button("🚀 Train Model", use_container_width=True)
    st.markdown("---")

    st.markdown("""
    <div style='font-size:0.73rem; color:#475569; line-height:1.7'>
    <b style='color:#38bdf8'>Model</b>: Multi-layer DNN<br>
    <b style='color:#38bdf8'>Optimizer</b>: AdamW + LR Decay<br>
    <b style='color:#38bdf8'>Regularization</b>: L2 + Dropout + BN<br>
    <b style='color:#38bdf8'>Callbacks</b>: EarlyStopping<br>
    <b style='color:#38bdf8'>Segments</b>: 5 customer classes
    </div>
    """, unsafe_allow_html=True)

# ─── Hero Banner ─────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-banner'>
  <div class='hero-title'>🧠 Customer Segmentation · Deep Learning</div>
  <div class='hero-sub'>
    Multi-class neural network &nbsp;·&nbsp; RFM + behavioural features &nbsp;·&nbsp;
    Real-time inference &nbsp;·&nbsp; Explainability dashboard
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Session State ────────────────────────────────────────────────────────────
if "trained" not in st.session_state:
    st.session_state.trained = False

# ─── Generate data ────────────────────────────────────────────────────────────
df = generate_dataset(n_samples)
X  = df[FEATURES].values
y  = df["segment"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size, random_state=42, stratify=y
)

# ─── TRAIN ────────────────────────────────────────────────────────────────────
if train_btn:
    st.session_state.trained = False
    model = build_model(len(FEATURES), 5, dropout, units)

    progress_bar = st.progress(0)
    status_text  = st.empty()

    class StreamlitCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            pct = int((epoch + 1) / self.params["epochs"] * 100)
            progress_bar.progress(pct)
            status_text.markdown(
                f"<span style='color:#38bdf8;font-family:monospace;font-size:0.82rem'>"
                f"Epoch {epoch+1}/{self.params['epochs']} &nbsp;·&nbsp; "
                f"acc: **{logs['accuracy']:.4f}** &nbsp;·&nbsp; "
                f"val_acc: **{logs['val_accuracy']:.4f}** &nbsp;·&nbsp; "
                f"loss: {logs['loss']:.4f}</span>",
                unsafe_allow_html=True,
            )

    callbacks = [
        EarlyStopping(patience=12, restore_best_weights=True, monitor="val_accuracy"),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, monitor="val_loss"),
        StreamlitCallback(),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=64,
        callbacks=callbacks,
        verbose=0,
    )

    progress_bar.empty()
    status_text.empty()

    y_pred  = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_prob  = model.predict(X_test, verbose=0)
    acc     = accuracy_score(y_test, y_pred)
    cm      = confusion_matrix(y_test, y_pred)
    report  = classification_report(y_test, y_pred, target_names=SEGMENT_NAMES, output_dict=True)
    hist_dict = {k: v.tolist() for k, v in history.history.items()}

    st.session_state.update({
        "trained": True, "acc": acc, "cm": cm,
        "report": report, "history": hist_dict,
        "model": model, "y_pred": y_pred, "y_prob": y_prob,
        "X_test": X_test, "y_test": y_test,
        "X_scaled": X_scaled, "y": y,
    })
    st.success(f"✅ Training complete — Test Accuracy: **{acc:.4f}** ({acc*100:.2f}%)")

# ─── RESULTS ─────────────────────────────────────────────────────────────────
if st.session_state.trained:
    acc     = st.session_state.acc
    cm      = st.session_state.cm
    report  = st.session_state.report
    history = st.session_state.history
    model   = st.session_state.model
    y_pred  = st.session_state.y_pred
    y_prob  = st.session_state.y_prob
    X_test  = st.session_state.X_test
    y_test  = st.session_state.y_test
    X_scaled_full = st.session_state.X_scaled
    y_full  = st.session_state.y

    # ── Metric Cards ─────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    macro_f1   = report["macro avg"]["f1-score"]
    macro_prec = report["macro avg"]["precision"]
    macro_rec  = report["macro avg"]["recall"]

    for col, label, val, delta in [
        (c1, "TEST ACCURACY",  f"{acc:.4f}",        f"{acc*100:.1f}%"),
        (c2, "MACRO F1",       f"{macro_f1:.4f}",   "balanced"),
        (c3, "MACRO PRECISION",f"{macro_prec:.4f}", "avg all classes"),
        (c4, "MACRO RECALL",   f"{macro_rec:.4f}",  "avg all classes"),
    ]:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-label'>{label}</div>
              <div class='metric-value'>{val}</div>
              <div class='metric-delta'>{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main Two-Column Layout (Confusion Matrix LEFT, Charts RIGHT) ──────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Confusion Matrix & Results",
        "📈 Training Curves",
        "🔍 Feature Importance",
        "🎯 Predict New Customer",
    ])

    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        col_left, col_right = st.columns([1, 1.15], gap="large")

        with col_left:
            st.markdown("<div class='section-header'>Confusion Matrix</div>",
                        unsafe_allow_html=True)
            fig_cm = plot_confusion_matrix(cm, SEGMENT_NAMES)
            st.pyplot(fig_cm, use_container_width=True)
            plt.close(fig_cm)

            # Per-class accuracy
            st.markdown("<div class='section-header'>Per-Class Metrics</div>",
                        unsafe_allow_html=True)
            rows = []
            for i, name in enumerate(SEGMENT_NAMES):
                r = report[name]
                rows.append({
                    "Segment": name,
                    "Precision": f"{r['precision']:.3f}",
                    "Recall":    f"{r['recall']:.3f}",
                    "F1":        f"{r['f1-score']:.3f}",
                    "Support":   int(r["support"]),
                })
            st.dataframe(pd.DataFrame(rows).set_index("Segment"),
                         use_container_width=True, height=220)

        with col_right:
            st.markdown("<div class='section-header'>PCA Cluster Visualization</div>",
                        unsafe_allow_html=True)
            fig_pca = plot_pca_clusters(X_scaled_full[:2000], y_full[:2000],
                                        SEGMENT_NAMES, SEGMENT_COLORS)
            st.pyplot(fig_pca, use_container_width=True)
            plt.close(fig_pca)

            st.markdown("<div class='section-header'>Segment Distribution</div>",
                        unsafe_allow_html=True)
            fig_dist = plot_segment_distribution(y_full, SEGMENT_NAMES, SEGMENT_COLORS)
            st.pyplot(fig_dist, use_container_width=True)
            plt.close(fig_dist)

    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("<div class='section-header'>Training History</div>",
                    unsafe_allow_html=True)
        fig_hist = plot_training_history(history)
        st.pyplot(fig_hist, use_container_width=True)
        plt.close(fig_hist)

        # Epoch table (last 15)
        hist_df = pd.DataFrame(history)
        hist_df.index += 1
        hist_df.index.name = "Epoch"
        hist_df.columns = [c.replace("val_", "Val ").replace("_", " ").title()
                           for c in hist_df.columns]
        st.markdown("<div class='section-header'>Epoch Log (last 15)</div>",
                    unsafe_allow_html=True)
        st.dataframe(hist_df.tail(15).style.format("{:.4f}"),
                     use_container_width=True, height=360)

    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("<div class='section-header'>Permutation Feature Importance</div>",
                    unsafe_allow_html=True)
        with st.spinner("Computing feature importance…"):
            fig_fi = plot_feature_importance(model, FEATURES, X_test[:500])
        st.pyplot(fig_fi, use_container_width=True)
        plt.close(fig_fi)

        st.info("Feature importance is computed by permuting each feature and measuring "
                "the average change in predicted probability across all classes.")

    # ════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("<div class='section-header'>Predict Segment for a New Customer</div>",
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            p_age    = st.slider("Age", 18, 80, 35)
            p_income = st.number_input("Annual Income ($)", 15000, 200000, 60000, 1000)
            p_spend  = st.slider("Spending Score", 1, 100, 50)
            p_recency= st.slider("Recency (days)", 1, 365, 30)
        with c2:
            p_freq   = st.slider("Purchase Frequency", 1, 50, 8)
            p_mon    = st.number_input("Monetary Value ($)", 100, 50000, 2000, 100)
            p_online = st.slider("Online Purchase Ratio", 0.0, 1.0, 0.5, 0.01)
        with c3:
            p_sat    = st.slider("Satisfaction (1-5)", 1, 5, 3)
            p_loyal  = st.slider("Loyalty Years", 0.0, 20.0, 3.0, 0.5)
            p_prod   = st.slider("Products Bought", 1, 30, 5)

        predict_btn = st.button("🔮 Predict Customer Segment", use_container_width=True)
        if predict_btn:
            inp = np.array([[p_age, p_income, p_spend, p_recency,
                             p_freq, p_mon, p_online, p_sat, p_loyal, p_prod]])
            inp_scaled = scaler.transform(inp)
            probs = model.predict(inp_scaled, verbose=0)[0]
            pred_class = np.argmax(probs)
            seg_name   = SEGMENT_NAMES[pred_class]
            seg_color  = SEGMENT_COLORS[pred_class]

            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#1a2340,#141b2d);
                        border:2px solid {seg_color}; border-radius:14px;
                        padding:1.5rem 2rem; margin:1rem 0;'>
              <div style='font-size:0.8rem;color:#94a3b8;letter-spacing:0.1em;
                          text-transform:uppercase;margin-bottom:0.4rem'>Predicted Segment</div>
              <div style='font-size:2rem;font-weight:700;color:{seg_color};
                          font-family:monospace'>{seg_name}</div>
              <div style='font-size:0.85rem;color:#94a3b8;margin-top:0.3rem'>
                Confidence: <b style='color:{seg_color}'>{probs[pred_class]*100:.1f}%</b>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Prob bars
            fig_pb, ax = plt.subplots(figsize=(8, 3))
            fig_pb.patch.set_facecolor("#0d1117")
            ax.set_facecolor("#111827")
            for spine in ax.spines.values():
                spine.set_edgecolor("#2d3a5e")
            bars = ax.barh(SEGMENT_NAMES, probs * 100,
                           color=SEGMENT_COLORS, height=0.6, edgecolor="#0d1117")
            for bar, p in zip(bars, probs):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                        f"{p*100:.1f}%", va="center", color="#e2e8f0",
                        fontsize=9, fontfamily="monospace")
            ax.set_xlabel("Confidence (%)", color="#94a3b8", fontsize=9)
            ax.tick_params(colors="#94a3b8", labelsize=9)
            ax.set_title("Class Probabilities", color="#e2e8f0",
                         fontfamily="monospace", fontsize=10, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_pb, use_container_width=True)
            plt.close(fig_pb)

else:
    # ── Pre-train Landing ─────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Dataset Preview</div>",
                unsafe_allow_html=True)
    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.dataframe(df.head(12).style.format({
            "annual_income": "${:,.0f}", "monetary": "${:,.0f}",
            "spending_score": "{:.1f}", "online_ratio": "{:.2f}",
            "loyalty_years": "{:.1f}",
        }), use_container_width=True, height=420)

    with col2:
        st.markdown("<div class='section-header'>Segment Legend</div>",
                    unsafe_allow_html=True)
        for name, color in zip(SEGMENT_NAMES, SEGMENT_COLORS):
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:10px;
                        padding:8px 12px;margin:5px 0;
                        background:#141b2d;border-radius:8px;
                        border-left:4px solid {color}'>
              <span style='color:{color};font-weight:700;font-size:0.9rem'>{name}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:#141b2d;border-radius:10px;padding:1rem;
                    border:1px solid #2d3a5e;font-size:0.82rem;color:#94a3b8'>
          <b style='color:#38bdf8'>Dataset:</b> {n_samples:,} synthetic customers<br>
          <b style='color:#38bdf8'>Features:</b> {len(FEATURES)} (RFM + behavioural)<br>
          <b style='color:#38bdf8'>Target:</b> 5 customer segments<br>
          <b style='color:#38bdf8'>Split:</b> {int((1-test_size)*100)}% train / {int(test_size*100)}% test
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center;padding:2rem;color:#475569;font-size:0.9rem'>
      ← Configure your model in the sidebar and click <b style='color:#38bdf8'>🚀 Train Model</b>
    </div>""", unsafe_allow_html=True)
