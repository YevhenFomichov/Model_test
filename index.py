import os
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import arch_basic
import arch_cnn
import arch_dual
import arch_roo
import arch_tcn
import arch_transformer

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SELECTED_MODELS_DIR = os.path.join(APP_DIR, "selected_models")

ARCH_MODULES = {
    "basic": arch_basic,
    "cnn": arch_cnn,
    "dual": arch_dual,
    "roo": arch_roo,
    "tcn": arch_tcn,
    "transformer": arch_transformer,
}

def list_all_models() -> List[str]:
    out = []
    for mod in ARCH_MODULES.values():
        out.extend(mod.list_models(SELECTED_MODELS_DIR))
    # unique + sorted
    out = sorted(set(out))
    return out

@st.cache_data(show_spinner=False)
def waveform_for_display(audio_bytes: bytes, suffix: str, sr: int = 44100):
    # use any module loader (they are identical); pick basic
    x = arch_basic.load_audio_mono_from_bytes(audio_bytes, suffix, sr)
    t = np.arange(len(x), dtype=np.float32) / float(sr)
    return t, x

def plot_waveform(t: np.ndarray, x: np.ndarray, title: str):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, x)
    ax.set_title(title)
    ax.set_xlabel("Time, sec")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)

def plot_predictions_overlay(dfs_by_model: Dict[str, pd.DataFrame], y_col: str, title: str, ylabel: str, colors: Dict[str, str]):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for model_name, df in dfs_by_model.items():
        if df is None or df.empty:
            continue
        ax.plot(df["time_sec"].to_numpy(), df[y_col].to_numpy(), label=model_name, color=colors.get(model_name, None))
    ax.set_title(title)
    ax.set_xlabel("Time, sec")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    st.pyplot(fig, clear_figure=True)

st.set_page_config(page_title="Inhale inference", layout="wide")
st.title("Inhale inference: waveform + dose/flow predictions (≤3 models)")

all_models = list_all_models()
if not all_models:
    st.error(f"No models found in '{SELECTED_MODELS_DIR}'.")
    st.stop()

with st.sidebar:
    st.header("Models")
    chosen = st.multiselect(
        "Select up to 3 models",
        options=all_models,
        format_func=lambda p: os.path.relpath(p, SELECTED_MODELS_DIR),
        max_selections=3,
    )

    st.header("Windowing")
    hop_sec = st.number_input(
        "Window hop (sec). Used only if 'Use hop = noise_length_sec' is OFF",
        min_value=0.1,
        value=5.0,
        step=0.1,
    )
    use_default_hop = st.checkbox("Use hop = noise_length_sec (per model)", value=True)

    st.header("Upload audio")
    uploads = st.file_uploader(
        "Upload audio files (10+ is OK)",
        type=["wav", "m4a", "mp3", "flac", "ogg"],
        accept_multiple_files=True,
    )

if not chosen:
    st.info("Select 1–3 models in the sidebar.")
    st.stop()
if not uploads:
    st.info("Upload one or more audio files in the sidebar.")
    st.stop()

palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4"])
chosen_names = [os.path.relpath(p, SELECTED_MODELS_DIR) for p in chosen]
colors = {chosen_names[i]: palette[i % len(palette)] for i in range(len(chosen_names))}

# load models
bundle = {}  # label -> (arch_group, model, cfg)
errors = []

@st.cache_resource(show_spinner=False)
def load_one_model(model_path: str):
    # infer arch group from filename (shared logic replicated in modules)
    # we can ask each module "can you handle this?"
    for arch_name, mod in ARCH_MODULES.items():
        if mod.can_handle_model(model_path):
            model, cfg = mod.load_model_and_cfg(model_path)
            return arch_name, model, cfg
    raise ValueError(f"No architecture handler found for: {os.path.basename(model_path)}")

for p in chosen:
    label = os.path.relpath(p, SELECTED_MODELS_DIR)
    try:
        arch_name, model, cfg = load_one_model(p)
        bundle[label] = (arch_name, model, cfg)
    except Exception as e:
        errors.append(f"{label}: {type(e).__name__}: {e}")

if errors:
    st.error("Some selected models failed to load:\n\n" + "\n".join(errors))
    st.stop()

# run inference
for up in uploads:
    audio_name = up.name
    suffix = os.path.splitext(audio_name)[1].lower()
    audio_bytes = up.read()

    st.markdown(f"## {audio_name}")

    t, x = waveform_for_display(audio_bytes, suffix, sr=44100)
    plot_waveform(t, x, title=f"Waveform: {audio_name}")

    dfs_by_model = {}
    with st.spinner(f"Predicting for {audio_name} with {len(bundle)} model(s)..."):
        for label, (arch_name, model, cfg) in bundle.items():
            hop = None if use_default_hop else float(hop_sec)
            mod = ARCH_MODULES[arch_name]
            df = mod.predict_audio_bytes(model=model, cfg=cfg, audio_bytes=audio_bytes, suffix=suffix, hop_sec=hop)
            dfs_by_model[label] = df

    plot_predictions_overlay(dfs_by_model, "dose_smooth", f"Dose (smoothed, window=5): {audio_name}", "Dose", colors)
    plot_predictions_overlay(dfs_by_model, "flow_smooth", f"Flow (smoothed, window=5): {audio_name}", "Flow", colors)

    with st.expander("Selected models details"):
        rows = []
        for label, (arch_name, _m, cfg) in bundle.items():
            rows.append({
                "model": label,
                "arch_group": arch_name,
                "model_type": cfg.get("model_type"),
                "feature_type": cfg.get("feature_type"),
                "transform": cfg.get("transformation_method"),
                "loss_type": cfg.get("loss_type"),
                "sample_rate": cfg.get("sample_rate"),
                "noise_length_sec": cfg.get("noise_length_sec"),
                "frame_length_sec": cfg.get("frame_length_sec"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
