import os
import glob
import shutil
from typing import Dict, Tuple, List, Optional

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

# --- folders in repo ---
FOLDERS = {
    "selected_models": os.path.join(APP_DIR, "selected_models"),
    "best_models": os.path.join(APP_DIR, "best_models"),
    "better_models": os.path.join(APP_DIR, "better_models"),
}

ARCH_MODULES = {
    "basic": arch_basic,
    "cnn": arch_cnn,
    "dual": arch_dual,
    "roo": arch_roo,
    "tcn": arch_tcn,
    "transformer": arch_transformer,
}

SUPPORTED_AUDIO = ["wav", "m4a", "mp3", "flac", "ogg"]


def ensure_dirs():
    for p in FOLDERS.values():
        os.makedirs(p, exist_ok=True)


def rel_to_folder(path: str) -> Tuple[str, str]:
    """Return (folder_key, relpath) for UI."""
    ap = os.path.abspath(path)
    for k, root in FOLDERS.items():
        root_abs = os.path.abspath(root)
        if ap.startswith(root_abs + os.sep) or ap == root_abs:
            return k, os.path.relpath(ap, root_abs)
    return "unknown", os.path.basename(ap)


def list_models_in_folder(folder_path: str) -> List[str]:
    pattern = os.path.join(os.path.abspath(folder_path), "**", "*.keras")
    return sorted(glob.glob(pattern, recursive=True))


def list_models_all() -> List[str]:
    out = []
    for root in FOLDERS.values():
        out.extend(list_models_in_folder(root))
    return sorted(set(map(os.path.abspath, out)))


def infer_arch_module(model_path: str):
    for _arch_name, mod in ARCH_MODULES.items():
        if mod.can_handle_model(model_path):
            return mod
    return None


def model_tag_from_path(model_path: str) -> str:
    return os.path.basename(model_path).replace(".keras", "")


def sidecar_candidates(model_path: str) -> List[str]:
    """
    Return list of extra files to move/copy with a model.
    We include common patterns:
      - config_<tag>.json
      - <tag>.json  (sometimes people store metadata this way)
    """
    d = os.path.dirname(os.path.abspath(model_path))
    tag = model_tag_from_path(model_path)
    cands = [
        os.path.join(d, f"config_{tag}.json"),
        os.path.join(d, f"{tag}.json"),
    ]
    return [p for p in cands if os.path.exists(p)]


def move_or_copy_model(model_path: str, dst_dir: str, op: str) -> Tuple[bool, str]:
    """
    op: 'move' | 'copy'
    """
    if op not in ("move", "copy"):
        return False, f"Unknown op: {op}"

    src = os.path.abspath(model_path)
    if not os.path.exists(src):
        return False, f"Source not found: {src}"

    os.makedirs(dst_dir, exist_ok=True)
    dst_model = os.path.join(dst_dir, os.path.basename(src))

    if os.path.abspath(os.path.dirname(src)) == os.path.abspath(dst_dir):
        return False, "Source and destination folders are the same."

    if os.path.exists(dst_model):
        return False, f"Destination already has: {dst_model}"

    files = [src] + sidecar_candidates(src)

    try:
        if op == "copy":
            for f in files:
                shutil.copy2(f, os.path.join(dst_dir, os.path.basename(f)))
        else:
            for f in files:
                shutil.move(f, os.path.join(dst_dir, os.path.basename(f)))
        return True, f"{op.upper()} OK: {os.path.basename(src)} (+{len(files)-1} sidecar)"
    except Exception as e:
        return False, f"{op.upper()} failed: {type(e).__name__}: {e}"


def delete_model_with_sidecars(model_path: str) -> Tuple[bool, str]:
    src = os.path.abspath(model_path)
    if not os.path.exists(src):
        return False, f"Not found: {src}"

    files = [src] + sidecar_candidates(src)
    try:
        for f in files:
            if os.path.exists(f):
                os.remove(f)
        return True, f"DELETE OK: {os.path.basename(src)} (+{len(files)-1} sidecar)"
    except Exception as e:
        return False, f"DELETE failed: {type(e).__name__}: {e}"


@st.cache_data(show_spinner=False)
def waveform_for_display(audio_bytes: bytes, suffix: str, sr: int = 44100):
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


def plot_predictions_overlay(
    dfs_by_model: Dict[str, pd.DataFrame],
    y_col: str,
    title: str,
    ylabel: str,
    colors: Dict[str, str],
):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for model_label, df in dfs_by_model.items():
        if df is None or df.empty:
            continue
        ax.plot(
            df["time_sec"].to_numpy(),
            df[y_col].to_numpy(),
            label=model_label,
            color=colors.get(model_label, None),
        )
    ax.set_title(title)
    ax.set_xlabel("Time, sec")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    st.pyplot(fig, clear_figure=True)


st.set_page_config(page_title="Inhale inference + model manager", layout="wide")
st.title("Inhale inference + model manager")

ensure_dirs()

# --- sidebar: model manager ---
with st.sidebar:
    st.header("Model folders")
    st.write({k: v for k, v in FOLDERS.items()})

    st.divider()
    st.header("Model manager")

    src_folder_key = st.selectbox("Source folder", options=list(FOLDERS.keys()), index=0)
    src_dir = FOLDERS[src_folder_key]

    src_models = list_models_in_folder(src_dir)
    if not src_models:
        st.info("No .keras models in selected source folder.")
    src_labels = [os.path.relpath(p, src_dir) for p in src_models]

    selected_for_ops = st.multiselect(
        "Select models to manage (multi-select)",
        options=src_models,
        format_func=lambda p: os.path.relpath(p, src_dir),
    )

    dst_folder_key = st.selectbox("Destination folder", options=list(FOLDERS.keys()), index=1)
    dst_dir = FOLDERS[dst_folder_key]

    colA, colB, colC = st.columns(3)
    do_copy = colA.button("COPY →", use_container_width=True)
    do_move = colB.button("MOVE →", use_container_width=True)
    do_delete = colC.button("DELETE", use_container_width=True)

    if (do_copy or do_move) and selected_for_ops:
        op = "copy" if do_copy else "move"
        msgs = []
        for mp in selected_for_ops:
            ok, msg = move_or_copy_model(mp, dst_dir, op)
            msgs.append(("✅" if ok else "❌") + " " + msg)
        st.write("\n".join(msgs))
        st.rerun()

    if do_delete and selected_for_ops:
        if st.checkbox("Confirm delete (irreversible)", value=False):
            msgs = []
            for mp in selected_for_ops:
                ok, msg = delete_model_with_sidecars(mp)
                msgs.append(("✅" if ok else "❌") + " " + msg)
            st.write("\n".join(msgs))
            st.rerun()
        else:
            st.warning("Enable confirmation checkbox to delete.")

    st.divider()
    st.header("Inference")

    infer_folder_key = st.selectbox("Inference folder", options=list(FOLDERS.keys()), index=0)
    infer_dir = FOLDERS[infer_folder_key]

    hop_sec = st.number_input(
        "Window hop (sec). Used only if 'Use hop = noise_length_sec' is OFF",
        min_value=0.1,
        value=5.0,
        step=0.1,
    )
    use_default_hop = st.checkbox("Use hop = noise_length_sec (per model)", value=True)

    uploads = st.file_uploader(
        "Upload audio files (10+ is OK)",
        type=SUPPORTED_AUDIO,
        accept_multiple_files=True,
    )

# --- models for inference ---
infer_models = list_models_in_folder(infer_dir)
if not infer_models:
    st.error(f"No models found in {infer_folder_key} ({infer_dir})")
    st.stop()

chosen = st.multiselect(
    "Select up to 3 models for inference",
    options=infer_models,
    format_func=lambda p: os.path.relpath(p, infer_dir),
    max_selections=3,
)

if not chosen:
    st.info("Select 1–3 models above.")
    st.stop()

if not uploads:
    st.info("Upload one or more audio files in the sidebar.")
    st.stop()

palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4"])
chosen_names = [os.path.relpath(p, infer_dir) for p in chosen]
colors = {chosen_names[i]: palette[i % len(palette)] for i in range(len(chosen_names))}

# --- model loading (cached) ---
@st.cache_resource(show_spinner=False)
def load_one_model(model_path: str):
    mod = infer_arch_module(model_path)
    if mod is None:
        raise ValueError(f"No architecture handler found for: {os.path.basename(model_path)}")
    model, cfg = mod.load_model_and_cfg(model_path)
    return mod, model, cfg


bundle = {}  # label -> (module, model, cfg)
errors = []
for p in chosen:
    label = os.path.relpath(p, infer_dir)
    try:
        mod, model, cfg = load_one_model(p)
        bundle[label] = (mod, model, cfg)
    except Exception as e:
        errors.append(f"{label}: {type(e).__name__}: {e}")

if errors:
    st.error("Some selected models failed to load:\n\n" + "\n".join(errors))
    st.stop()

# --- inference per audio ---
for up in uploads:
    audio_name = up.name
    suffix = os.path.splitext(audio_name)[1].lower()
    audio_bytes = up.read()

    st.markdown(f"## {audio_name}")
    t, x = waveform_for_display(audio_bytes, suffix, sr=44100)
    plot_waveform(t, x, title=f"Waveform: {audio_name}")

    dfs_by_model = {}
    with st.spinner(f"Predicting for {audio_name} with {len(bundle)} model(s)..."):
        for label, (mod, model, cfg) in bundle.items():
            hop = None if use_default_hop else float(hop_sec)
            df = mod.predict_audio_bytes(model=model, cfg=cfg, audio_bytes=audio_bytes, suffix=suffix, hop_sec=hop)
            dfs_by_model[label] = df

    plot_predictions_overlay(dfs_by_model, "dose_smooth", f"Dose (smoothed, window=5): {audio_name}", "Dose", colors)
    plot_predictions_overlay(dfs_by_model, "flow_smooth", f"Flow (smoothed, window=5): {audio_name}", "Flow", colors)

    with st.expander("Selected models details"):
        rows = []
        for label, (_mod, _m, cfg) in bundle.items():
            rows.append({
                "model": label,
                "model_type": cfg.get("model_type"),
                "feature_type": cfg.get("feature_type"),
                "transform": cfg.get("transformation_method"),
                "loss_type": cfg.get("loss_type"),
                "sample_rate": cfg.get("sample_rate"),
                "noise_length_sec": cfg.get("noise_length_sec"),
                "frame_length_sec": cfg.get("frame_length_sec"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
