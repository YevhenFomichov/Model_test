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

SELECTED_MODELS_DIR = os.path.join(APP_DIR, "selected_models")
BEST_MODELS_DIR = os.path.join(APP_DIR, "best_models")
BETTER_MODELS_DIR = os.path.join(APP_DIR, "better_models")

ARCH_MODULES = {
    "basic": arch_basic,
    "cnn": arch_cnn,
    "dual": arch_dual,
    "roo": arch_roo,
    "tcn": arch_tcn,
    "transformer": arch_transformer,
}

ALL_MODEL_DIRS = {
    "selected_models": SELECTED_MODELS_DIR,
    "best_models": BEST_MODELS_DIR,
    "better_models": BETTER_MODELS_DIR,
}


# -----------------------------
# Helpers: safe paths + scanning
# -----------------------------
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _is_inside(base_dir: str, path: str) -> bool:
    base = os.path.realpath(base_dir)
    p = os.path.realpath(path)
    try:
        common = os.path.commonpath([base, p])
    except Exception:
        return False
    return common == base


def _rel_to_app(path: str) -> str:
    try:
        return os.path.relpath(path, APP_DIR)
    except Exception:
        return path


def _scan_keras_in_dir(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    pattern = os.path.join(os.path.abspath(folder), "**", "*.keras")
    return sorted(glob.glob(pattern, recursive=True))


def _list_all_models_across_folders() -> List[str]:
    out = []
    for _name, d in ALL_MODEL_DIRS.items():
        out.extend(_scan_keras_in_dir(d))
    return sorted(set(os.path.abspath(p) for p in out))


def _find_config_for_model(model_path: str) -> Optional[str]:
    """
    Tries to locate config for a given model:
      - config_<tag>.json next to model
      - otherwise: any config_*.json next to model (fallback)
    """
    d = os.path.dirname(os.path.abspath(model_path))
    tag = os.path.basename(model_path).replace(".keras", "")
    exact = os.path.join(d, f"config_{tag}.json")
    if os.path.exists(exact):
        return exact
    cands = sorted(glob.glob(os.path.join(d, "config_*.json")))
    return cands[0] if cands else None


def _safe_move(src: str, dst_dir: str) -> str:
    """
    Moves file to dst_dir preserving basename.
    Returns new path.
    """
    _ensure_dir(dst_dir)
    if not os.path.isfile(src):
        raise FileNotFoundError(src)
    if not _is_inside(APP_DIR, src):
        raise ValueError(f"Refusing to move file outside APP_DIR: {src}")
    if not _is_inside(APP_DIR, dst_dir):
        raise ValueError(f"Refusing to move to dir outside APP_DIR: {dst_dir}")

    dst = os.path.join(dst_dir, os.path.basename(src))
    # avoid accidental overwrite
    if os.path.exists(dst):
        raise FileExistsError(f"Destination already exists: {dst}")

    shutil.move(src, dst)
    return dst


# -----------------------------
# Model discovery for inference (only selected_models)
# -----------------------------
def list_all_models_selected_only() -> List[str]:
    """
    inference UI should list only selected_models (as you had).
    """
    out = []
    for mod in ARCH_MODULES.values():
        out.extend(mod.list_models(SELECTED_MODELS_DIR))
    return sorted(set(out))


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
    for model_name, df in dfs_by_model.items():
        if df is None or df.empty:
            continue
        ax.plot(
            df["time_sec"].to_numpy(),
            df[y_col].to_numpy(),
            label=model_name,
            color=colors.get(model_name, None),
        )
    ax.set_title(title)
    ax.set_xlabel("Time, sec")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    st.pyplot(fig, clear_figure=True)


# -----------------------------
# Load model routing (by handler)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_one_model(model_path: str):
    for arch_name, mod in ARCH_MODULES.items():
        if mod.can_handle_model(model_path):
            model, cfg = mod.load_model_and_cfg(model_path)
            return arch_name, model, cfg
    raise ValueError(f"No architecture handler found for: {os.path.basename(model_path)}")


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Inhale inference", layout="wide")
st.title("Inhale inference: waveform + dose/flow predictions (≤3 models)")

# ensure folders exist (safe)
for d in ALL_MODEL_DIRS.values():
    _ensure_dir(d)

with st.sidebar:
    st.header("Model manager (optional)")
    enable_manager = st.checkbox("Enable moving models between folders", value=False)

    if enable_manager:
        st.caption("Move .keras files (and matching config_*.json when found) between folders.")
        src_folder_name = st.selectbox("Source folder", options=list(ALL_MODEL_DIRS.keys()), index=0)
        dst_folder_name = st.selectbox("Destination folder", options=list(ALL_MODEL_DIRS.keys()), index=1)

        src_dir = ALL_MODEL_DIRS[src_folder_name]
        dst_dir = ALL_MODEL_DIRS[dst_folder_name]

        src_models = _scan_keras_in_dir(src_dir)
        if not src_models:
            st.info(f"No .keras files in {src_folder_name}.")
        else:
            to_move = st.multiselect(
                "Select models to move",
                options=src_models,
                format_func=lambda p: _rel_to_app(p),
            )

            move_with_config = st.checkbox("Also move matching config_*.json (if found next to model)", value=True)

            if st.button("Move selected", type="primary", disabled=(len(to_move) == 0 or src_folder_name == dst_folder_name)):
                moved = []
                failed = []
                for mp in to_move:
                    try:
                        # move model
                        new_mp = _safe_move(mp, dst_dir)
                        moved.append((_rel_to_app(mp), _rel_to_app(new_mp)))

                        # move config (optional)
                        if move_with_config:
                            cfg_path = _find_config_for_model(new_mp)  # after move, config still in old dir; so also check old
                            # better: check old dir explicitly too
                            cfg_old = _find_config_for_model(os.path.join(src_dir, os.path.basename(new_mp)))
                            if cfg_old and os.path.exists(cfg_old):
                                _safe_move(cfg_old, dst_dir)

                    except Exception as e:
                        failed.append(f"{_rel_to_app(mp)}: {type(e).__name__}: {e}")

                # clear caches because file set changed
                st.cache_resource.clear()
                st.cache_data.clear()

                if moved:
                    st.success("Moved:\n" + "\n".join([f"{a}  ->  {b}" for a, b in moved]))
                if failed:
                    st.error("Failed:\n" + "\n".join(failed))

    st.divider()
    st.header("Inference")
    st.caption("Inference list shows ONLY selected_models/")

    all_models = list_all_models_selected_only()
    if not all_models:
        st.error(f"No models found in '{_rel_to_app(SELECTED_MODELS_DIR)}'.")
        st.stop()

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

# load selected models
bundle = {}  # label -> (arch_group, model, cfg)
errors = []
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

# run inference for each upload
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
