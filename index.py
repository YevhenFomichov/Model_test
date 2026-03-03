import os
import glob
import shutil
from typing import Dict, List, Optional, Tuple

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

# Inference MUST remain based on selected_models only
SELECTED_MODELS_DIR = os.path.join(APP_DIR, "selected_models")

ARCH_MODULES = {
    "basic": arch_basic,
    "cnn": arch_cnn,
    "dual": arch_dual,
    "roo": arch_roo,
    "tcn": arch_tcn,
    "transformer": arch_transformer,
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


def _find_config_for_model(model_path: str) -> Optional[str]:
    d = os.path.dirname(os.path.abspath(model_path))
    tag = os.path.basename(model_path).replace(".keras", "")
    exact = os.path.join(d, f"config_{tag}.json")
    if os.path.exists(exact):
        return exact
    cands = sorted(glob.glob(os.path.join(d, "config_*.json")))
    return cands[0] if cands else None


def _safe_move(src: str, dst_dir: str) -> str:
    _ensure_dir(dst_dir)
    if not os.path.isfile(src):
        raise FileNotFoundError(src)
    if not _is_inside(APP_DIR, src):
        raise ValueError(f"Refusing to move file outside APP_DIR: {src}")
    if not _is_inside(APP_DIR, dst_dir):
        raise ValueError(f"Refusing to move to dir outside APP_DIR: {dst_dir}")

    dst = os.path.join(dst_dir, os.path.basename(src))
    if os.path.exists(dst):
        raise FileExistsError(f"Destination already exists: {dst}")

    shutil.move(src, dst)
    return dst


def discover_model_dirs() -> Dict[str, str]:
    """
    Auto-discover folders in APP_DIR that end with '_models'.
    This includes selected_models, best_models, better_models, plus any new *_models folders you add.
    """
    out: Dict[str, str] = {}
    for name in sorted(os.listdir(APP_DIR)):
        p = os.path.join(APP_DIR, name)
        if os.path.isdir(p) and name.endswith("_models"):
            out[name] = p
    # Ensure selected_models exists in dict (and on disk)
    out.setdefault("selected_models", SELECTED_MODELS_DIR)
    _ensure_dir(out["selected_models"])
    return out


# -----------------------------
# Model discovery for inference (only selected_models)
# -----------------------------
def list_all_models_selected_only() -> List[str]:
    out = []
    for mod in ARCH_MODULES.values():
        out.extend(mod.list_models(SELECTED_MODELS_DIR))
    return sorted(set(out))


# -----------------------------
# Audio + plotting helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def load_waveform(audio_bytes: bytes, suffix: str, sr: int = 44100) -> Tuple[int, np.ndarray]:
    # all handlers should share same loader; we use the stable one from arch_basic
    x = arch_basic.load_audio_mono_from_bytes(audio_bytes, suffix, sr)
    return sr, x


def make_common_time_grid(sr: int, n_samples: int, max_points: int) -> np.ndarray:
    duration = n_samples / float(sr)
    if n_samples <= 1 or duration <= 0:
        return np.zeros((0,), dtype=np.float32)
    n = int(min(max_points, n_samples))
    return np.linspace(0.0, duration, num=n, endpoint=False, dtype=np.float32)


def downsample_waveform_to_grid(x: np.ndarray, sr: int, t_grid: np.ndarray) -> np.ndarray:
    if t_grid.size == 0:
        return np.zeros((0,), dtype=np.float32)
    t0 = np.arange(len(x), dtype=np.float32) / float(sr)
    return np.interp(t_grid, t0, x.astype(np.float32), left=x[0], right=x[-1]).astype(np.float32)


def interpolate_predictions_to_grid(df: pd.DataFrame, t_grid: np.ndarray, col: str) -> np.ndarray:
    """
    Interpolate prediction series onto t_grid so it matches waveform duration exactly.
    Uses edge-hold extrapolation to cover full [0, duration).
    """
    if df is None or df.empty or t_grid.size == 0:
        return np.full((t_grid.size,), np.nan, dtype=np.float32)

    d = df[["time_sec", col]].dropna().sort_values("time_sec")
    if d.empty:
        return np.full((t_grid.size,), np.nan, dtype=np.float32)

    t = d["time_sec"].to_numpy(dtype=np.float32)
    y = d[col].to_numpy(dtype=np.float32)

    duration = float(t_grid[-1]) if t_grid.size else 0.0
    mask = (t >= 0.0) & (t <= duration + 1e-6)
    if np.any(mask):
        t = t[mask]
        y = y[mask]

    if t.size == 0:
        return np.full((t_grid.size,), np.nan, dtype=np.float32)

    return np.interp(t_grid, t, y, left=y[0], right=y[-1]).astype(np.float32)


def plot_waveform_and_predictions(
    t_grid: np.ndarray,
    x_grid: np.ndarray,
    preds_by_model: Dict[str, Dict[str, np.ndarray]],
    colors: Dict[str, str],
    title_prefix: str,
):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t_grid, x_grid)
    ax.set_title(f"{title_prefix} — Waveform")
    ax.set_xlabel("Time, sec")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for name, p in preds_by_model.items():
        ax.plot(t_grid, p["dose"], label=name, color=colors.get(name, None))
    ax.set_title(f"{title_prefix} — Dose (smoothed, window=5) — aligned to waveform")
    ax.set_xlabel("Time, sec")
    ax.set_ylabel("Dose")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    st.pyplot(fig, clear_figure=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for name, p in preds_by_model.items():
        ax.plot(t_grid, p["flow"], label=name, color=colors.get(name, None))
    ax.set_title(f"{title_prefix} — Flow (smoothed, window=5) — aligned to waveform")
    ax.set_xlabel("Time, sec")
    ax.set_ylabel("Flow")
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


def _guess_streamlit_audio_mime(suffix: str) -> str:
    s = suffix.lower()
    if s == ".wav":
        return "audio/wav"
    if s == ".mp3":
        return "audio/mpeg"
    if s == ".m4a":
        return "audio/mp4"
    if s == ".flac":
        return "audio/flac"
    if s == ".ogg":
        return "audio/ogg"
    return "audio/*"


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Inhale inference", layout="wide")
st.title("Inhale inference: waveform + dose/flow predictions (≤3 models)")

ALL_MODEL_DIRS = discover_model_dirs()

# ensure discovered dirs exist
for d in ALL_MODEL_DIRS.values():
    _ensure_dir(d)

with st.sidebar:
    st.header("Model manager (optional)")
    enable_manager = st.checkbox("Enable moving models between folders", value=False)

    if enable_manager:
        st.caption("Folders are auto-detected: any directory in repo root ending with '_models'.")
        st.caption("Move .keras files (and matching config_*.json when found) between folders.")

        folder_names = list(ALL_MODEL_DIRS.keys())
        # Put selected_models first for convenience
        folder_names = ["selected_models"] + [n for n in folder_names if n != "selected_models"]

        src_folder_name = st.selectbox("Source folder", options=folder_names, index=0)
        dst_folder_name = st.selectbox(
            "Destination folder",
            options=folder_names,
            index=min(1, len(folder_names) - 1),
        )

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

            if st.button(
                "Move selected",
                type="primary",
                disabled=(len(to_move) == 0 or src_folder_name == dst_folder_name),
            ):
                moved = []
                failed = []
                for mp in to_move:
                    try:
                        old_dir = os.path.dirname(os.path.abspath(mp))
                        old_mp = mp

                        # move model
                        new_mp = _safe_move(old_mp, dst_dir)
                        moved.append((_rel_to_app(old_mp), _rel_to_app(new_mp)))

                        # move config (optional) — look in old location
                        if move_with_config:
                            cfg_old = _find_config_for_model(old_mp)
                            if cfg_old and os.path.exists(cfg_old) and os.path.dirname(cfg_old) == old_dir:
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

    st.header("Plot resolution")
    max_plot_points = st.number_input(
        "Max points per plot (controls downsampling, affects waveform & aligned predictions)",
        min_value=2000,
        max_value=200000,
        value=20000,
        step=1000,
    )

    st.header("Audio player")
    show_player = st.checkbox("Show audio player for each uploaded file", value=True)
    autoplay = st.checkbox("Autoplay (browser-dependent)", value=False)

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
bundle = {}
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

    if show_player:
        mime = _guess_streamlit_audio_mime(suffix)
        try:
            st.audio(audio_bytes, format=mime, autoplay=autoplay)
        except TypeError:
            st.audio(audio_bytes, format=mime)

    sr, x = load_waveform(audio_bytes, suffix, sr=44100)
    t_grid = make_common_time_grid(sr=sr, n_samples=len(x), max_points=int(max_plot_points))
    x_grid = downsample_waveform_to_grid(x, sr=sr, t_grid=t_grid)

    dfs_by_model = {}
    with st.spinner(f"Predicting for {audio_name} with {len(bundle)} model(s)..."):
        for label, (arch_name, model, cfg) in bundle.items():
            hop = None if use_default_hop else float(hop_sec)
            mod = ARCH_MODULES[arch_name]
            df = mod.predict_audio_bytes(model=model, cfg=cfg, audio_bytes=audio_bytes, suffix=suffix, hop_sec=hop)
            dfs_by_model[label] = df

    preds_by_model = {}
    for label, df in dfs_by_model.items():
        dose = interpolate_predictions_to_grid(df, t_grid, "dose_smooth")
        flow = interpolate_predictions_to_grid(df, t_grid, "flow_smooth")
        preds_by_model[label] = {"dose": dose, "flow": flow}

    plot_waveform_and_predictions(
        t_grid=t_grid,
        x_grid=x_grid,
        preds_by_model=preds_by_model,
        colors=colors,
        title_prefix=audio_name,
    )

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
