import glob
import os
import re
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import arch_basic
import arch_cnn
import arch_dual
import arch_roo
import arch_tcn
import arch_transformer
from github_store import GitHubConfig, GitHubRepoStore

APP_DIR = os.path.dirname(os.path.abspath(__file__))

ARCH_MODULES = {
    "basic": arch_basic,
    "cnn": arch_cnn,
    "dual": arch_dual,
    "roo": arch_roo,
    "tcn": arch_tcn,
    "transformer": arch_transformer,
}

MODEL_FOLDERS = [
    "best_models",
    "better_models",
    "delete_models",
    "selected_models",
    "tested_models",
]

SELECTED_FOLDER = "selected_models"


# -----------------------------
# Audio + plotting helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def load_waveform(audio_bytes: bytes, suffix: str, sr: int = 44100) -> Tuple[int, np.ndarray]:
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
# GitHub store
# -----------------------------
def get_github_store() -> GitHubRepoStore:
    gh = st.secrets.get("github", None)
    if gh is None:
        raise RuntimeError("Missing [github] section in Streamlit secrets.")

    cfg = GitHubConfig(
        owner=str(gh["owner"]),
        repo=str(gh["repo"]),
        token=str(gh["token"]),
        branch=str(gh.get("branch", "main")),
    )
    return GitHubRepoStore(cfg)


def _sanitize_branch_name(s: str) -> str:
    s = re.sub(r"[^0-9A-Za-z_\-/]+", "-", s).strip("-")
    return s[:80] if len(s) > 80 else s


# -----------------------------
# Local inference model loading
# -----------------------------
def list_local_selected_models() -> List[str]:
    selected_dir_local = os.path.join(APP_DIR, SELECTED_FOLDER)
    out = []
    for mod in ARCH_MODULES.values():
        out.extend(mod.list_models(selected_dir_local))
    return sorted(set(out))


@st.cache_resource(show_spinner=False)
def load_one_model_local(model_path: str):
    for arch_name, mod in ARCH_MODULES.items():
        if mod.can_handle_model(model_path):
            model, cfg = mod.load_model_and_cfg(model_path)
            return arch_name, model, cfg
    raise ValueError(f"No architecture handler found for: {os.path.basename(model_path)}")


# -----------------------------
# Streamlit app
# -----------------------------
st.set_page_config(page_title="Inhale inference + GitHub PR moving", layout="wide")
st.title("Inhale inference: waveform + dose/flow predictions (≤3 models)")

with st.sidebar:
    st.header("GitHub model manager (moves via PR)")
    st.caption("This creates a GitHub Pull Request that moves selected files between folders.")

    entered_pwd = st.text_input("Admin password", type="password", key="admin_pwd")
    secret_pwd = st.secrets.get("ADMIN_MOVE_PASSWORD", None)

    if not secret_pwd:
        st.warning("ADMIN_MOVE_PASSWORD is not set in secrets. Move feature is disabled.")
        can_move = False
    elif not entered_pwd:
        st.info("Enter admin password to unlock model moving.")
        can_move = False
    elif entered_pwd == str(secret_pwd):
        st.success("Password accepted. Model moving is unlocked.")
        can_move = True
    else:
        st.error("Wrong password.")
        can_move = False

    enable_manager = st.checkbox(
        "Enable moving models between folders",
        value=False,
        disabled=(not can_move),
    )

    move_with_config = st.checkbox(
        "Also move matching config_*.json",
        value=True,
        disabled=(not can_move),
    )

    st.divider()
    st.header("Inference")
    st.caption("Inference reads from local repo checkout (selected_models/)")

    all_models_local = list_local_selected_models()
    if not all_models_local:
        st.error("No models found in local selected_models/")
        st.stop()

    chosen = st.multiselect(
        "Select up to 3 models",
        options=all_models_local,
        format_func=lambda p: os.path.relpath(p, os.path.join(APP_DIR, SELECTED_FOLDER)),
        max_selections=3,
    )

    st.header("Windowing")
    hop_sec = st.number_input("Window hop (sec)", min_value=0.1, value=5.0, step=0.1)
    use_default_hop = st.checkbox("Use hop = noise_length_sec (per model)", value=True)

    st.header("Plot resolution")
    max_plot_points = st.number_input(
        "Max points per plot",
        min_value=2000,
        max_value=200000,
        value=20000,
        step=1000,
    )

    st.header("Audio player")
    show_player = st.checkbox("Show audio player", value=True)
    autoplay = st.checkbox("Autoplay (browser-dependent)", value=False)

    st.header("Upload audio")
    uploads = st.file_uploader(
        "Upload audio files",
        type=["wav", "m4a", "mp3", "flac", "ogg"],
        accept_multiple_files=True,
    )

# -----------------------------
# GitHub move UI
# -----------------------------
if enable_manager and can_move:
    store = get_github_store()

    st.subheader("Move models in GitHub (creates PR)")

    c1, c2 = st.columns(2)
    with c1:
        src_folder = st.selectbox(
            "Source folder",
            options=MODEL_FOLDERS,
            index=MODEL_FOLDERS.index("selected_models"),
            key="src_folder",
        )
    with c2:
        dst_folder = st.selectbox(
            "Destination folder",
            options=MODEL_FOLDERS,
            index=MODEL_FOLDERS.index("tested_models"),
            key="dst_folder",
        )

    src_local = os.path.join(APP_DIR, src_folder)
    os.makedirs(src_local, exist_ok=True)
    keras_local = sorted(glob.glob(os.path.join(src_local, "*.keras")))

    if not keras_local:
        st.info(f"No .keras files found locally in {src_folder}/")
    else:
        selected_files = st.multiselect(
            "Select .keras files to move",
            options=keras_local,
            format_func=lambda p: os.path.basename(p),
        )

        if st.button(
            "Create PR to move selected",
            type="primary",
            disabled=(len(selected_files) == 0 or src_folder == dst_folder),
        ):
            stamp = time.strftime("%Y%m%d-%H%M%S")
            branch = _sanitize_branch_name(f"move-models/{src_folder}-to-{dst_folder}-{stamp}")
            store.create_branch_from(store.cfg.branch, branch)

            moved = []
            failed = []

            for local_path in selected_files:
                base = os.path.basename(local_path)
                src_path = f"{src_folder}/{base}"
                dst_path = f"{dst_folder}/{base}"

                try:
                    store.move_file_via_branch(
                        src_path=src_path,
                        dst_path=dst_path,
                        branch=branch,
                        message_prefix="Move model",
                        overwrite=False,
                    )
                    moved.append(f"{src_path} -> {dst_path}")

                    if move_with_config:
                        tag = base.replace(".keras", "")
                        cfg_src = f"{src_folder}/config_{tag}.json"
                        cfg_dst = f"{dst_folder}/config_{tag}.json"
                        try:
                            _bytes, _sha = store.get_content(cfg_src, ref=branch)
                            store.move_file_via_branch(
                                src_path=cfg_src,
                                dst_path=cfg_dst,
                                branch=branch,
                                message_prefix="Move config",
                                overwrite=False,
                            )
                            moved.append(f"{cfg_src} -> {cfg_dst}")
                        except Exception:
                            pass

                except Exception as e:
                    failed.append(f"{src_path}: {type(e).__name__}: {e}")

            if moved and not failed:
                pr_url = store.create_pull_request(
                    head_branch=branch,
                    title=f"Move models: {src_folder} -> {dst_folder} ({stamp})",
                    body="Automated move from Streamlit UI.\n\nFiles moved:\n- " + "\n- ".join(moved),
                )
                st.success(f"PR created: {pr_url}")
            else:
                if moved:
                    st.info("Moved on branch:\n" + "\n".join(moved))
                if failed:
                    st.error("Failed:\n" + "\n".join(failed))

    st.divider()

# -----------------------------
# Inference
# -----------------------------
if not chosen:
    st.info("Select 1–3 models in the sidebar.")
    st.stop()

if not uploads:
    st.info("Upload one or more audio files in the sidebar.")
    st.stop()

selected_dir_local = os.path.join(APP_DIR, SELECTED_FOLDER)

palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4"])
chosen_names = [os.path.relpath(p, selected_dir_local) for p in chosen]
colors = {chosen_names[i]: palette[i % len(palette)] for i in range(len(chosen_names))}

bundle = {}
errors = []
for p in chosen:
    label = os.path.relpath(p, selected_dir_local)
    try:
        arch_name, model, cfg = load_one_model_local(p)
        bundle[label] = (arch_name, model, cfg)
    except Exception as e:
        errors.append(f"{label}: {type(e).__name__}: {e}")

if errors:
    st.error("Some selected models failed to load:\n\n" + "\n".join(errors))
    st.stop()

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
            df = mod.predict_audio_bytes(
                model=model,
                cfg=cfg,
                audio_bytes=audio_bytes,
                suffix=suffix,
                hop_sec=hop,
            )
            dfs_by_model[label] = df

    preds_by_model = {}
    for label, df in dfs_by_model.items():
        preds_by_model[label] = {
            "dose": interpolate_predictions_to_grid(df, t_grid, "dose_smooth"),
            "flow": interpolate_predictions_to_grid(df, t_grid, "flow_smooth"),
        }

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
