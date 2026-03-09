import hashlib
import os
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
REMOTE_CACHE_DIR = os.path.join(APP_DIR, ".remote_model_cache")
os.makedirs(REMOTE_CACHE_DIR, exist_ok=True)

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
    "dose_models",
    "flow_models",
]

DEFAULT_INFERENCE_FOLDERS = MODEL_FOLDERS[:]
REGISTRY_PATH = "model_registry.json"


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
# GitHub store / registry
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


def bootstrap_registry(store: GitHubRepoStore) -> dict:
    model_paths = store.list_files(suffix=".keras")
    models = {}

    for path in model_paths:
        parts = path.split("/")
        top_folder = parts[0] if len(parts) > 1 else "selected_models"
        if top_folder not in MODEL_FOLDERS:
            continue

        json_path = path.replace(".keras", ".json")
        has_json = store.exists(json_path)

        models[path] = {
            "category": top_folder,
            "physical_path": path,
            "json_path": json_path if has_json else None,
            "name": os.path.basename(path),
        }

    return {"version": 1, "models": models}


@st.cache_data(show_spinner=False)
def load_registry_from_remote() -> dict:
    store = get_github_store()
    if store.exists(REGISTRY_PATH):
        reg = store.read_json(REGISTRY_PATH)
        if "models" in reg:
            return reg
    return bootstrap_registry(store)


def get_registry() -> dict:
    if "model_registry" not in st.session_state:
        st.session_state["model_registry"] = load_registry_from_remote()
    return st.session_state["model_registry"]


def refresh_registry_from_remote() -> dict:
    st.cache_data.clear()
    fresh = load_registry_from_remote()
    st.session_state["model_registry"] = fresh
    return fresh


def get_registry_sha() -> str | None:
    if "model_registry_sha" not in st.session_state:
        store = get_github_store()
        st.session_state["model_registry_sha"] = (
            store.get_file_sha(REGISTRY_PATH, ref=store.cfg.branch)
            if store.exists(REGISTRY_PATH, ref=store.cfg.branch)
            else None
        )
    return st.session_state["model_registry_sha"]


def refresh_registry_sha() -> str | None:
    store = get_github_store()
    sha = (
        store.get_file_sha(REGISTRY_PATH, ref=store.cfg.branch)
        if store.exists(REGISTRY_PATH, ref=store.cfg.branch)
        else None
    )
    st.session_state["model_registry_sha"] = sha
    return sha


def save_registry(registry: dict, message: str, expected_sha: str | None):
    store = get_github_store()
    store.write_json(
        REGISTRY_PATH,
        registry,
        message=message,
        branch=store.cfg.branch,
        expected_sha=expected_sha,
    )
    st.session_state["model_registry"] = registry
    st.cache_data.clear()
    st.cache_resource.clear()
    refresh_registry_sha()


def get_models_by_categories(registry: dict, categories: List[str]) -> List[str]:
    out = []
    for model_key, meta in registry.get("models", {}).items():
        if meta.get("category") in categories:
            out.append(model_key)
    return sorted(set(out))


def build_conflict_rows(
    old_registry: dict,
    new_registry: dict,
    requested_model_keys: List[str],
    requested_dst_category: str,
) -> pd.DataFrame:
    rows = []
    old_models = old_registry.get("models", {})
    new_models = new_registry.get("models", {})
    requested = set(requested_model_keys)

    keys = sorted(set(old_models.keys()) | set(new_models.keys()))
    for key in keys:
        old_cat = old_models.get(key, {}).get("category")
        new_cat = new_models.get(key, {}).get("category")

        if old_cat != new_cat or key in requested:
            rows.append({
                "model": os.path.basename(key),
                "you_requested_change": key in requested,
                "your_requested_category": requested_dst_category if key in requested else None,
                "old_category_seen_by_you": old_cat,
                "current_remote_category": new_cat,
            })

    return pd.DataFrame(rows)


def apply_category_update_safely(model_keys: List[str], dst_category: str, message: str) -> dict:
    expected_sha = get_registry_sha()
    session_registry = get_registry()
    latest = refresh_registry_from_remote()

    if "models" not in latest:
        latest = {"version": 1, "models": {}}

    updated = {
        "version": latest.get("version", 1),
        "models": {k: dict(v) for k, v in latest["models"].items()},
    }

    for model_key in model_keys:
        if model_key not in updated["models"]:
            raise KeyError(f"Model not found in latest registry: {model_key}")
        updated["models"][model_key]["category"] = dst_category

    try:
        save_registry(updated, message=message, expected_sha=expected_sha)
        return updated
    except RuntimeError as e:
        current_remote = refresh_registry_from_remote()
        current_sha = refresh_registry_sha()

        st.session_state["pending_conflict"] = {
            "model_keys": list(model_keys),
            "dst_category": dst_category,
            "message": message,
            "expected_sha": expected_sha,
            "current_sha": current_sha,
            "old_registry": session_registry,
            "remote_registry": current_remote,
            "conflict_rows": build_conflict_rows(
                session_registry,
                current_remote,
                model_keys,
                dst_category,
            ).to_dict(orient="records"),
            "error_text": str(e),
        }
        raise


def retry_pending_conflict_override() -> dict:
    pending = st.session_state.get("pending_conflict")
    if not pending:
        raise RuntimeError("No pending conflict")

    latest = refresh_registry_from_remote()
    latest_sha = refresh_registry_sha()

    updated = {
        "version": latest.get("version", 1),
        "models": {k: dict(v) for k, v in latest["models"].items()},
    }

    for model_key in pending["model_keys"]:
        if model_key not in updated["models"]:
            raise KeyError(f"Model not found in latest registry: {model_key}")
        updated["models"][model_key]["category"] = pending["dst_category"]

    save_registry(updated, message=pending["message"], expected_sha=latest_sha)
    del st.session_state["pending_conflict"]
    return updated


# -----------------------------
# Remote cache
# -----------------------------
def _cache_subdir_for_key(model_key: str) -> str:
    h = hashlib.sha1(model_key.encode("utf-8")).hexdigest()[:12]
    return os.path.join(REMOTE_CACHE_DIR, h)


def ensure_remote_model_cached(store: GitHubRepoStore, model_key: str, registry: dict) -> str:
    if model_key not in registry["models"]:
        raise KeyError(f"Model missing in registry: {model_key}")

    meta = registry["models"][model_key]
    physical_path = meta["physical_path"]
    json_path = meta.get("json_path")

    cache_dir = _cache_subdir_for_key(model_key)
    os.makedirs(cache_dir, exist_ok=True)

    local_model_path = os.path.join(cache_dir, os.path.basename(physical_path))
    model_bytes = store.get_raw_content(physical_path)
    with open(local_model_path, "wb") as f:
        f.write(model_bytes)
        f.flush()
        os.fsync(f.fileno())

    if (not os.path.exists(local_model_path)) or os.path.getsize(local_model_path) == 0:
        raise FileNotFoundError(f"Failed to cache model locally: {local_model_path}")

    if json_path and store.exists(json_path):
        local_json_path = os.path.join(cache_dir, os.path.basename(json_path))
        json_bytes = store.get_raw_content(json_path)
        with open(local_json_path, "wb") as f:
            f.write(json_bytes)
            f.flush()
            os.fsync(f.fileno())

    return local_model_path


# -----------------------------
# Loading models
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_one_model_from_registry_key(model_key: str):
    store = get_github_store()
    registry = get_registry()
    local_model_path = ensure_remote_model_cached(store, model_key, registry)

    for arch_name, mod in ARCH_MODULES.items():
        if hasattr(mod, "can_handle_model") and mod.can_handle_model(local_model_path):
            if not hasattr(mod, "load_model_and_cfg"):
                raise AttributeError(f"Module {mod.__name__} has no load_model_and_cfg")
            model, cfg = mod.load_model_and_cfg(local_model_path)
            return arch_name, model, cfg

    raise ValueError(f"No architecture handler found for: {os.path.basename(model_key)}")


def display_label_for_model(model_key: str) -> str:
    return os.path.basename(model_key)


# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="Inhale inference + safe logical moving", layout="wide")
st.title("Inhale inference: waveform + dose/flow predictions (≤3 models)")

registry = get_registry()
_ = get_registry_sha()

pending = st.session_state.get("pending_conflict")
if pending:
    st.warning("model_registry.json изменился, пока у тебя была открыта старая версия.")
    if pending.get("error_text"):
        st.caption(pending["error_text"])

    conflict_df = pd.DataFrame(pending.get("conflict_rows", []))
    if not conflict_df.empty:
        st.markdown("### Изменения в реестре и твой запрос")
        st.dataframe(conflict_df, use_container_width=True)
    else:
        st.info("Удалось обнаружить конфликт, но список изменений пуст.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Принять новые изменения и повторить перенос", type="primary"):
            try:
                retry_pending_conflict_override()
                st.success("Реестр обновлён до последней версии, перенос повторён.")
                st.rerun()
            except Exception as e:
                st.error(f"Не удалось повторить перенос: {type(e).__name__}: {e}")
    with c2:
        if st.button("Отменить конфликт"):
            del st.session_state["pending_conflict"]
            refresh_registry_from_remote()
            refresh_registry_sha()
            st.rerun()

with st.sidebar:
    st.header("Model manager (safe logical moving)")
    st.caption("Categories are changed in model_registry.json. Model files are not physically moved.")

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
        "Enable moving models between categories",
        value=False,
        disabled=(not can_move),
        key="enable_manager",
    )

    st.divider()
    st.header("Inference")

    if "inference_folders" not in st.session_state:
        st.session_state["inference_folders"] = DEFAULT_INFERENCE_FOLDERS[:]

    inference_folders = st.multiselect(
        "Categories used for model testing",
        options=MODEL_FOLDERS,
        default=st.session_state["inference_folders"],
        key="inference_folders",
    )

    current_registry = get_registry()
    all_models = get_models_by_categories(current_registry, inference_folders)
    if not all_models:
        st.error("No models found in selected categories.")
        st.stop()

    chosen = st.multiselect(
        "Select up to 3 models",
        options=all_models,
        format_func=display_label_for_model,
        max_selections=3,
        key="chosen_models",
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

if enable_manager and can_move:
    st.subheader("Move models between categories safely")

    c1, c2 = st.columns(2)
    with c1:
        src_folder = st.selectbox(
            "Source category",
            options=MODEL_FOLDERS,
            index=MODEL_FOLDERS.index("selected_models"),
            key="src_folder",
        )
    with c2:
        dst_folder = st.selectbox(
            "Destination category",
            options=MODEL_FOLDERS,
            index=MODEL_FOLDERS.index("tested_models"),
            key="dst_folder",
        )

    current_registry = get_registry()
    src_models = get_models_by_categories(current_registry, [src_folder])

    if not src_models:
        st.info(f"No models in category {src_folder}")
    else:
        selected_files = st.multiselect(
            "Select models to move",
            options=src_models,
            format_func=lambda k: os.path.basename(k),
            key="move_selected_files",
        )

        if st.button(
            "Move selected safely",
            type="primary",
            disabled=(len(selected_files) == 0 or src_folder == dst_folder),
            key="move_selected_button",
        ):
            try:
                apply_category_update_safely(
                    model_keys=selected_files,
                    dst_category=dst_folder,
                    message=f"Update model categories: {src_folder} -> {dst_folder}",
                )

                st.session_state["chosen_models"] = []
                st.session_state["move_selected_files"] = []

                current_folders = list(st.session_state.get("inference_folders", []))
                if dst_folder not in current_folders:
                    current_folders.append(dst_folder)
                    st.session_state["inference_folders"] = current_folders

                st.success("Categories updated safely in model_registry.json")
                st.rerun()
            except RuntimeError:
                st.rerun()
            except Exception as e:
                st.error(f"Failed to update registry: {type(e).__name__}: {e}")

    st.divider()

if not chosen:
    st.info("Select 1–3 models in the sidebar.")
    st.stop()

if not uploads:
    st.info("Upload one or more audio files in the sidebar.")
    st.stop()

palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4"])
chosen_names = [display_label_for_model(k) for k in chosen]
colors = {chosen_names[i]: palette[i % len(palette)] for i in range(len(chosen_names))}

bundle = {}
errors = []
for model_key in chosen:
    label = display_label_for_model(model_key)
    try:
        arch_name, model, cfg = load_one_model_from_registry_key(model_key)
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
