# index.py
import os
import json
import glob
import tempfile
import zipfile
import shutil
import inspect
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import tensorflow as tf
from tensorflow.keras import layers, models
from pydub import AudioSegment
import matplotlib.pyplot as plt

# Standalone Keras (preferred loader in many Streamlit environments)
try:
    import keras  # Keras 3 / standalone
    HAS_KERAS = True
except Exception:
    keras = None
    HAS_KERAS = False

# Optional librosa
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

SUPPORTED_EXTENSIONS = (".wav", ".m4a", ".mp3", ".flac", ".ogg")

WINDOW_MODELS = {"cnn2d_resnet", "crnn", "tcn_time", "transformer_time", "dual_branch_time"}
FRAME_MODELS  = {"basic", "cnn_lstm", "tcn", "multitask", "roo_tflite"}

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SELECTED_MODELS_DIR = os.path.join(APP_DIR, "selected_models")

HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


# =========================================================
# Audio utils
# =========================================================
def load_audio_mono_from_path(path: str, sr: int) -> np.ndarray:
    seg = AudioSegment.from_file(path)
    seg = seg.set_frame_rate(sr).set_channels(1)
    samples = np.array(seg.get_array_of_samples())
    sw = seg.sample_width
    denom = {1: 128.0, 2: 32768.0, 4: 2147483648.0}.get(sw, float(2 ** (8 * sw - 1)))
    return (samples.astype(np.float32) / denom)


def transform_audio(audio: np.ndarray, method: Optional[str]) -> np.ndarray:
    if method is None:
        return audio.astype(np.float32, copy=False)

    a = audio.astype(np.float32, copy=False)
    eps = 1e-12

    if method == "normalize":
        m = float(np.max(np.abs(a)))
        return a if m < eps else (a / m)

    if method == "standardize":
        mu = float(np.mean(a))
        sd = float(np.std(a))
        return (a - mu) if sd < eps else ((a - mu) / sd)

    if method == "min-max":
        mn = float(np.min(a))
        mx = float(np.max(a))
        d = mx - mn
        return (a - mn) if d < eps else ((a - mn) / d)

    raise ValueError(f"Unsupported transformation: {method}")


def frame_audio(audio: np.ndarray, frame_len_samples: int) -> np.ndarray:
    n_frames = len(audio) // frame_len_samples
    if n_frames <= 0:
        return np.zeros((0, frame_len_samples), dtype=np.float32)
    a = audio[:n_frames * frame_len_samples]
    return a.reshape(n_frames, frame_len_samples).astype(np.float32)


def normalize_feature_matrix(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if X.size == 0:
        return X.astype(np.float32)
    mu = float(X.mean())
    sd = float(X.std())
    return ((X - mu) / sd).astype(np.float32) if sd > eps else (X - mu).astype(np.float32)


# =========================================================
# Feature extraction (same as training)
# =========================================================
def extract_logmel(audio: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    if not HAS_LIBROSA:
        raise RuntimeError("librosa required for logmel features")
    n_mels = int(cfg.get("n_mels", 64))
    n_fft = int(cfg.get("n_fft", 512))
    hop_length = int(cfg.get("hop_length", 256))
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.T.astype(np.float32)


def extract_pcen_mel(audio: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    if not HAS_LIBROSA:
        raise RuntimeError("librosa required for pcen_mel features")
    n_mels = int(cfg.get("n_mels", 64))
    n_fft = int(cfg.get("n_fft", 512))
    hop_length = int(cfg.get("hop_length", 256))
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, power=1.0
    ).astype(np.float32)
    pcen = librosa.pcen(mel * (2**31), sr=sr).astype(np.float32)
    return pcen.T


def extract_spectrogram(audio: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    if not HAS_LIBROSA:
        raise RuntimeError("librosa required for spectrogram features")
    n_fft = int(cfg.get("stft_n_fft", 512))
    hop_length = int(cfg.get("stft_hop_length", 256))
    win_length = cfg.get("stft_win_length", None)
    win_length = int(win_length) if win_length is not None else None
    S = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    return np.abs(S).T.astype(np.float32)


def extract_mfcc(audio: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    if not HAS_LIBROSA:
        raise RuntimeError("librosa required for mfcc features")
    n_mfcc = int(cfg.get("num_mfcc", 13))
    n_fft = int(cfg.get("stft_n_fft", 512))
    hop_length = int(cfg.get("stft_hop_length", 256))
    M = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return M.T.astype(np.float32)


def extract_mfcc_deltas(audio: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    if not HAS_LIBROSA:
        raise RuntimeError("librosa required for mfcc_deltas features")
    n_mfcc = int(cfg.get("num_mfcc", 13))
    n_fft = int(cfg.get("stft_n_fft", 512))
    hop_length = int(cfg.get("stft_hop_length", 256))
    M = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    ).astype(np.float32)
    d1 = librosa.feature.delta(M).astype(np.float32)
    d2 = librosa.feature.delta(M, order=2).astype(np.float32)
    X = np.concatenate([M, d1, d2], axis=0)
    return X.T.astype(np.float32)


def extract_td_spec_stats(audio: np.ndarray, sr: int, cfg: dict) -> Tuple[np.ndarray, int]:
    frame_len = int(sr * float(cfg["frame_length_sec"]))
    Xf = frame_audio(audio, frame_len_samples=frame_len)
    if Xf.shape[0] == 0:
        return np.zeros((0, 7), dtype=np.float32), frame_len

    out = np.zeros((Xf.shape[0], 7), dtype=np.float32)

    def _zcr(frame: np.ndarray) -> float:
        if frame.size <= 1:
            return 0.0
        s = np.sign(frame)
        s[s == 0] = 1
        return float(np.mean(s[1:] != s[:-1]))

    def _rms(frame: np.ndarray) -> float:
        return float(np.sqrt(np.mean(frame * frame) + 1e-12))

    def _crest_factor(frame: np.ndarray) -> float:
        rms = _rms(frame)
        peak = float(np.max(np.abs(frame))) if frame.size else 0.0
        return float(peak / (rms + 1e-12))

    def _teager(frame: np.ndarray) -> float:
        if frame.size < 3:
            return 0.0
        x = frame
        tke = x[1:-1] * x[1:-1] - x[:-2] * x[2:]
        return float(np.mean(tke))

    def _spectral_stats(frame: np.ndarray, sr_: int) -> Tuple[float, float, float]:
        if frame.size == 0:
            return 0.0, 0.0, 0.0
        x = frame.astype(np.float32, copy=False)
        win = np.hanning(len(x)).astype(np.float32)
        xw = x * win
        spec = np.abs(np.fft.rfft(xw)) + 1e-12
        freqs = np.fft.rfftfreq(len(xw), d=1.0 / sr_).astype(np.float32)

        power = spec * spec
        p_sum = float(np.sum(power))
        if p_sum <= 0:
            return 0.0, 0.0, 0.0

        centroid = float(np.sum(freqs * power) / p_sum)
        gm = float(np.exp(np.mean(np.log(spec))))
        am = float(np.mean(spec))
        flatness = float(gm / (am + 1e-12))

        c = np.cumsum(power)
        thr = 0.85 * c[-1]
        idx = int(np.searchsorted(c, thr))
        idx = max(0, min(idx, len(freqs) - 1))
        rolloff = float(freqs[idx])
        return centroid, flatness, rolloff

    for i in range(Xf.shape[0]):
        fr = Xf[i]
        c, f, r = _spectral_stats(fr, sr)
        out[i, 0] = _rms(fr)
        out[i, 1] = _zcr(fr)
        out[i, 2] = _crest_factor(fr)
        out[i, 3] = _teager(fr)
        out[i, 4] = c
        out[i, 5] = f
        out[i, 6] = r

    return out.astype(np.float32), frame_len


def extract_features_from_waveform(audio: np.ndarray, sr: int, cfg: dict) -> Tuple[np.ndarray, int]:
    ftype = str(cfg.get("feature_type", "raw")).lower()

    if ftype == "raw":
        frame_len = int(sr * float(cfg.get("frame_length_sec", 0.05)))
        X = frame_audio(audio, frame_len_samples=frame_len)
        return X.astype(np.float32), frame_len

    if ftype == "td_spec_stats":
        X, hop = extract_td_spec_stats(audio, sr, cfg)
        return X.astype(np.float32), hop

    if ftype in ("logmel", "pcen_mel", "spectrogram", "mfcc", "mfcc_deltas") and not HAS_LIBROSA:
        raise RuntimeError(f"feature_type='{ftype}' requires librosa")

    if ftype == "logmel":
        X = extract_logmel(audio, sr, cfg)
        return X.astype(np.float32), int(cfg.get("hop_length", 256))

    if ftype == "pcen_mel":
        X = extract_pcen_mel(audio, sr, cfg)
        return X.astype(np.float32), int(cfg.get("hop_length", 256))

    if ftype == "spectrogram":
        X = extract_spectrogram(audio, sr, cfg)
        return X.astype(np.float32), int(cfg.get("stft_hop_length", 256))

    if ftype == "mfcc":
        X = extract_mfcc(audio, sr, cfg)
        return X.astype(np.float32), int(cfg.get("stft_hop_length", 256))

    if ftype == "mfcc_deltas":
        X = extract_mfcc_deltas(audio, sr, cfg)
        return X.astype(np.float32), int(cfg.get("stft_hop_length", 256))

    raise ValueError(f"Unsupported feature_type: {ftype}")


# =========================================================
# Parse model filename + cfg
# =========================================================
def infer_train_unit_for_model(model_type: str) -> str:
    mt = str(model_type).lower()
    return "window" if mt in WINDOW_MODELS else "frame"


def parse_model_filename(model_path: str) -> Dict[str, Optional[str]]:
    base = os.path.basename(model_path)
    if base.lower().endswith(".keras"):
        base = base[:-6]
    parts = base.split("_")
    if len(parts) < 4:
        raise ValueError(
            f"Can't parse model filename '{os.path.basename(model_path)}' "
            f"(expected arch_feat_transform_loss.keras)"
        )
    arch = parts[0].lower()
    feat = parts[1].lower()
    transform = parts[2]
    loss = "_".join(parts[3:])
    transform = None if transform == "None" else transform
    return {"model_type": arch, "feature_type": feat, "transformation_method": transform, "loss_type": loss}


def find_nearby_config_json(model_path: str) -> Optional[str]:
    d = os.path.dirname(os.path.abspath(model_path))
    cands = sorted(glob.glob(os.path.join(d, "config_*.json")))
    if not cands:
        return None
    tag = os.path.basename(model_path).replace(".keras", "")
    exact = os.path.join(d, f"config_{tag}.json")
    if os.path.exists(exact):
        return exact
    return cands[0]


def build_effective_cfg(model_path: str) -> dict:
    default_cfg = {
        "sample_rate": 44100,
        "frame_length_sec": 0.05,
        "noise_length_sec": 5.0,

        "feature_type": "raw",
        "transformation_method": None,

        "n_mels": 64,
        "n_fft": 512,
        "hop_length": 256,

        "stft_n_fft": 512,
        "stft_hop_length": 256,
        "stft_win_length": None,

        "num_mfcc": 13,

        # transformer_time params (only for rebuild fallback)
        "transformer_d_model": 128,
        "transformer_num_heads": 4,
        "transformer_ff_dim": 256,
        "transformer_num_layers": 3,
        "dropout_rate": 0.3,

        "model_type": "roo_tflite",
        "train_unit": "frame",
    }

    cfg = dict(default_cfg)

    cfg_path = find_nearby_config_json(model_path)
    if cfg_path is not None:
        with open(cfg_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        cfg.update(loaded)

    info = parse_model_filename(model_path)
    cfg["model_type"] = info["model_type"]
    cfg["feature_type"] = info["feature_type"]
    cfg["transformation_method"] = info["transformation_method"]
    cfg["loss_type"] = info["loss_type"]
    cfg["train_unit"] = infer_train_unit_for_model(cfg["model_type"])
    return cfg


# =========================================================
# Artifact detection
# =========================================================
def is_git_lfs_pointer(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(300)
        return (b"git-lfs" in head) or (b"version https://git-lfs.github.com/spec" in head) or (b"oid sha256:" in head)
    except Exception:
        return False


def is_hdf5_file(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(len(HDF5_MAGIC))
        return head == HDF5_MAGIC
    except Exception:
        return False


def keras_artifact_kind(path: str) -> Tuple[str, Dict[str, Any]]:
    """
    kind in: keras_zip | savedmodel_dir | hdf5 | missing | lfs_pointer | unknown
    """
    p = os.path.abspath(path)
    diag: Dict[str, Any] = {"path": p}

    if not os.path.exists(p):
        return "missing", diag

    diag["is_file"] = os.path.isfile(p)
    diag["is_dir"] = os.path.isdir(p)
    try:
        diag["size_bytes"] = os.path.getsize(p) if os.path.isfile(p) else None
    except Exception:
        diag["size_bytes"] = None

    if os.path.isfile(p) and is_git_lfs_pointer(p):
        return "lfs_pointer", diag

    if os.path.isdir(p):
        return "savedmodel_dir", diag

    if os.path.isfile(p):
        try:
            diag["is_zipfile"] = zipfile.is_zipfile(p)
        except Exception:
            diag["is_zipfile"] = False
        if diag["is_zipfile"]:
            return "keras_zip", diag

        diag["is_hdf5"] = is_hdf5_file(p)
        if diag["is_hdf5"]:
            return "hdf5", diag

    return "unknown", diag


# =========================================================
# Transformer rebuild (fallback when "unsupported callable")
# =========================================================
def _transformer_encoder_block(x, num_heads, d_model, ff_dim, dropout):
    attn = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // max(1, num_heads)
    )(x, x)
    attn = layers.Dropout(dropout)(attn)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn)

    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dropout(dropout)(ff)
    ff = layers.Dense(d_model)(ff)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff)
    return x


def build_model_transformer_time(input_shape: Tuple[int, int, int], cfg: dict) -> tf.keras.Model:
    inp = layers.Input(shape=input_shape)
    drop = float(cfg.get("dropout_rate", 0.3))
    T, D, _ = input_shape

    x = layers.Reshape((T, D))(inp)
    d_model = int(cfg.get("transformer_d_model", 128))
    num_heads = int(cfg.get("transformer_num_heads", 4))
    ff_dim = int(cfg.get("transformer_ff_dim", 256))
    num_layers = int(cfg.get("transformer_num_layers", 3))

    x = layers.Dense(d_model)(x)

    # Use Embedding + explicit positions; implemented via Keras layers to avoid Lambda serialization issues
    pos_idx = tf.range(start=0, limit=T, delta=1)
    pos_emb = layers.Embedding(input_dim=max(256, T + 1), output_dim=d_model)(pos_idx)
    x = x + pos_emb

    for _ in range(num_layers):
        x = _transformer_encoder_block(x, num_heads=num_heads, d_model=d_model, ff_dim=ff_dim, dropout=drop)

    x = layers.Dense(128)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(drop)(x)

    out = layers.Dense(2, activation="linear", name="dose_flow")(x)
    return models.Model(inp, out)


def pad_or_trim_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    if len(x) == target_len:
        return x.astype(np.float32, copy=False)
    if len(x) > target_len:
        return x[:target_len].astype(np.float32, copy=False)
    pad = np.zeros((target_len - len(x),), dtype=np.float32)
    return np.concatenate([x.astype(np.float32, copy=False), pad], axis=0)


def make_model_input_from_audio_window(audio_win: np.ndarray, cfg: dict) -> Tuple[np.ndarray, Dict[str, Any]]:
    sr = int(cfg["sample_rate"])
    win_len = int(sr * float(cfg["noise_length_sec"]))

    audio_win = pad_or_trim_1d(audio_win, win_len)
    mixed = transform_audio(audio_win, cfg.get("transformation_method", None))

    ftype = str(cfg.get("feature_type", "raw")).lower()
    X2d, hop_in_samples = extract_features_from_waveform(mixed, sr, cfg)

    if ftype != "raw":
        X2d = normalize_feature_matrix(X2d)

    if ftype in ("raw", "td_spec_stats"):
        frame_len = int(sr * float(cfg["frame_length_sec"]))
        target_frames = win_len // frame_len
        if X2d.shape[0] != target_frames:
            X2d = X2d[:target_frames]
            if X2d.shape[0] < target_frames:
                pad = np.zeros((target_frames - X2d.shape[0], X2d.shape[1]), dtype=np.float32)
                X2d = np.concatenate([X2d, pad], axis=0)

    X = X2d[..., None].astype(np.float32)  # (T,D,1)
    info = {
        "hop_in_samples": int(hop_in_samples),
        "sr": sr,
        "frames_per_window": int(X.shape[0]),
        "feature_dim": int(X.shape[1]) if X.shape[0] > 0 else 0,
    }
    return X, info


def _extract_weights_path_from_keras_zip(keras_path: str, tmpdir: str) -> str:
    with zipfile.ZipFile(keras_path, "r") as zf:
        zf.extractall(tmpdir)

    # Common Keras v3 layout names:
    # - model.weights.h5
    # - variables.h5 / weights.h5 (rare)
    candidates = []
    for root, _dirs, files in os.walk(tmpdir):
        for fn in files:
            lfn = fn.lower()
            if lfn.endswith(".weights.h5") or lfn == "model.weights.h5" or lfn.endswith("weights.h5"):
                candidates.append(os.path.join(root, fn))

    if not candidates:
        raise FileNotFoundError("No weights file found inside .keras archive")

    # Prefer exact model.weights.h5 if present
    candidates.sort(key=lambda p: (os.path.basename(p).lower() != "model.weights.h5", len(p)))
    return candidates[0]


def rebuild_and_load_transformer_weights(model_path: str, cfg: dict) -> tf.keras.Model:
    # Determine expected input shape from cfg by running feature pipeline on zeros window
    sr = int(cfg["sample_rate"])
    win_len = int(sr * float(cfg["noise_length_sec"]))
    dummy = np.zeros((win_len,), dtype=np.float32)
    X, _info = make_model_input_from_audio_window(dummy, cfg)  # (T,D,1)
    input_shape = tuple(X.shape)

    model = build_model_transformer_time(input_shape=input_shape, cfg=cfg)

    kind, _diag = keras_artifact_kind(model_path)
    if kind == "keras_zip":
        with tempfile.TemporaryDirectory() as td:
            weights_path = _extract_weights_path_from_keras_zip(model_path, td)
            model.load_weights(weights_path)
        return model

    if kind == "hdf5":
        # HDF5 file contains weights groups
        model.load_weights(model_path)
        return model

    # savedmodel_dir is unlikely in your setup for ".keras", but keep a clear error
    raise RuntimeError(f"Unsupported artifact kind for weights-only load: {kind}")


# =========================================================
# Robust loader (normal load_model first; fallback for transformer unsupported callable)
# =========================================================
def _enable_unsafe_deser_if_available():
    if not HAS_KERAS:
        return
    try:
        if hasattr(keras, "config") and hasattr(keras.config, "enable_unsafe_deserialization"):
            keras.config.enable_unsafe_deserialization()
    except Exception:
        pass


def _get_tfoplambda_class():
    if HAS_KERAS:
        cls = getattr(keras.layers, "TFOpLambda", None)
        if cls is not None:
            return cls
        try:
            from keras.src.layers.core.tf_op_layer import TFOpLambda as K3TFOpLambda  # type: ignore
            return K3TFOpLambda
        except Exception:
            pass
        try:
            from keras.layers.core.tf_op_layer import TFOpLambda as K2TFOpLambda  # type: ignore
            return K2TFOpLambda
        except Exception:
            pass
    cls = getattr(tf.keras.layers, "TFOpLambda", None)
    return cls


def _load_model_with_tfoplambda(path: str):
    tfop = _get_tfoplambda_class()
    if HAS_KERAS:
        lambda_layer = keras.layers.Lambda
        scope_ctx = keras.utils.custom_object_scope
    else:
        lambda_layer = tf.keras.layers.Lambda
        scope_ctx = tf.keras.utils.custom_object_scope

    custom_objects = {"TFOpLambda": (tfop if tfop is not None else lambda_layer)}

    # Prefer standalone keras loader (Keras 3)
    if HAS_KERAS:
        _enable_unsafe_deser_if_available()
        load_fn = getattr(getattr(keras, "saving", None), "load_model", None) or keras.models.load_model
        try:
            with scope_ctx(custom_objects):
                return load_fn(path, compile=False, custom_objects=custom_objects, safe_mode=False)
        except TypeError:
            with scope_ctx(custom_objects):
                return load_fn(path, compile=False, custom_objects=custom_objects)

    # Fallback tf.keras
    with tf.keras.utils.custom_object_scope(custom_objects):
        return tf.keras.models.load_model(path, compile=False, custom_objects=custom_objects)


# =========================================================
# Inference
# =========================================================
def model_predict_frames(model, X_t_d_1: np.ndarray, cfg: dict) -> np.ndarray:
    train_unit = str(cfg.get("train_unit", "frame")).lower()

    if train_unit == "frame":
        pred = model.predict(X_t_d_1, verbose=0)
        if isinstance(pred, (list, tuple)):
            pred = pred[-1]
        pred = np.asarray(pred)
        if pred.ndim != 2 or pred.shape[1] != 2:
            raise ValueError(f"Unexpected prediction shape for frame model: {pred.shape}")
        return pred.astype(np.float32)

    if train_unit == "window":
        pred = model.predict(X_t_d_1[None, ...], verbose=0)
        if isinstance(pred, (list, tuple)):
            pred = pred[-1]
        pred = np.asarray(pred)
        if pred.ndim == 3 and pred.shape[0] == 1 and pred.shape[2] == 2:
            return pred[0].astype(np.float32)
        raise ValueError(f"Unexpected prediction shape for window model: {pred.shape}")

    raise ValueError(f"Unsupported train_unit: {train_unit}")


def rolling_mean_5(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    return pd.Series(x).rolling(window=5, min_periods=1, center=True).mean().to_numpy(dtype=np.float32)


def predict_audio_bytes_for_model(
    model,
    cfg: dict,
    audio_bytes: bytes,
    audio_suffix: str,
    window_hop_sec: Optional[float],
) -> pd.DataFrame:
    sr = int(cfg["sample_rate"])
    win_sec = float(cfg["noise_length_sec"])
    hop_sec = win_sec if window_hop_sec is None else float(window_hop_sec)

    win_len = int(sr * win_sec)
    hop_len = max(1, int(sr * hop_sec))

    with tempfile.NamedTemporaryFile(delete=True, suffix=audio_suffix) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        audio = load_audio_mono_from_path(tmp.name, sr)

    n = len(audio)
    rows = []
    w_idx = 0

    for start in range(0, max(1, n), hop_len):
        end = start + win_len
        if start >= n and n > 0:
            break

        chunk = audio[start:end]
        X, info = make_model_input_from_audio_window(chunk, cfg)
        pred = model_predict_frames(model, X, cfg)  # (T,2)

        hop_samples = max(1, int(info["hop_in_samples"]))
        for i in range(pred.shape[0]):
            t_s = (start + i * hop_samples) / sr
            rows.append({
                "time_sec": float(t_s),
                "dose_pred": float(pred[i, 0]),
                "flow_pred": float(pred[i, 1]),
                "window_idx": w_idx,
                "frame_idx": i,
            })

        w_idx += 1
        if end >= n:
            break

    df = pd.DataFrame(rows)
    if not df.empty:
        df["dose_smooth"] = rolling_mean_5(df["dose_pred"].to_numpy())
        df["flow_smooth"] = rolling_mean_5(df["flow_pred"].to_numpy())
    return df


# =========================================================
# Model discovery
# =========================================================
def discover_models(selected_models_dir: str) -> Tuple[List[str], List[Tuple[str, str, Dict[str, Any]]]]:
    pattern = os.path.join(os.path.abspath(selected_models_dir), "**", "*.keras")
    paths = sorted(glob.glob(pattern, recursive=True))
    paths = [os.path.abspath(p) for p in paths]

    valid = []
    skipped = []
    for p in paths:
        kind, diag = keras_artifact_kind(p)
        if kind in ("keras_zip", "savedmodel_dir", "hdf5"):
            valid.append(p)
        else:
            skipped.append((p, kind, diag))
    return valid, skipped


# =========================================================
# Streamlit caching: model loader + cfg
# =========================================================
@st.cache_resource(show_spinner=False)
def load_model_and_cfg(model_path: str):
    model_path = os.path.abspath(model_path)
    kind, diag = keras_artifact_kind(model_path)

    if kind == "missing":
        raise FileNotFoundError(f"Model path missing: {model_path}")
    if kind == "lfs_pointer":
        raise RuntimeError(
            f"'{os.path.basename(model_path)}' is a Git LFS pointer, not the real model file. diag={diag}"
        )
    if kind == "unknown":
        raise RuntimeError(
            f"'{os.path.basename(model_path)}' is not a recognized Keras artifact (zip/dir/hdf5). diag={diag}"
        )

    cfg = build_effective_cfg(model_path)

    ft = str(cfg.get("feature_type", "raw")).lower()
    if ft in ("logmel", "pcen_mel", "spectrogram", "mfcc", "mfcc_deltas") and not HAS_LIBROSA:
        raise RuntimeError(
            f"Model '{os.path.basename(model_path)}' requires librosa (feature_type={ft}), but librosa is not installed."
        )

    # If HDF5 masquerading as .keras, still OK: detect kind="hdf5" and handle in loader
    try:
        model = _load_model_with_tfoplambda(model_path)
        return model, cfg
    except Exception as e:
        msg = str(e)

        # Key fallback: transformer_time sometimes can't be deserialized ("unsupported callable")
        if cfg.get("model_type", "").lower() == "transformer_time" and ("unsupported callable" in msg.lower()):
            model = rebuild_and_load_transformer_weights(model_path, cfg)
            return model, cfg

        raise


@st.cache_data(show_spinner=False)
def waveform_for_display(audio_bytes: bytes, suffix: str, display_sr: int = 44100) -> Tuple[np.ndarray, np.ndarray]:
    with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        x = load_audio_mono_from_path(tmp.name, display_sr)
    t = np.arange(len(x), dtype=np.float32) / float(display_sr)
    return t, x


# =========================================================
# Plot helpers
# =========================================================
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


# =========================================================
# App
# =========================================================
st.set_page_config(page_title="Inhale inference", layout="wide")
st.title("Inhale inference: waveform + dose/flow predictions (≤3 models)")

valid_model_paths, skipped_models = discover_models(SELECTED_MODELS_DIR)

if not valid_model_paths:
    st.error(f"No valid model artifacts found in '{SELECTED_MODELS_DIR}'.")
    if skipped_models:
        st.code("\n".join([f"{p} -> {kind}" for (p, kind, _d) in skipped_models[:50]]))
    st.stop()

model_labels = {p: os.path.relpath(p, SELECTED_MODELS_DIR) for p in valid_model_paths}

with st.sidebar:
    st.header("Models")
    chosen = st.multiselect(
        "Select up to 3 models",
        options=valid_model_paths,
        format_func=lambda p: model_labels.get(p, os.path.basename(p)),
        default=[],
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
        type=[e.replace(".", "") for e in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
    )

if not chosen:
    st.info("Select 1–3 models in the sidebar.")
    st.stop()

if not uploads:
    st.info("Upload one or more audio files in the sidebar.")
    st.stop()

palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4"])
chosen_names = [model_labels[p] for p in chosen]
colors = {chosen_names[i]: palette[i % len(palette)] for i in range(len(chosen_names))}

models_cfgs = {}
load_errors = []
for p in chosen:
    name = model_labels[p]
    try:
        m, cfg = load_model_and_cfg(p)
        models_cfgs[name] = (m, cfg, p)
    except Exception as e:
        load_errors.append(f"{name}: {type(e).__name__}: {e}")

if load_errors:
    st.error("Some selected models failed to load:\n\n" + "\n".join(load_errors))
    st.stop()

for up in uploads:
    audio_name = up.name
    suffix = os.path.splitext(audio_name)[1].lower()
    audio_bytes = up.read()

    st.markdown(f"## {audio_name}")

    t, x = waveform_for_display(audio_bytes, suffix, display_sr=44100)
    plot_waveform(t, x, title=f"Waveform: {audio_name}")

    dfs_by_model: Dict[str, pd.DataFrame] = {}
    with st.spinner(f"Predicting for {audio_name} with {len(models_cfgs)} model(s)..."):
        for model_name, (model, cfg, _model_path) in models_cfgs.items():
            window_hop_sec = None if use_default_hop else float(hop_sec)
            df_pred = predict_audio_bytes_for_model(
                model=model,
                cfg=cfg,
                audio_bytes=audio_bytes,
                audio_suffix=suffix,
                window_hop_sec=window_hop_sec,
            )
            dfs_by_model[model_name] = df_pred

    plot_predictions_overlay(
        dfs_by_model=dfs_by_model,
        y_col="dose_smooth",
        title=f"Dose (smoothed, window=5): {audio_name}",
        ylabel="Dose",
        colors=colors,
    )
    plot_predictions_overlay(
        dfs_by_model=dfs_by_model,
        y_col="flow_smooth",
        title=f"Flow (smoothed, window=5): {audio_name}",
        ylabel="Flow",
        colors=colors,
    )

    with st.expander("Model details (cfg summary)"):
        rows = []
        for model_name, (_model, cfg, model_path) in models_cfgs.items():
            rows.append({
                "model": model_name,
                "model_path": model_path,
                "train_unit": cfg.get("train_unit"),
                "feature_type": cfg.get("feature_type"),
                "transform": cfg.get("transformation_method"),
                "sample_rate": cfg.get("sample_rate"),
                "noise_length_sec": cfg.get("noise_length_sec"),
                "frame_length_sec": cfg.get("frame_length_sec"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
