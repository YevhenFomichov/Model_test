import os
import glob
import json
import zipfile
import shutil
import tempfile
import inspect
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
from pydub import AudioSegment
import tensorflow as tf

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

ARCH_GROUP = "transformer"
MODEL_TYPES = {"transformer_time"}
KNOWN_FEATURE_TYPES = {"raw", "td_spec_stats", "logmel", "pcen_mel", "spectrogram", "mfcc", "mfcc_deltas"}
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


def list_models(selected_models_dir: str) -> List[str]:
    pattern = os.path.join(os.path.abspath(selected_models_dir), "**", "*.keras")
    paths = sorted(glob.glob(pattern, recursive=True))
    out = []
    for p in paths:
        try:
            info = parse_model_filename(p)
            if info["model_type"] in MODEL_TYPES:
                out.append(os.path.abspath(p))
        except Exception:
            continue
    return out


def can_handle_model(model_path: str) -> bool:
    try:
        info = parse_model_filename(model_path)
        return info["model_type"] in MODEL_TYPES
    except Exception:
        return False


def parse_model_filename(model_path: str) -> Dict[str, Optional[str]]:
    base = os.path.basename(model_path)
    if base.lower().endswith(".keras"):
        base = base[:-6]
    parts = base.split("_")

    feat_idx = None
    for i, tok in enumerate(parts):
        if tok.lower() in KNOWN_FEATURE_TYPES:
            feat_idx = i
            break
    if feat_idx is None or feat_idx == 0 or feat_idx + 2 >= len(parts):
        raise ValueError(f"Can't parse model filename: {os.path.basename(model_path)}")

    model_type = "_".join(parts[:feat_idx]).lower()
    feature_type = parts[feat_idx].lower()
    transform = parts[feat_idx + 1]
    loss = "_".join(parts[feat_idx + 2:])
    transform = None if transform == "None" else transform

    return {
        "model_type": model_type,
        "feature_type": feature_type,
        "transformation_method": transform,
        "loss_type": loss,
    }


def find_nearby_config_json(model_path: str) -> Optional[str]:
    d = os.path.dirname(os.path.abspath(model_path))
    tag = os.path.basename(model_path).replace(".keras", "")
    exact = os.path.join(d, f"config_{tag}.json")
    if os.path.exists(exact):
        return exact
    cands = sorted(glob.glob(os.path.join(d, "config_*.json")))
    return cands[0] if cands else None


def build_effective_cfg(model_path: str) -> dict:
    cfg = {
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
        "model_type": "transformer_time",
        "loss_type": "mse",
    }
    cfg_path = find_nearby_config_json(model_path)
    if cfg_path is not None:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    cfg.update(parse_model_filename(model_path))
    return cfg


def is_hdf5_file(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(len(HDF5_MAGIC))
        return head == HDF5_MAGIC
    except Exception:
        return False


def _get_safe_custom_objects():
    # помогает пройти слой TFOpLambda/SlicingOpLambda как "известный" тип
    return {
        "TFOpLambda": tf.keras.layers.Lambda,
        "SlicingOpLambda": tf.keras.layers.Lambda,
    }


def _looks_like_unsupported_callable(err: Exception) -> bool:
    s = f"{type(err).__name__}: {err}".lower()
    return ("unsupported callable" in s) or ("unsafe" in s and "deserial" in s) or ("lambda" in s and "deserialize" in s)


def _load_with_tfkeras(path: str, custom_objects: dict):
    # tf.keras: safe_mode иногда есть, иногда нет — поэтому пробуем аккуратно
    with tf.keras.utils.custom_object_scope(custom_objects):
        try:
            return tf.keras.models.load_model(path, compile=False)
        except TypeError as e:
            # возможно есть safe_mode=False как kwarg (в некоторых окружениях)
            if "safe_mode" in str(e):
                return tf.keras.models.load_model(path, compile=False, safe_mode=False)
            raise


def _load_with_keras3(path: str, custom_objects: dict):
    # keras 3: умеет safe_mode=False / enable_unsafe_deserialization
    import keras  # noqa

    # enable_unsafe_deserialization() если доступно
    try:
        if hasattr(keras, "config") and hasattr(keras.config, "enable_unsafe_deserialization"):
            keras.config.enable_unsafe_deserialization()
    except Exception:
        pass

    # пробуем keras.saving.load_model (preferred)
    if hasattr(keras, "saving") and hasattr(keras.saving, "load_model"):
        fn = keras.saving.load_model
        sig = None
        try:
            sig = inspect.signature(fn)
        except Exception:
            sig = None

        kwargs = {"compile": False, "custom_objects": custom_objects}
        if sig is not None and "safe_mode" in sig.parameters:
            kwargs["safe_mode"] = False

        return fn(path, **kwargs)

    # fallback: keras.models.load_model
    if hasattr(keras, "models") and hasattr(keras.models, "load_model"):
        fn = keras.models.load_model
        sig = None
        try:
            sig = inspect.signature(fn)
        except Exception:
            sig = None

        kwargs = {"compile": False, "custom_objects": custom_objects}
        if sig is not None and "safe_mode" in sig.parameters:
            kwargs["safe_mode"] = False

        return fn(path, **kwargs)

    raise RuntimeError("keras load_model is not available in this environment")


def load_model_and_cfg(model_path: str):
    cfg = build_effective_cfg(model_path)
    ft = str(cfg.get("feature_type", "raw")).lower()
    if ft in ("logmel", "pcen_mel", "spectrogram", "mfcc", "mfcc_deltas") and not HAS_LIBROSA:
        raise RuntimeError(f"Model requires librosa (feature_type={ft}), but librosa is not installed")

    custom_objects = _get_safe_custom_objects()

    # поддержка HDF5, замаскированного под .keras
    load_path = model_path
    tmp_path = None
    if (not zipfile.is_zipfile(model_path)) and is_hdf5_file(model_path):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
            tmp_path = tmp.name
        shutil.copyfile(model_path, tmp_path)
        load_path = tmp_path

    try:
        # 1) сначала пробуем tf.keras
        try:
            model = _load_with_tfkeras(load_path, custom_objects)
            return model, cfg
        except Exception as e1:
            # 2) если похожее на unsafe callable / unknown layer — пробуем keras (Keras 3) unsafe
            if _looks_like_unsupported_callable(e1) or ("unknown layer" in str(e1).lower()) or ("tfolambda" in str(e1).lower()):
                try:
                    model = _load_with_keras3(load_path, custom_objects)
                    return model, cfg
                except Exception as e2:
                    # вернуть более информативно
                    raise RuntimeError(
                        f"Failed to load transformer model with tf.keras and keras unsafe.\n"
                        f"tf.keras error: {type(e1).__name__}: {e1}\n"
                        f"keras error: {type(e2).__name__}: {e2}"
                    ) from e2
            raise
    finally:
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def load_audio_mono_from_bytes(audio_bytes: bytes, suffix: str, sr: int) -> np.ndarray:
    with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        seg = AudioSegment.from_file(tmp.name)
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


def extract_logmel(audio: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    n_mels = int(cfg.get("n_mels", 64))
    n_fft = int(cfg.get("n_fft", 512))
    hop_length = int(cfg.get("hop_length", 256))
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels,
                                         n_fft=n_fft, hop_length=hop_length, power=2.0)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.T.astype(np.float32)


def extract_pcen_mel(audio: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    n_mels = int(cfg.get("n_mels", 64))
    n_fft = int(cfg.get("n_fft", 512))
    hop_length = int(cfg.get("hop_length", 256))
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels,
                                         n_fft=n_fft, hop_length=hop_length, power=1.0).astype(np.float32)
    pcen = librosa.pcen(mel * (2**31), sr=sr).astype(np.float32)
    return pcen.T


def extract_spectrogram(audio: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    n_fft = int(cfg.get("stft_n_fft", 512))
    hop_length = int(cfg.get("stft_hop_length", 256))
    win_length = cfg.get("stft_win_length", None)
    win_length = int(win_length) if win_length is not None else None
    S = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    return np.abs(S).T.astype(np.float32)


def extract_mfcc(audio: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    n_mfcc = int(cfg.get("num_mfcc", 13))
    n_fft = int(cfg.get("stft_n_fft", 512))
    hop_length = int(cfg.get("stft_hop_length", 256))
    M = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return M.T.astype(np.float32)


def extract_mfcc_deltas(audio: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    n_mfcc = int(cfg.get("num_mfcc", 13))
    n_fft = int(cfg.get("stft_n_fft", 512))
    hop_length = int(cfg.get("stft_hop_length", 256))
    M = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length).astype(np.float32)
    d1 = librosa.feature.delta(M).astype(np.float32)
    d2 = librosa.feature.delta(M, order=2).astype(np.float32)
    X = np.concatenate([M, d1, d2], axis=0)
    return X.T.astype(np.float32)


def extract_td_spec_stats(audio: np.ndarray, sr: int, cfg: dict):
    frame_len = int(sr * float(cfg["frame_length_sec"]))
    Xf = frame_audio(audio, frame_len_samples=frame_len)
    if Xf.shape[0] == 0:
        return np.zeros((0, 7), dtype=np.float32), frame_len

    out = np.zeros((Xf.shape[0], 7), dtype=np.float32)
    for i in range(Xf.shape[0]):
        fr = Xf[i]
        out[i, 0] = float(np.sqrt(np.mean(fr * fr) + 1e-12))
        out[i, 1] = float(np.mean(np.sign(fr[1:]) != np.sign(fr[:-1]))) if fr.size > 1 else 0.0
        rms = float(np.sqrt(np.mean(fr * fr) + 1e-12))
        peak = float(np.max(np.abs(fr))) if fr.size else 0.0
        out[i, 2] = float(peak / (rms + 1e-12))
        if fr.size >= 3:
            tke = fr[1:-1] * fr[1:-1] - fr[:-2] * fr[2:]
            out[i, 3] = float(np.mean(tke))
        else:
            out[i, 3] = 0.0

        x = fr.astype(np.float32, copy=False)
        win = np.hanning(len(x)).astype(np.float32)
        xw = x * win
        spec = np.abs(np.fft.rfft(xw)) + 1e-12
        freqs = np.fft.rfftfreq(len(xw), d=1.0 / sr).astype(np.float32)
        power = spec * spec
        ps = float(np.sum(power))
        if ps > 0:
            out[i, 4] = float(np.sum(freqs * power) / ps)
            gm = float(np.exp(np.mean(np.log(spec))))
            am = float(np.mean(spec))
            out[i, 5] = float(gm / (am + 1e-12))
            c = np.cumsum(power)
            thr = 0.85 * c[-1]
            idx = int(np.searchsorted(c, thr))
            idx = max(0, min(idx, len(freqs) - 1))
            out[i, 6] = float(freqs[idx])
        else:
            out[i, 4] = 0.0
            out[i, 5] = 0.0
            out[i, 6] = 0.0

    return out.astype(np.float32), frame_len


def extract_features_from_waveform(audio: np.ndarray, sr: int, cfg: dict):
    ftype = str(cfg.get("feature_type", "raw")).lower()

    if ftype == "raw":
        frame_len = int(sr * float(cfg["frame_length_sec"]))
        return frame_audio(audio, frame_len), frame_len

    if ftype == "td_spec_stats":
        return extract_td_spec_stats(audio, sr, cfg)

    if ftype in ("logmel", "pcen_mel", "spectrogram", "mfcc", "mfcc_deltas") and not HAS_LIBROSA:
        raise RuntimeError(f"feature_type='{ftype}' requires librosa")

    if ftype == "logmel":
        return extract_logmel(audio, sr, cfg), int(cfg.get("hop_length", 256))
    if ftype == "pcen_mel":
        return extract_pcen_mel(audio, sr, cfg), int(cfg.get("hop_length", 256))
    if ftype == "spectrogram":
        return extract_spectrogram(audio, sr, cfg), int(cfg.get("stft_hop_length", 256))
    if ftype == "mfcc":
        return extract_mfcc(audio, sr, cfg), int(cfg.get("stft_hop_length", 256))
    if ftype == "mfcc_deltas":
        return extract_mfcc_deltas(audio, sr, cfg), int(cfg.get("stft_hop_length", 256))

    raise ValueError(f"Unsupported feature_type: {ftype}")


def pad_or_trim_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    if len(x) == target_len:
        return x.astype(np.float32, copy=False)
    if len(x) > target_len:
        return x[:target_len].astype(np.float32, copy=False)
    pad = np.zeros((target_len - len(x),), dtype=np.float32)
    return np.concatenate([x.astype(np.float32, copy=False), pad], axis=0)


def pad_or_trim_2d(X: np.ndarray, target_T: int, target_D: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("Expected 2D feature matrix")
    T, D = X.shape

    if D != target_D:
        if D > target_D:
            X = X[:, :target_D]
        else:
            X = np.concatenate([X, np.zeros((T, target_D - D), dtype=np.float32)], axis=1)

    T = X.shape[0]
    if T != target_T:
        if T > target_T:
            X = X[:target_T, :]
        else:
            X = np.concatenate([X, np.zeros((target_T - T, X.shape[1]), dtype=np.float32)], axis=0)

    return X.astype(np.float32, copy=False)


def rolling_mean_5(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    return pd.Series(x).rolling(window=5, min_periods=1, center=True).mean().to_numpy(dtype=np.float32)


def predict_audio_bytes(model, cfg: dict, audio_bytes: bytes, suffix: str, hop_sec: Optional[float]) -> pd.DataFrame:
    sr = int(cfg.get("sample_rate", 44100))
    audio = load_audio_mono_from_bytes(audio_bytes, suffix, sr)

    in_shape = model.input_shape  # (None,T,D,1)
    if not (isinstance(in_shape, (list, tuple)) and len(in_shape) == 4):
        raise ValueError(f"transformer: unexpected input_shape={in_shape}")
    target_T = int(in_shape[1])
    target_D = int(in_shape[2])

    win_sec = float(cfg.get("noise_length_sec", 5.0))
    hop_sec_eff = win_sec if hop_sec is None else float(hop_sec)
    win_len = int(sr * win_sec)
    hop_len = max(1, int(sr * hop_sec_eff))

    rows = []
    for start in range(0, max(1, len(audio)), hop_len):
        chunk = pad_or_trim_1d(audio[start:start + win_len], win_len)
        chunk = transform_audio(chunk, cfg.get("transformation_method", None))

        X2d, hop_in_samples = extract_features_from_waveform(chunk, sr, cfg)
        if str(cfg.get("feature_type", "raw")).lower() != "raw":
            X2d = normalize_feature_matrix(X2d)

        X2d = pad_or_trim_2d(X2d, target_T, target_D)
        X = X2d[..., None].astype(np.float32)

        pred = model.predict(X[None, ...], verbose=0)
        if isinstance(pred, (list, tuple)):
            pred = pred[-1]
        pred = np.asarray(pred, dtype=np.float32)
        if pred.ndim != 3 or pred.shape[1] != target_T or pred.shape[2] != 2:
            raise ValueError(f"transformer: unexpected pred shape {pred.shape}")
        pred = pred[0]

        hop_in_samples = max(1, int(hop_in_samples))
        for i in range(pred.shape[0]):
            t_s = (start + i * hop_in_samples) / sr
            rows.append({"time_sec": float(t_s), "dose_pred": float(pred[i, 0]), "flow_pred": float(pred[i, 1])})

        if start + win_len >= len(audio):
            break

    df = pd.DataFrame(rows)
    if not df.empty:
        df["dose_smooth"] = rolling_mean_5(df["dose_pred"].to_numpy())
        df["flow_smooth"] = rolling_mean_5(df["flow_pred"].to_numpy())
    return df
