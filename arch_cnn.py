from typing import Optional
import numpy as np
import pandas as pd
import core

ARCH_NAME = "cnn"

def predict_audio_bytes(model, cfg: dict, audio_bytes: bytes, suffix: str, hop_sec: Optional[float]) -> pd.DataFrame:
    mt = str(cfg.get("model_type", "")).lower()
    if mt == "cnn_lstm":
        return _predict_frame(model, cfg, audio_bytes, suffix, hop_sec)
    if mt in ("cnn2d_resnet", "crnn"):
        return _predict_window(model, cfg, audio_bytes, suffix, hop_sec)
    raise ValueError(f"cnn handler got unsupported model_type={mt}")

def _predict_frame(model, cfg: dict, audio_bytes: bytes, suffix: str, hop_sec: Optional[float]) -> pd.DataFrame:
    sr = int(cfg.get("sample_rate", 44100))
    audio = core.load_audio_mono_from_bytes(audio_bytes, suffix, sr)

    in_shape = model.input_shape  # (None, D, 1)
    if not (isinstance(in_shape, (list, tuple)) and len(in_shape) == 3):
        raise ValueError(f"cnn_lstm: unexpected input_shape={in_shape}")
    target_D = int(in_shape[1])

    win_sec = float(cfg.get("noise_length_sec", 5.0))
    hop_sec_eff = win_sec if hop_sec is None else float(hop_sec)
    win_len = int(sr * win_sec)
    hop_len = max(1, int(sr * hop_sec_eff))

    rows = []
    for start in range(0, max(1, len(audio)), hop_len):
        chunk = core.pad_or_trim_1d(audio[start:start + win_len], win_len)
        chunk = core.transform_audio(chunk, cfg.get("transformation_method", None))

        X2d, hop_in_samples = core.extract_features_from_waveform(chunk, sr, cfg)
        if str(cfg.get("feature_type", "raw")).lower() != "raw":
            X2d = core.normalize_feature_matrix(X2d)

        if X2d.shape[1] != target_D:
            X2d = core.pad_or_trim_2d(X2d, X2d.shape[0], target_D)

        X = X2d[..., None].astype(np.float32)
        pred = model.predict(X, verbose=0)
        if isinstance(pred, (list, tuple)):
            pred = pred[-1]
        pred = np.asarray(pred, dtype=np.float32)
        if pred.ndim != 2 or pred.shape[1] != 2:
            raise ValueError(f"cnn_lstm: unexpected pred shape {pred.shape}")

        hop_in_samples = max(1, int(hop_in_samples))
        for i in range(pred.shape[0]):
            t_s = (start + i * hop_in_samples) / sr
            rows.append({"time_sec": float(t_s), "dose_pred": float(pred[i, 0]), "flow_pred": float(pred[i, 1])})

        if start + win_len >= len(audio):
            break

    df = pd.DataFrame(rows)
    if not df.empty:
        df["dose_smooth"] = core.rolling_mean_5(df["dose_pred"].to_numpy())
        df["flow_smooth"] = core.rolling_mean_5(df["flow_pred"].to_numpy())
    return df

def _predict_window(model, cfg: dict, audio_bytes: bytes, suffix: str, hop_sec: Optional[float]) -> pd.DataFrame:
    sr = int(cfg.get("sample_rate", 44100))
    audio = core.load_audio_mono_from_bytes(audio_bytes, suffix, sr)

    in_shape = model.input_shape  # (None, T, D, 1)
    if not (isinstance(in_shape, (list, tuple)) and len(in_shape) == 4):
        raise ValueError(f"cnn(window): unexpected input_shape={in_shape}")
    target_T = int(in_shape[1])
    target_D = int(in_shape[2])

    win_sec = float(cfg.get("noise_length_sec", 5.0))
    hop_sec_eff = win_sec if hop_sec is None else float(hop_sec)
    win_len = int(sr * win_sec)
    hop_len = max(1, int(sr * hop_sec_eff))

    rows = []
    for start in range(0, max(1, len(audio)), hop_len):
        chunk = core.pad_or_trim_1d(audio[start:start + win_len], win_len)
        chunk = core.transform_audio(chunk, cfg.get("transformation_method", None))

        X2d, hop_in_samples = core.extract_features_from_waveform(chunk, sr, cfg)
        if str(cfg.get("feature_type", "raw")).lower() != "raw":
            X2d = core.normalize_feature_matrix(X2d)

        X2d = core.pad_or_trim_2d(X2d, target_T, target_D)
        X = X2d[..., None].astype(np.float32)

        pred = model.predict(X[None, ...], verbose=0)
        if isinstance(pred, (list, tuple)):
            pred = pred[-1]
        pred = np.asarray(pred, dtype=np.float32)
        if pred.ndim != 3 or pred.shape[1] != target_T or pred.shape[2] != 2:
            raise ValueError(f"cnn(window): unexpected pred shape {pred.shape}")
        pred = pred[0]

        hop_in_samples = max(1, int(hop_in_samples))
        for i in range(pred.shape[0]):
            t_s = (start + i * hop_in_samples) / sr
            rows.append({"time_sec": float(t_s), "dose_pred": float(pred[i, 0]), "flow_pred": float(pred[i, 1])})

        if start + win_len >= len(audio):
            break

    df = pd.DataFrame(rows)
    if not df.empty:
        df["dose_smooth"] = core.rolling_mean_5(df["dose_pred"].to_numpy())
        df["flow_smooth"] = core.rolling_mean_5(df["flow_pred"].to_numpy())
    return df
