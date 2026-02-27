import os
import io
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict

import streamlit as st
import pandas as pd
import numpy as np

# твой inference core
from infer_core import (
    run_inference,
    resolve_inputs,
    build_effective_cfg,
    parse_model_filename,
)

# -------------------------
# Git helpers
# -------------------------
def ensure_repo(repo_url: str, branch: str, local_dir: Path) -> Path:
    """
    Клонирует или подтягивает репо в local_dir.
    Использует системный git.
    """
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    if (local_dir / ".git").exists():
        # pull
        os.system(f'git -C "{local_dir}" fetch --all --prune')
        os.system(f'git -C "{local_dir}" checkout "{branch}"')
        os.system(f'git -C "{local_dir}" pull')
    else:
        os.system(f'git clone --branch "{branch}" --single-branch "{repo_url}" "{local_dir}"')
    return local_dir

def find_keras_models(repo_path: Path) -> List[Path]:
    return sorted(repo_path.rglob("*.keras"))

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Inhale inference", layout="wide")

st.title("Inhale: inference по 1–3 моделям из Git")

with st.sidebar:
    st.header("Источник моделей (Git)")
    repo_url = st.text_input("Repo URL", value="")
    branch = st.text_input("Branch", value="main")
    repo_subdir = st.text_input("Subdir (optional)", value="")  # например "inhale_model_runs_grid"
    refresh = st.button("Обновить / подтянуть репо")

# кеш-локальная папка для репо
CACHE_ROOT = Path.home() / ".cache" / "inhale_streamlit"
REPO_DIR = CACHE_ROOT / "repo"

if refresh:
    if not repo_url.strip():
        st.sidebar.error("Укажи Repo URL")
    else:
        with st.sidebar:
            st.info("Обновляю репозиторий...")
        ensure_repo(repo_url.strip(), branch.strip(), REPO_DIR)
        st.sidebar.success("Готово")

# попытка использовать уже существующий клон без refresh
if (REPO_DIR / ".git").exists():
    repo_path = REPO_DIR
    if repo_subdir.strip():
        repo_path = repo_path / repo_subdir.strip()

    if repo_path.exists():
        models = find_keras_models(repo_path)
    else:
        models = []
else:
    models = []

st.subheader("Выбор моделей (до 3)")

if not models:
    st.warning("Модели не найдены. Укажи Repo URL и нажми «Обновить / подтянуть репо».")
    st.stop()

# красивое отображение: относительный путь
model_options = {str(p.relative_to(REPO_DIR)): p for p in models}

selected = st.multiselect(
    "Выбери модели",
    options=list(model_options.keys()),
    default=[],
)

if len(selected) > 3:
    st.error("Можно выбрать максимум 3 модели.")
    st.stop()

st.subheader("Аудио")

uploaded = st.file_uploader(
    "Загрузи 1+ аудиофайлов",
    type=["wav", "mp3", "m4a", "flac", "ogg"],
    accept_multiple_files=True
)

hop_sec = st.number_input(
    "Window hop (сек), пусто = hop = noise_length_sec из конфига",
    min_value=0.01,
    value=5.0,
    step=0.1
)

use_default_hop = st.checkbox("Использовать hop=noise_length_sec (игнорировать поле выше)", value=True)
hop_arg = None if use_default_hop else float(hop_sec)

run_btn = st.button("Запустить предсказания", type="primary", disabled=(len(selected) == 0 or len(uploaded) == 0))

# -------------------------
# Execute
# -------------------------
if run_btn:
    # сохраняем загруженные аудио во временную папку (потому что run_inference ждёт пути)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        audio_paths = []
        for uf in uploaded:
            out = tmpdir / uf.name
            out.write_bytes(uf.getbuffer())
            audio_paths.append(str(out))

        # результаты по каждой модели
        for rel_model_path in selected:
            model_path = model_options[rel_model_path]
            st.markdown("---")
            st.markdown(f"### Модель: `{rel_model_path}`")

            # печатаем inferred настройки
            try:
                cfg = build_effective_cfg(str(model_path), overrides_from_name=True)
                parsed = parse_model_filename(str(model_path))
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("arch", parsed["model_type"])
                col2.metric("feature", parsed["feature_type"])
                col3.metric("transform", str(parsed["transformation_method"]))
                col4.metric("train_unit", cfg.get("train_unit", ""))
            except Exception as e:
                st.error(f"Не смог собрать effective cfg: {type(e).__name__}: {e}")
                continue

            # отдельная папка вывода под модель (во временной директории)
            out_dir = tmpdir / "out" / rel_model_path.replace("/", "__").replace("\\", "__")
            out_dir.mkdir(parents=True, exist_ok=True)

            with st.spinner("Считаю..."):
                try:
                    paths = run_inference(
                        model_path=str(model_path),
                        audio_inputs=audio_paths,
                        out_dir=str(out_dir),
                        window_hop_sec=hop_arg,
                    )
                except Exception as e:
                    st.error(f"Inference упал: {type(e).__name__}: {e}")
                    continue

            # читаем CSV
            df_frames = pd.read_csv(paths["frames_csv"])
            df_sum = pd.read_csv(paths["summary_csv"])

            # показываем summary
            st.write("**Summary по окнам**")
            st.dataframe(df_sum, use_container_width=True)

            # графики dose/flow по времени (агрегируем по time_sec)
            st.write("**Таймлайн (per-frame)**")
            # можно фильтровать по аудиофайлу
            audios = sorted(df_frames["audio_path"].unique().tolist())
            audio_sel = st.selectbox("Аудио файл", audios, key=f"audio_sel::{rel_model_path}")

            dfp = df_frames[df_frames["audio_path"] == audio_sel].copy()
            dfp = dfp.sort_values("time_sec")

            st.line_chart(dfp.set_index("time_sec")[["dose_pred"]])
            st.line_chart(dfp.set_index("time_sec")[["flow_pred"]])

            # download buttons
            st.download_button(
                "Скачать frames CSV",
                data=Path(paths["frames_csv"]).read_bytes(),
                file_name=Path(paths["frames_csv"]).name,
                mime="text/csv",
                key=f"dl_frames::{rel_model_path}"
            )
            st.download_button(
                "Скачать summary CSV",
                data=Path(paths["summary_csv"]).read_bytes(),
                file_name=Path(paths["summary_csv"]).name,
                mime="text/csv",
                key=f"dl_sum::{rel_model_path}"
            )
            st.download_button(
                "Скачать effective cfg JSON",
                data=Path(paths["cfg_json"]).read_bytes(),
                file_name=Path(paths["cfg_json"]).name,
                mime="application/json",
                key=f"dl_cfg::{rel_model_path}"
            )
