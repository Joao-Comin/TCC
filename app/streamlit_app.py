"""
Pagina basica: roda o modelo YOLO treinado (best.pt) sobre um video local
(pasta video/), em loop continuo, mostrando cada frame analisado na hora
(ao vivo) com os EPIs marcados e a confianca de cada deteccao. Na barra
lateral da para trocar a versao do modelo, o video e a confianca minima.

Uso:
    venv_win\\Scripts\\streamlit run app/streamlit_app.py

Para parar: feche a aba do navegador ou aperte Ctrl+C no terminal (o loop e
continuo e nao para sozinho).
"""

import glob
import os
import time

import cv2
import streamlit as st
from ultralytics import YOLO

MODEL_PATH = "models/best.pt"
VIDEO_DIR = "video"

# nomes amigaveis para o seletor; arquivos fora da lista aparecem pelo nome
MODEL_LABELS = {
    "models/best.pt": "v5 (atual)",
    "models/best_v6.pt": "v6 (960, mais precisa)",
    "models/best_24-09.pt": "v4",
    "models/best_antigo.pt": "v2",
}

CONF = 0.40
IMGSZ = 960

st.set_page_config(page_title="Detector de EPI", layout="centered")
st.title("Detector de EPI - video em loop")


@st.cache_resource
def load_model(path):
    return YOLO(path)


def list_local_videos():
    exts = ("*.mp4", "*.avi", "*.mov", "*.mkv")
    candidates = []
    for ext in exts:
        candidates.extend(glob.glob(os.path.join(VIDEO_DIR, ext)))
    # o adicionado mais recentemente primeiro (vira o padrao do seletor)
    return sorted(candidates, key=os.path.getmtime, reverse=True)


def run_live_loop(model, video_path: str, frame_slot, conf: float):
    while True:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_time = time.time()
        try:
            while True:
                # sempre pega o frame que deveria estar passando AGORA no
                # tempo real do video, pulando os que ficaram pra tras se o
                # processamento estiver mais lento que o video - mantem a
                # velocidade de reproducao real em vez de acumular atraso
                target_frame = int((time.time() - start_time) * fps)
                if target_frame >= total_frames:
                    break

                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(frame, conf=conf, imgsz=IMGSZ, verbose=False)
                annotated_bgr = results[0].plot()
                frame_slot.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), channels="RGB")
        finally:
            cap.release()


if not os.path.exists(MODEL_PATH):
    st.error(f"Modelo nao encontrado em '{MODEL_PATH}'.")
    st.stop()

videos = list_local_videos()
if not videos:
    st.error(f"Nenhum video encontrado na pasta '{VIDEO_DIR}/'.")
    st.stop()
video_path = st.sidebar.selectbox("Video", videos, format_func=os.path.basename)

# seletor para comparar versoes do modelo no mesmo video
# ordem do MODEL_LABELS (o atual primeiro, vira o padrao); desconhecidos no fim
ordem = list(MODEL_LABELS)
modelos = sorted((p.replace("\\", "/") for p in glob.glob("models/*.pt")),
                 key=lambda p: (ordem.index(p) if p in ordem else len(ordem), p))
modelo_path = st.sidebar.selectbox(
    "Modelo", modelos, format_func=lambda p: MODEL_LABELS.get(p, os.path.basename(p))
)
conf = st.sidebar.slider("Confianca minima", 0.05, 0.90, CONF, 0.05)
st.caption(f"Video: {os.path.basename(video_path)}")

if st.button("Iniciar deteccao", type="primary"):
    model = load_model(modelo_path)
    frame_slot = st.empty()
    run_live_loop(model, video_path, frame_slot, conf)
