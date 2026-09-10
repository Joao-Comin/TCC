"""
Pagina basica: cola um link do YouTube, o app baixa o video e roda o modelo
YOLO treinado (best.pt) para identificar os EPIs, mostrando cada frame
analisado na hora (ao vivo), em vez de processar tudo e so mostrar no final.

Uso:
    venv_win\\Scripts\\streamlit run app/streamlit_app.py
"""

import os

import cv2
import streamlit as st
import yt_dlp
from ultralytics import YOLO

MODEL_PATH = "models/best.pt"
DOWNLOADS_DIR = "downloads"

st.set_page_config(page_title="Detector de EPI", layout="centered")
st.title("Detector de EPI em video do YouTube")
st.caption(
    "Cola o link de um video do YouTube (ex: chao de fabrica) e o modelo "
    "treinado identifica capacete, colete, luvas, botas e oculos de protecao "
    "conforme cada frame e analisado."
)


@st.cache_resource
def load_model(path):
    return YOLO(path)


def download_youtube_video(url: str) -> str:
    os.makedirs(DOWNLOADS_DIR, exist_ok=True)
    out_template = os.path.join(DOWNLOADS_DIR, "%(id)s.%(ext)s")

    ydl_opts = {
        # so precisamos da faixa de video (deteccao nao usa audio) - assim nao
        # depende de existir um formato ja pronto com audio+video juntos, nem
        # de ffmpeg pra mesclar faixas separadas
        "format": "bestvideo[ext=mp4][height<=480]/bestvideo[height<=480]/bestvideo/best",
        "outtmpl": out_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        # YouTube exige resolver um desafio via JS para liberar todos os
        # formatos; sem isso a extracao trava/falha. Usa o Node.js do sistema.
        "js_runtimes": {"node": {}},
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)


def run_live_detection(model, video_path: str, conf: float, vid_stride: int, frame_slot, status_slot):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = 0
    processed = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % vid_stride == 0:
                results = model.predict(frame, conf=conf, verbose=False)
                annotated_bgr = results[0].plot()
                frame_slot.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), channels="RGB")
                processed += 1
                status_slot.text(f"Analisando frame {frame_idx + 1}/{total_frames}...")

            frame_idx += 1
    finally:
        cap.release()

    status_slot.text(f"Concluido - {processed} frames analisados de {total_frames} totais.")


if not os.path.exists(MODEL_PATH):
    st.error(
        f"Modelo nao encontrado em '{MODEL_PATH}'. Baixe o `best.pt` do seu "
        "Google Drive (MyDrive/TCC/runs/train-colab-v2/weights/best.pt) e "
        "coloque nesse caminho antes de usar a pagina."
    )
    st.stop()

youtube_url = st.text_input("Link do YouTube", placeholder="https://www.youtube.com/watch?v=...")

col1, col2 = st.columns(2)
with col1:
    conf = st.slider("Confianca minima", 0.1, 0.9, 0.4, 0.05)
with col2:
    vid_stride = st.slider(
        "Pular frames (maior = mais rapido, menos suave)", 1, 10, 2
    )

if st.button("Processar video", type="primary", disabled=not youtube_url):
    try:
        with st.spinner("Baixando video do YouTube..."):
            video_path = download_youtube_video(youtube_url)

        st.success(f"Video baixado: {os.path.basename(video_path)}")

        model = load_model(MODEL_PATH)

        frame_slot = st.empty()
        status_slot = st.empty()
        run_live_detection(model, video_path, conf, vid_stride, frame_slot, status_slot)

    except Exception as e:
        st.error(f"Erro: {e}")
