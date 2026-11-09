"""
Servidor do app de demonstracao (substitui o app/streamlit_app.py, que fica
guardado como versao antiga).

Por que nao trava: o video NAO passa pelo Python para ser exibido. O
navegador toca o arquivo original com o <video> (liso, na velocidade
normal) e o servidor so manda, por WebSocket, as deteccoes de cada frame
analisado. O navegador desenha as caixas por cima do frame que esta na tela
(requestVideoFrameCallback). A analise corre um pouco A FRENTE do video; se
o hardware nao der conta de todos os frames, ela analisa 1 a cada k (passo
adaptativo) e o navegador interpola as caixas das pessoas entre um frame
analisado e o seguinte. O video nunca fica em camera lenta.

Camera ao vivo: o navegador captura a webcam, manda cada frame em JPEG pelo
mesmo WebSocket e desenha a resposta por cima da imagem da camera.

Rotas:
  GET  /                  interface (app/static/index.html)
  GET  /api/estado        videos, modelos, classes e dispositivo
  GET  /videos/<nome>     arquivo de video (aceita Range, para poder pular)
  POST /api/videos        envia um video novo para a pasta video/
  WS   /ws                sessao de analise (protocolo na classe Sessao)

Uso (a partir da pasta TCC):
    venv_win\\Scripts\\python app/servidor.py
    e abrir http://localhost:8000
"""

import asyncio
import glob
import json
import math
import os
import re
import sys
import threading
import time
import warnings

warnings.filterwarnings("ignore")
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RAIZ)  # caminhos relativos a pasta TCC, como nos scripts
sys.path.insert(0, RAIZ)
os.environ.setdefault("YOLO_OFFLINE", "true")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from epi import CLASSES, NOMES_PT  # noqa: E402
from epi.detector import Detector  # noqa: E402

PASTA_VIDEOS = "video"
PASTA_OV = "models/openvino"
PASTA_STATIC = os.path.join("app", "static")
EXTENSOES = (".mp4", ".webm", ".m4v", ".mov")  # o que o navegador consegue tocar
PESSOA_PT = "yolo26n-pose.pt"
PORTA = 8000

# nomes amigaveis; "padrao" abre selecionado. A v5 e o padrao porque no teste
# real, no limiar do app (conf 0.40), teve precisao e recall maiores que a v6
# (results/avaliacao_teste_real/estrategias/RESUMO.md)
MODELOS = {
    "v5": {"rotulo": "v5 — maior mAP50 e F1 no teste real", "pt": "models/best.pt", "padrao": True},
    "v6": {"rotulo": "v6 — treinada em 960 px", "pt": "models/best_v6.pt"},
    "v4": {"rotulo": "v4", "pt": "models/best_24-09.pt"},
    "v2": {"rotulo": "v2 (linha de base)", "pt": "models/best_antigo.pt"},
}
PADRAO = next(k for k, v in MODELOS.items() if v.get("padrao"))
CONF_ENVIO = 0.15      # o navegador filtra pela confianca escolhida (>= 0.15)
ADIANTAMENTO_S = 1.5   # quanto a analise pode correr a frente do video
CORES = ["#facc15", "#fb923c", "#38bdf8", "#a78bfa", "#34d399"]  # mesma ordem de CLASSES


def gpu_intel():
    try:
        import openvino as ov
        core = ov.Core()
        if "GPU" in core.available_devices:
            return core.get_property("GPU", "FULL_DEVICE_NAME")
    except Exception:
        pass
    return None


GPU = gpu_intel()


def caminhos(chave):
    """Arquivos do modelo: OpenVINO (pastas de scripts/exportar_openvino.py) se houver, senao .pt."""
    ov = {
        "epi": os.path.join(PASTA_OV, f"{chave}_frame_544x960_openvino_model"),
        "recorte": os.path.join(PASTA_OV, f"{chave}_recorte_640x384_openvino_model"),
        "pessoa": os.path.join(PASTA_OV, "pessoa_384x640_openvino_model"),
    }
    if all(os.path.isdir(p) for p in ov.values()):
        return ov, f"OpenVINO — {GPU or 'CPU'}"
    return {"epi": MODELOS[chave]["pt"], "recorte": None, "pessoa": PESSOA_PT}, "PyTorch — CPU"


_detectores = {}
_lock_det = threading.Lock()   # um frame por vez no pipeline (um usuario so, demo local)


def detector(chave, recortes):
    with _lock_det:
        if (chave, recortes) not in _detectores:
            c, _ = caminhos(chave)
            # paralelo: pessoas e EPIs ao mesmo tempo na GPU (37 ms/frame contra 43 ms em sequencia)
            _detectores[(chave, recortes)] = Detector(
                c["epi"], c["pessoa"], conf=CONF_ENVIO, iou=0.6, intel="gpu" if GPU else "cpu",
                recortes=recortes, modelo_epi_recorte=c["recorte"], paralelo=True)
        return _detectores[(chave, recortes)]


def info_video(caminho):
    cap = cv2.VideoCapture(caminho)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"nome": os.path.basename(caminho), "fps": fps, "frames": n, "duracao": n / fps if fps else 0,
            "w": w, "h": h, "mtime": os.path.getmtime(caminho)}


def listar_videos():
    vids = [p for p in glob.glob(os.path.join(PASTA_VIDEOS, "*")) if p.lower().endswith(EXTENSOES)]
    return sorted((info_video(p) for p in vids), key=lambda v: -v["mtime"])


def caminho_video(nome):
    p = os.path.realpath(os.path.join(PASTA_VIDEOS, nome))
    if os.path.dirname(p) != os.path.realpath(PASTA_VIDEOS) or not os.path.isfile(p):
        raise HTTPException(404, "video nao encontrado")
    return p


def pacote(r, w, h, i):
    """Resultado do pipeline -> mensagem compacta (coordenadas normalizadas 0-1)."""
    pc, ps, pid = r["pessoas"]
    ec, es, ek = r["epis"]
    kps = r["pontos"]
    esc = np.array([w, h, w, h], np.float32)
    pessoas = []
    for j in range(len(pc)):
        p = [int(pid[j]), *np.round(pc[j] / esc, 4).tolist(), round(float(ps[j]), 3), r["visiveis"][j]]
        if kps is not None:
            k = kps[j]
            p.append(np.round(np.column_stack([k[:, 0] / w, k[:, 1] / h, k[:, 2]]), 3).ravel().tolist())
        pessoas.append(p)
    dono = r["epi_pessoa"]
    epis = [[int(ek[j]), *np.round(ec[j] / esc, 4).tolist(), round(float(es[j]), 3),
             int(pid[dono[j]]) if dono[j] >= 0 else -1] for j in range(len(ec))]
    return {"tipo": "det", "i": i, "p": pessoas, "e": epis}


class Sessao:
    """Analise de um video para uma conexao WebSocket (roda numa thread).

    Navegador -> servidor (JSON):
      {"tipo": "abrir", "video": nome, "modelo": "v5", "recortes": false, "t": 0, "g": 1}
      {"tipo": "tempo", "t": 12.3, "pausado": false}   a cada ~250 ms
      {"tipo": "pular", "t": 40.0, "g": 2}              o usuario arrastou a barra
      ("g" = geracao: volta em cada deteccao, para o navegador descartar as
       que ainda estavam em transito quando ele pulou)
      {"tipo": "camera", "modelo": "v5", "recortes": false}   (a analise do video para)
      binario: 4 bytes (numero do frame, uint32) + JPEG   (modo camera)
    Servidor -> navegador:
      {"tipo": "pronto", ...}  {"tipo": "det", "i": frame, "p": [...], "e": [...]}
      {"tipo": "status", "fps": ..., "passo": k, "ms": {...}}  {"tipo": "erro", "msg": ...}
    """

    def __init__(self, enviar):
        self.enviar = enviar
        self.lock = threading.Lock()
        self.pedido = None
        self.t_cli, self.pausado = 0.0, True
        self.fim = threading.Event()
        self.cap = None
        self.det = None
        self.geracao = 0
        threading.Thread(target=self._loop, daemon=True).start()

    # -- mensagens do navegador ----------------------------------------
    def receber(self, msg):
        with self.lock:
            if msg["tipo"] in ("abrir", "pular"):
                if msg["tipo"] == "pular" and self.pedido and self.pedido["tipo"] == "abrir":
                    # "abrir" ainda nao atendido: o pular nao pode apaga-lo (senao a
                    # analise segue no video antigo); o abrir so passa a comecar no novo tempo
                    self.pedido = {**self.pedido, "t": msg["t"], "g": msg.get("g", self.pedido.get("g"))}
                else:
                    self.pedido = msg
                self.t_cli = float(msg.get("t", 0))
            elif msg["tipo"] == "tempo":
                self.t_cli, self.pausado = float(msg["t"]), bool(msg.get("pausado"))
            elif msg["tipo"] == "parar":  # modo camera: a analise do video para
                self.pedido = msg

    def fechar(self):
        self.fim.set()

    # -- thread de analise ---------------------------------------------
    def _abrir(self, msg):
        self.enviar({"tipo": "carregando", "msg": "Carregando modelos..."})
        chave, recortes = msg.get("modelo", PADRAO), bool(msg.get("recortes"))
        self.det = detector(chave, recortes)
        self.cap = cv2.VideoCapture(caminho_video(msg["video"]))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.w, self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.pos = 0
        self.ema, self.passo, self.acum = None, 1, 0.0
        self.det.resetar_rastreamento()
        self.prox = int(float(msg.get("t", 0)) * self.fps)
        self.enviar({"tipo": "pronto", "fps": self.fps, "frames": self.total, "w": self.w, "h": self.h,
                     "modelo": chave, "dispositivo": caminhos(chave)[1], "recortes": recortes})

    def _ler(self, i):
        # leitura sequencial: so faz seek (caro em H.264) quando precisa voltar ou saltar longe
        if i < self.pos or i - self.pos > 2 * self.fps:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            self.pos = i
        while self.pos < i:
            self.cap.grab()
            self.pos += 1
        ok, frame = self.cap.read()
        self.pos += 1
        return frame if ok else None

    def _loop(self):
        n_status, t_status, soma = 0, time.perf_counter(), {}
        while not self.fim.is_set():
            with self.lock:
                pedido, self.pedido = self.pedido, None
                t_cli, pausado = self.t_cli, self.pausado
            try:
                if pedido:
                    self.geracao = pedido.get("g", 0)
                if pedido and pedido["tipo"] == "abrir":
                    self._abrir(pedido)
                elif pedido and pedido["tipo"] == "pular" and self.cap is not None:
                    self.det.resetar_rastreamento()
                    self.prox = int(float(pedido["t"]) * self.fps)
                elif pedido and pedido["tipo"] == "parar" and self.cap is not None:
                    self.cap.release()
                    self.cap = None
            except Exception as e:  # noqa: BLE001 - qualquer falha vai para a tela
                self.enviar({"tipo": "erro", "msg": str(e)})
                self.cap = None
            if self.cap is None:
                time.sleep(0.02)
                continue

            atual = int(t_cli * self.fps)
            if self.prox < atual:  # a analise ficou para tras do video: pula para perto dele
                self.prox = atual + max(1, int(0.2 * self.fps))
            if self.prox >= self.total or self.prox > atual + ADIANTAMENTO_S * self.fps:
                time.sleep(0.005)
                continue

            t0 = time.perf_counter()
            frame = self._ler(self.prox)
            if frame is None:
                self.prox = self.total
                continue
            with _lock_det:
                r = self.det.processar(frame, rastrear=True)
            ms = (time.perf_counter() - t0) * 1000
            msg = pacote(r, self.w, self.h, self.prox)
            msg["k"], msg["g"] = math.ceil(self.passo), self.geracao
            self.enviar(msg)

            # passo adaptativo e fracionario: se cada frame leva 37 ms num video
            # de 30 fps, o passo medio fica ~1,2 (analisa ~5 de cada 6 frames),
            # em vez de arredondar para 2 e jogar fora metade dos frames
            self.ema = ms if self.ema is None else 0.85 * self.ema + 0.15 * ms
            self.passo = 1.0 if pausado else max(1.0, self.ema / 1000 * self.fps * 1.1)
            self.acum += self.passo
            pulo = max(1, math.floor(self.acum))
            self.acum -= pulo
            self.prox += pulo

            n_status += 1
            for k, v in {**r["tempos"], "total_ms": ms}.items():
                soma[k] = soma.get(k, 0) + v
            agora = time.perf_counter()
            if agora - t_status >= 1.0:
                self.enviar({"tipo": "status", "fps": n_status / (agora - t_status), "passo": round(self.passo, 2),
                             "ms": {k: round(v / n_status, 1) for k, v in soma.items()}})
                n_status, t_status, soma = 0, agora, {}


app = FastAPI(title="Monitor de EPI")
app.mount("/static", StaticFiles(directory=PASTA_STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(PASTA_STATIC, "index.html"))


@app.get("/api/estado")
def estado():
    return {
        "videos": listar_videos(),
        "modelos": [{"chave": k, "rotulo": v["rotulo"], "padrao": v.get("padrao", False),
                     "dispositivo": caminhos(k)[1]} for k, v in MODELOS.items() if os.path.exists(v["pt"])],
        "classes": [{"id": i, "nome": c, "pt": NOMES_PT[c], "cor": CORES[i]} for i, c in enumerate(CLASSES)],
        "gpu": GPU,
    }


@app.get("/videos/{nome:path}")
def video(nome: str):
    return FileResponse(caminho_video(nome))


@app.post("/api/videos")
async def enviar_video(arquivo: UploadFile):
    nome = re.sub(r"[^\w.\- ()]", "_", os.path.basename(arquivo.filename or "video.mp4"))
    if not nome.lower().endswith(EXTENSOES):
        raise HTTPException(400, f"formato nao suportado pelo navegador; use {', '.join(EXTENSOES)}")
    destino = os.path.join(PASTA_VIDEOS, nome)
    with open(destino, "wb") as f:
        while bloco := await arquivo.read(1 << 20):
            f.write(bloco)
    return info_video(destino)


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()
    fila = asyncio.Queue()

    def enviar(msg):  # chamado da thread de analise
        loop.call_soon_threadsafe(fila.put_nowait, json.dumps(msg, separators=(",", ":")))

    async def remetente():
        while True:
            await websocket.send_text(await fila.get())

    sessao = Sessao(enviar)
    tarefa = asyncio.create_task(remetente())
    camera = {"det": None}
    try:
        while True:
            m = await websocket.receive()
            if m.get("type") == "websocket.disconnect":
                break
            if m.get("text"):
                msg = json.loads(m["text"])
                if msg["tipo"] == "camera":
                    sessao.receber({"tipo": "parar"})
                    camera["det"] = await asyncio.to_thread(detector, msg.get("modelo", PADRAO), bool(msg.get("recortes")))
                    camera["det"].resetar_rastreamento()
                    enviar({"tipo": "pronto", "camera": True, "modelo": msg.get("modelo", PADRAO),
                            "dispositivo": caminhos(msg.get("modelo", PADRAO))[1]})
                else:
                    camera["det"] = None
                    sessao.receber(msg)
            elif m.get("bytes") and camera["det"] is not None:
                dados = m["bytes"]
                i = int.from_bytes(dados[:4], "little")
                frame = cv2.imdecode(np.frombuffer(dados[4:], np.uint8), cv2.IMREAD_COLOR)
                t0 = time.perf_counter()

                def rodar():
                    with _lock_det:
                        return camera["det"].processar(frame, rastrear=True)

                r = await asyncio.to_thread(rodar)
                msg = pacote(r, frame.shape[1], frame.shape[0], i)
                msg["ms"] = round((time.perf_counter() - t0) * 1000, 1)
                enviar(msg)
    except WebSocketDisconnect:
        pass
    finally:
        sessao.fechar()
        tarefa.cancel()


def aquecer():
    """Carrega e compila o modelo padrao ao subir o servidor (a 1a compilacao na GPU leva alguns segundos)."""
    chave = PADRAO
    det = detector(chave, False)
    det.processar(np.zeros((540, 960, 3), np.uint8))
    print(f"  modelo {chave} pronto ({caminhos(chave)[1]})", flush=True)


if __name__ == "__main__":
    print(f"Monitor de EPI -> http://localhost:{PORTA}", flush=True)
    threading.Thread(target=aquecer, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=PORTA, log_level="warning")
