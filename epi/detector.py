"""
Deteccao de EPIs e de pessoas num frame.

Etapas (cada uma liga/desliga pelos parametros do Detector):
1. EPIs no frame inteiro, com o modelo treinado (ex.: models/best_v6.pt).
2. Pessoas com um YOLO pre-treinado no COCO. Com um modelo de pose
   (ex.: yolo26n-pose.pt) vem tambem os 17 pontos do corpo, usados para
   saber quais EPIs da para cobrar (epi/conformidade.py). Rastreamento
   opcional: ID fixo por pessoa ao longo do video (TrackTrack sem compensacao
   de movimento da camera: o que menos trocou o ID das pessoas em
   scripts/comparar_rastreadores.py, e com custo de ~1 ms por frame).
3. Refino por recorte (opcional): o modelo de EPI roda de novo em cada
   pessoa recortada (com margem) e ampliada; as caixas voltam para as
   coordenadas do frame e sao unidas as do frame inteiro com NMS por classe.
   Tambem da para usar fatias fixas do frame (estilo SAHI).
4. Associacao: cada EPI vai para a pessoa cuja caixa mais o contem.

Modelos .pt rodam no PyTorch (CPU). Pastas *_openvino_model (ou exportadas
por scripts/exportar_openvino.py) rodam no OpenVINO, na GPU integrada Intel
(intel:gpu) ou na CPU (intel:cpu), com a forma de entrada da exportacao.

Tudo devolve numpy: caixas (N, 4) em xyxy de pixels, scores (N,), classes (N,).
"""

import os
import time

import numpy as np
import torch
import yaml
from torchvision.ops import batched_nms
from ultralytics import YOLO
from ultralytics.trackers.basetrack import BaseTrack

from epi.conformidade import mascaras_pessoas

PESSOA_COCO = 0
# TrackTrack sem compensacao de movimento da camera: o que menos trocou o ID das
# pessoas e o mais barato (scripts/comparar_rastreadores.py; ver o .yaml)
RASTREADOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tracktrack_sem_gmc.yaml")
# margem do recorte em fracao da caixa da pessoa: lados, topo (capacete
# costuma passar da caixa) e base (botas)
MARGEM_RECORTE = (0.15, 0.10, 0.05)


def eh_openvino(caminho):
    return bool(caminho) and os.path.isdir(caminho) and any(f.endswith(".xml") for f in os.listdir(caminho))


def metadados_exportacao(caminho):
    """metadata.yaml da exportacao OpenVINO (None se for .pt)."""
    if not eh_openvino(caminho):
        return None
    with open(os.path.join(caminho, "metadata.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def imgsz_exportado(caminho):
    """Forma de entrada [h, w] gravada na exportacao OpenVINO (None se for .pt)."""
    meta = metadados_exportacao(caminho)
    return meta["imgsz"] if meta else None


def carregar_yolo(caminho):
    # o .pt guarda a tarefa no checkpoint; na pasta OpenVINO ela vem do metadata.yaml
    meta = metadados_exportacao(caminho)
    return YOLO(caminho, task=meta["task"]) if meta else YOLO(caminho)


def vazio():
    return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)


def extrair(resultado):
    b = resultado.boxes
    if b is None or len(b) == 0:
        return vazio()
    return (b.xyxy.cpu().numpy().astype(np.float32), b.conf.cpu().numpy().astype(np.float32),
            b.cls.cpu().numpy().astype(int))


def juntar(*partes):
    partes = [p for p in partes if len(p[0])]
    if not partes:
        return vazio()
    return tuple(np.concatenate(x) for x in zip(*partes))


def nms_por_classe(caixas, scores, classes, iou=0.5):
    if len(caixas) == 0:
        return caixas, scores, classes
    keep = batched_nms(torch.from_numpy(caixas), torch.from_numpy(scores),
                       torch.from_numpy(classes), iou).numpy()
    return caixas[keep], scores[keep], classes[keep]


def expandir(caixa, w, h, margem=MARGEM_RECORTE):
    x1, y1, x2, y2 = caixa
    lado, topo, base = margem
    bw, bh = x2 - x1, y2 - y1
    return (int(max(0, x1 - lado * bw)), int(max(0, y1 - topo * bh)),
            int(min(w, x2 + lado * bw)), int(min(h, y2 + base * bh)))


def associar(caixas_epi, caixas_pessoa, w, h, min_contido=0.5):
    """Indice da pessoa de cada EPI (-1 = nenhuma).

    Usa a fracao da caixa do EPI que cai dentro da caixa (expandida) da
    pessoa; empate (EPI inteiro dentro de duas pessoas sobrepostas) fica
    com a pessoa de caixa menor, que e a mais especifica.
    """
    n = len(caixas_epi)
    if n == 0 or len(caixas_pessoa) == 0:
        return np.full(n, -1)
    p = np.array([expandir(c, w, h) for c in caixas_pessoa], np.float32)
    e = caixas_epi
    ix = np.clip(np.minimum(e[:, None, 2], p[None, :, 2]) - np.maximum(e[:, None, 0], p[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(e[:, None, 3], p[None, :, 3]) - np.maximum(e[:, None, 1], p[None, :, 1]), 0, None)
    area_e = np.maximum((e[:, 2] - e[:, 0]) * (e[:, 3] - e[:, 1]), 1e-6)
    frac = (ix * iy) / area_e[:, None]
    area_p = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
    # arredonda a fracao para o empate cair no criterio da area
    chave = np.round(frac, 2) - 1e-9 * area_p[None, :]
    j = chave.argmax(1)
    return np.where(frac[np.arange(n), j] >= min_contido, j, -1)


class Detector:
    """Pipeline configuravel. Os parametros padrao sao os do app."""

    def __init__(self, modelo_epi, modelo_pessoa=None, imgsz=960, conf=0.25, iou=0.6,
                 intel="gpu", recortes=False, imgsz_recorte=640, conf_pessoa=0.30,
                 imgsz_pessoa=640, tta=False, fatias=None, iou_uniao=0.5,
                 rastreador=RASTREADOR, modelo_epi_recorte=None, intel_pessoa=None,
                 paralelo=False):
        def disp(caminho, alvo=intel):
            return f"intel:{alvo}" if eh_openvino(caminho) else "cpu"

        self.m_epi, self.d_epi = carregar_yolo(modelo_epi), disp(modelo_epi)
        # modelo separado para os recortes: no OpenVINO a forma de entrada
        # e fixa, entao o recorte usa outra exportacao (640x384)
        rec = modelo_epi_recorte or modelo_epi
        self.m_rec = carregar_yolo(rec) if modelo_epi_recorte else self.m_epi
        self.d_rec = disp(rec)
        self.m_pessoa = carregar_yolo(modelo_pessoa) if modelo_pessoa else None
        self.d_pessoa = disp(modelo_pessoa, intel_pessoa or intel) if modelo_pessoa else None
        # paralelo: pessoas e EPIs ao mesmo tempo, em threads (o OpenVINO solta o
        # GIL durante a inferencia); rende mais com cada modelo num dispositivo
        self.paralelo = None
        if paralelo and modelo_pessoa:
            from concurrent.futures import ThreadPoolExecutor
            self.paralelo = ThreadPoolExecutor(max_workers=1)
        self.pose = self.m_pessoa is not None and self.m_pessoa.task == "pose"
        # exportacao OpenVINO estatica so aceita a forma com que foi exportada
        self.imgsz = imgsz_exportado(modelo_epi) or imgsz
        self.imgsz_recorte = imgsz_exportado(rec) or imgsz_recorte
        self.imgsz_pessoa = (imgsz_exportado(modelo_pessoa) if modelo_pessoa else None) or imgsz_pessoa
        self.conf, self.iou, self.recortes = conf, iou, recortes
        self.conf_pessoa, self.tta, self.fatias, self.iou_uniao = conf_pessoa, tta, fatias, iou_uniao
        self.rastreador = rastreador

    # -- etapas ---------------------------------------------------------
    def epis_frame(self, frame):
        r = self.m_epi.predict(frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                               augment=self.tta, device=self.d_epi, verbose=False)[0]
        return extrair(r)

    def pessoas(self, frame, rastrear=False):
        """(caixas, scores, ids, pontos). Sem rastreamento os ids sao -1; sem pose, pontos = None."""
        kw = dict(imgsz=self.imgsz_pessoa, classes=[PESSOA_COCO], device=self.d_pessoa, verbose=False)
        if rastrear:
            # o rastreador precisa das deteccoes fracas (2a etapa de associacao);
            # o corte de confianca fica nos limiares do proprio rastreador
            r = self.m_pessoa.track(frame, persist=True, tracker=self.rastreador, conf=0.1, **kw)[0]
        else:
            r = self.m_pessoa.predict(frame, conf=self.conf_pessoa, **kw)[0]
        c, s, _ = extrair(r)
        ids = r.boxes.id if r.boxes is not None else None
        ids = ids.cpu().numpy().astype(int) if ids is not None else np.full(len(c), -1)
        kps = r.keypoints.data.cpu().numpy() if self.pose and r.keypoints is not None and len(c) else None
        if rastrear:
            ok = s >= self.conf_pessoa * 0.5  # trilha ativa com deteccao fraca continua visivel
            c, s, ids = c[ok], s[ok], ids[ok]
            kps = kps[ok] if kps is not None else None
        return c, s, ids, kps

    def resetar_rastreamento(self, zerar_ids=False):
        """Esquece as trilhas (troca de video ou salto no tempo). Por padrao os IDs continuam
        crescendo, para 'Pessoa #3' nao virar outra pessoa no registro de ocorrencias."""
        pred = getattr(self.m_pessoa, "predictor", None) if self.m_pessoa else None
        for t in getattr(pred, "trackers", []) or []:
            n = BaseTrack._count
            t.reset()
            if not zerar_ids:
                BaseTrack._count = n

    def epis_em_regioes(self, frame, regioes):
        """Roda o modelo de EPI em cada regiao (x1, y1, x2, y2) e devolve as caixas no frame."""
        partes = []
        for x1, y1, x2, y2 in regioes:
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue
            r = self.m_rec.predict(frame[y1:y2, x1:x2], imgsz=self.imgsz_recorte, conf=self.conf,
                                   iou=self.iou, device=self.d_rec, verbose=False)[0]
            c, s, k = extrair(r)
            if len(c):
                partes.append((c + np.array([x1, y1, x1, y1], np.float32), s, k))
        return juntar(*partes)

    def regioes_fatias(self, w, h):
        nx, ny, sobre = self.fatias
        tw, th = w / (nx - (nx - 1) * sobre), h / (ny - (ny - 1) * sobre)
        return [(int(i * tw * (1 - sobre)), int(j * th * (1 - sobre)),
                 int(min(w, i * tw * (1 - sobre) + tw)), int(min(h, j * th * (1 - sobre) + th)))
                for i in range(nx) for j in range(ny)]

    # -- frame completo -------------------------------------------------
    def processar(self, frame, rastrear=False, classes=None):
        """Roda o pipeline num frame BGR. classes = indices de EPI a manter (None = todos)."""
        h, w = frame.shape[:2]
        tempos = {}
        pc, ps, pid, kps = vazio()[0], vazio()[1], np.zeros(0, int), None
        if self.paralelo is not None:
            def pessoas_cronometrado():
                t0 = time.perf_counter()
                return self.pessoas(frame, rastrear), (time.perf_counter() - t0) * 1000

            futuro = self.paralelo.submit(pessoas_cronometrado)
            t = time.perf_counter()
            epis = self.epis_frame(frame)
            tempos["epis_ms"] = (time.perf_counter() - t) * 1000
            (pc, ps, pid, kps), tempos["pessoas_ms"] = futuro.result()
        else:
            t = time.perf_counter()
            if self.m_pessoa is not None:
                pc, ps, pid, kps = self.pessoas(frame, rastrear)
            tempos["pessoas_ms"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            epis = self.epis_frame(frame)
            tempos["epis_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        extras = []
        if self.recortes and len(pc):
            extras.append(self.epis_em_regioes(frame, [expandir(c, w, h) for c in pc]))
        if self.fatias:
            extras.append(self.epis_em_regioes(frame, self.regioes_fatias(w, h)))
        if extras:
            epis = nms_por_classe(*juntar(epis, *extras), iou=self.iou_uniao)
        tempos["refino_ms"] = (time.perf_counter() - t) * 1000

        ec, es, ek = epis
        if classes is not None:
            ok = np.isin(ek, list(classes))
            ec, es, ek = ec[ok], es[ok], ek[ok]
        return {
            "epis": (ec, es, ek),
            "epi_pessoa": associar(ec, pc, w, h),
            "pessoas": (pc, ps, pid),
            "pontos": kps,
            "visiveis": mascaras_pessoas(pc, kps, w, h) if self.m_pessoa is not None else [],
            "tempos": tempos,
        }
