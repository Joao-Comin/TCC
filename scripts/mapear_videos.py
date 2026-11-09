"""
Passa pelos videos da pasta video/ (1 amostra por segundo) e responde duas
perguntas de metodo:

1) De qual video vieram os frames ja anotados (novos_frames_dataset:
   train_extra + real_world_test)? Cada frame anotado e comparado com todas
   as amostras dos videos (correlacao de miniaturas 64x36 em cinza); acima de
   0.90 considera-se que o frame saiu daquele video. Serve para montar um
   teste real NOVO com um video que nunca entrou no treino.

2) Onde estao os planos continuos (sem corte de cena)? Videos do YouTube/TV
   sao editados e trocam de cena a cada poucos segundos; um rastreador nao
   deve manter o ID atraves de um corte. Para comparar rastreadores
   (scripts/comparar_rastreadores.py) so servem planos longos com pessoas.
   Corte = correlacao entre amostras vizinhas < 0.5.

Saidas em results/mapa_videos/:
  origem_frames_anotados.csv   frame anotado -> video, segundo, correlacao
  planos.csv                   planos >= 8 s: video, inicio, fim, pessoas (pose, mediana)

Uso (a partir da pasta TCC):
    venv_win\\Scripts\\python scripts/mapear_videos.py
"""

import csv
import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("YOLO_OFFLINE", "true")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from epi.detector import carregar_yolo  # noqa: E402

SAIDA = "results/mapa_videos"
ANOTADOS = ["novos_frames_dataset/train_extra/images/*_rep0.*", "novos_frames_dataset/real_world_test/images/*"]
LIMIAR_MESMO = 0.90
LIMIAR_CORTE = 0.5
PLANO_MIN_S = 8


def miniatura(img):
    m = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    m -= m.mean()
    return m / (np.linalg.norm(m) + 1e-6)


def main():
    os.makedirs(SAIDA, exist_ok=True)
    pose = carregar_yolo("yolo26n-pose.pt")
    amostras = []  # (video, segundo, miniatura, n_pessoas)
    for v in sorted(p for p in glob.glob("video/*") if p.lower().endswith((".mp4", ".webm", ".mov"))):
        cap = cv2.VideoCapture(v)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for s in range(int(n / fps)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(s * fps))
            ok, f = cap.read()
            if not ok:
                break
            r = pose.predict(f, imgsz=640, conf=0.4, classes=[0], verbose=False)[0]
            amostras.append((os.path.basename(v), s, miniatura(f), len(r.boxes)))
        print(f"  {os.path.basename(v)[:50]}: {s + 1} amostras", flush=True)
    mats = np.stack([a[2] for a in amostras])

    # 1) origem dos frames anotados
    linhas, por_video = [], {}
    for padrao in ANOTADOS:
        for p in sorted(glob.glob(padrao)):
            sim = mats @ miniatura(cv2.imread(p))
            j = int(sim.argmax())
            video = amostras[j][0] if sim[j] >= LIMIAR_MESMO else "(outro video)"
            por_video[video] = por_video.get(video, 0) + 1
            linhas.append([os.path.basename(p), "teste" if "real_world_test" in p else "treino", video,
                           amostras[j][1], f"{sim[j]:.3f}"])
    with open(os.path.join(SAIDA, "origem_frames_anotados.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "uso", "video", "segundo", "correlacao"])
        w.writerows(linhas)
    print("frames anotados por video de origem:", por_video)

    # 2) planos continuos com pessoas
    planos, ini = [], 0
    for k in range(1, len(amostras) + 1):
        fim_plano = (k == len(amostras) or amostras[k][0] != amostras[ini][0]
                     or float(amostras[k][2] @ amostras[k - 1][2]) < LIMIAR_CORTE)
        if fim_plano:
            dur = amostras[k - 1][1] - amostras[ini][1] + 1
            pessoas = int(np.median([a[3] for a in amostras[ini:k]]))
            if dur >= PLANO_MIN_S and pessoas >= 1:
                planos.append([amostras[ini][0], amostras[ini][1], amostras[k - 1][1] + 1, dur, pessoas])
            ini = k
    with open(os.path.join(SAIDA, "planos.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "inicio_s", "fim_s", "duracao_s", "pessoas_mediana"])
        w.writerows(sorted(planos, key=lambda x: -x[3]))
    print(f"{len(planos)} planos continuos >= {PLANO_MIN_S} s com pessoas; resultados em {SAIDA}/")


if __name__ == "__main__":
    main()
