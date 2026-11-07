"""
Exporta os modelos usados pelo app para OpenVINO (runtime da Intel), que
roda na GPU integrada Iris Xe do notebook. No benchmark
(scripts/benchmark_velocidade.py) o v6 em 960 passou de ~13 FPS (PyTorch,
CPU) para ~46 FPS (OpenVINO, GPU).

A forma de entrada fica fixa (estatica), o que mais acelera na GPU:
  <modelo>_frame_544x960    frame inteiro de video 16:9 (imgsz 960)
  <modelo>_recorte_640x384  recorte de uma pessoa (mais alto que largo)
  pessoa_384x640            pessoas + pontos do corpo (YOLO26n-pose, COCO), imgsz 640
Frames de outra proporcao tambem funcionam (o Ultralytics completa com
faixas cinzas), so aproveitam menos a entrada.

Saida: models/openvino/<nome>_openvino_model/ (o sufixo e exigido pelo
Ultralytics para reconhecer o formato). O app usa essas pastas quando existem e
cai para o .pt (PyTorch na CPU) quando nao.

Uso (a partir da pasta TCC):
    venv_win\\Scripts\\python scripts/exportar_openvino.py
    venv_win\\Scripts\\python scripts/exportar_openvino.py --modelos v6=models/best_v6.pt --pessoa yolo11n-pose.pt
"""

import argparse
import os
import shutil
import tempfile
import warnings

warnings.filterwarnings("ignore")

from ultralytics import YOLO  # noqa: E402

SAIDA = "models/openvino"
FORMAS = {"frame": (544, 960), "recorte": (640, 384)}
FORMA_PESSOA = (384, 640)


def exportar(pt, forma, destino):
    """Exporta numa copia temporaria (o Ultralytics grava ao lado do .pt) e move para destino."""
    with tempfile.TemporaryDirectory() as tmp:
        copia = os.path.join(tmp, "m.pt")
        shutil.copy(pt, copia)
        pasta = YOLO(copia).export(format="openvino", imgsz=list(forma), dynamic=False, half=False, verbose=False)
        if os.path.exists(destino):
            shutil.rmtree(destino)
        shutil.move(str(pasta), destino)
    print(f"  {pt} -> {destino}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", nargs="+", default=["v6=models/best_v6.pt", "v5=models/best.pt"])
    ap.add_argument("--pessoa", default="yolo26n-pose.pt", help="detector de pessoas com pose (COCO)")
    args = ap.parse_args()
    os.makedirs(SAIDA, exist_ok=True)
    for spec in args.modelos:
        nome, pt = spec.split("=")
        for tipo, (h, w) in FORMAS.items():
            exportar(pt, (h, w), os.path.join(SAIDA, f"{nome}_{tipo}_{h}x{w}_openvino_model"))
    h, w = FORMA_PESSOA
    exportar(args.pessoa, FORMA_PESSOA, os.path.join(SAIDA, f"pessoa_{h}x{w}_openvino_model"))


if __name__ == "__main__":
    main()
