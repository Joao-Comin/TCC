"""
Mesmo protocolo dos numeros do texto do TCC (model.val do Ultralytics no
teste real: conf 0.001, IoU 0.6 no NMS, lote 4, CPU), acrescentando o que
o model.val nao da: intervalo de confianca de 95% por bootstrap e
comparacao pareada entre versoes e resolucoes.

Como funciona: durante o model.val(), o Ultralytics guarda, para cada
imagem, quais predicoes acertaram (por limiar de IoU), as confiancas e as
classes. Este script so copia essas estatisticas por imagem (envolvendo
DetMetrics.update_stats) e depois reamostra as 25 imagens com reposicao,
recalculando o mAP com a propria funcao ap_per_class do Ultralytics. Os
valores pontuais saem identicos aos do model.val().

Saidas em results/avaliacao_teste_real/val_ultralytics/:
  metricas.csv           P, R, F1, mAP50, mAP50-95 por classe (iguais ao model.val)
  ic_bootstrap.csv       IC 95% do mAP50 (geral e por classe)
  comparacoes.csv        diferencas pareadas (B - A) com IC 95%
  RESUMO.md              tabelas prontas para o texto

Uso (a partir da pasta TCC):
    venv_win\\Scripts\\python scripts/avaliar_val_bootstrap.py
"""

import argparse
import csv
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("YOLO_OFFLINE", "true")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.metrics import DetMetrics  # noqa: E402

from epi import CLASSES  # noqa: E402
from epi.metricas import amostras_bootstrap, metricas  # noqa: E402

TESTE = "novos_frames_dataset/real_world_test"
SAIDA = "results/avaliacao_teste_real/val_ultralytics"
MODELOS = [("v2", "models/best_antigo.pt"), ("v4", "models/best_24-09.pt"), ("v5", "models/best.pt"),
           ("v6", "models/best_v6.pt")]
TAMANHOS = (640, 960)
# comparacoes pareadas (A, B): B - A nas mesmas reamostragens
COMPARACOES = [
    (("v2", 640), ("v4", 640)), (("v2", 640), ("v5", 640)), (("v4", 640), ("v5", 640)), (("v5", 640), ("v6", 960)),
    (("v2", 640), ("v6", 960)), (("v2", 960), ("v5", 960)), (("v5", 960), ("v6", 960)), (("v4", 960), ("v5", 960)),
    (("v2", 640), ("v2", 960)), (("v4", 640), ("v4", 960)), (("v5", 640), ("v5", 960)), (("v6", 640), ("v6", 960)),
]

_capturadas = []
_original = DetMetrics.update_stats


def _update_stats(self, stat):
    _capturadas.append((stat["tp"].copy(), np.asarray(stat["conf"]).copy(), np.asarray(stat["pred_cls"]).copy(),
                        np.asarray(stat["target_cls"]).copy()))
    return _original(self, stat)


DetMetrics.update_stats = _update_stats


def br(x, casas=3):
    return f"{float(x):.{casas}f}".replace(".", ",")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(SAIDA, exist_ok=True)
    yaml_teste = os.path.join(SAIDA, "teste_real.yaml")
    with open(yaml_teste, "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(TESTE).replace(os.sep, '/')}\ntrain: images\nval: images\nnames: {CLASSES}\n")

    stats, valores, amostras = {}, {}, {}
    linhas_m, linhas_ic = [], []
    for nome, pt in MODELOS:
        for sz in TAMANHOS:
            _capturadas.clear()
            r = YOLO(pt).val(data=yaml_teste, imgsz=sz, conf=0.001, iou=0.6, device="cpu", batch=4, workers=0,
                             plots=False, verbose=False, project=os.path.join(SAIDA, "runs"), name=f"{nome}_{sz}",
                             exist_ok=True)
            st = list(_capturadas)
            m = metricas(st, CLASSES)
            # conferencia: o calculo sobre as estatisticas capturadas tem que bater com o model.val
            assert abs(m["geral"]["mAP50"] - r.box.map50) < 1e-6, (m["geral"]["mAP50"], r.box.map50)
            chave = (nome, sz)
            stats[chave], valores[chave] = st, m
            amostras[chave] = amostras_bootstrap(st, CLASSES, args.bootstrap, 0, "mAP50")
            for classe, v in m.items():
                linhas_m.append([nome, sz, classe, v["n"], f"{v['P']:.4f}", f"{v['R']:.4f}", f"{v['F1']:.4f}",
                                 f"{v['mAP50']:.4f}", f"{v['mAP50-95']:.4f}"])
            for classe in ["geral"] + CLASSES:
                lo, hi = np.nanpercentile(amostras[chave][classe], [2.5, 97.5])
                linhas_ic.append([nome, sz, classe, f"{m.get(classe, {}).get('mAP50', np.nan):.4f}", f"{lo:.4f}", f"{hi:.4f}"])
            print(f"  {nome} {sz}: mAP50 {m['geral']['mAP50']:.4f} (= model.val {r.box.map50:.4f}) "
                  f"IC95 {np.nanpercentile(amostras[chave]['geral'], 2.5):.3f}-{np.nanpercentile(amostras[chave]['geral'], 97.5):.3f}",
                  flush=True)

    linhas_c = []
    for a, b in COMPARACOES:
        for classe in ["geral"] + CLASSES:
            d = amostras[b][classe] - amostras[a][classe]
            d = d[~np.isnan(d)]
            if not len(d):
                continue
            lo, hi = np.percentile(d, [2.5, 97.5])
            pa = valores[a].get(classe, {}).get("mAP50", np.nan)
            pb = valores[b].get(classe, {}).get("mAP50", np.nan)
            linhas_c.append([f"{a[0]} ({a[1]})", f"{b[0]} ({b[1]})", classe, f"{pa:.4f}", f"{pb:.4f}", f"{pb - pa:.4f}",
                             f"{lo:.4f}", f"{hi:.4f}", f"{np.mean(d > 0):.3f}", "sim" if lo > 0 or hi < 0 else "nao"])

    def gravar(nome, cab, linhas):
        with open(os.path.join(SAIDA, nome), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cab)
            w.writerows(linhas)

    gravar("metricas.csv", ["modelo", "imgsz", "classe", "instancias", "P", "R", "F1", "mAP50", "mAP50-95"], linhas_m)
    gravar("ic_bootstrap.csv", ["modelo", "imgsz", "classe", "mAP50", "ic95_inf", "ic95_sup"], linhas_ic)
    gravar("comparacoes.csv", ["A", "B", "classe", "mAP50_A", "mAP50_B", "dif", "dif_ic95_inf", "dif_ic95_sup",
                               "frac_amostras_B_melhor", "significativa"], linhas_c)

    out = ["# Teste real (25 frames, 102 caixas) - protocolo model.val do Ultralytics", "",
           "Valores pontuais identicos ao model.val (conf 0,001; IoU 0,6; lote 4). IC 95% por bootstrap "
           f"({args.bootstrap} reamostragens das imagens).", "",
           "| Versao | imgsz | mAP50 geral (IC 95%) | Capacete | Colete | Luvas | Botas | Oculos |", "|---|---|---|---|---|---|---|---|"]
    ic = {(l[0], l[1], l[2]): l for l in linhas_ic}
    for nome, _ in MODELOS:
        for sz in TAMANHOS:
            g = ic[(nome, sz, "geral")]
            cls = [br(ic[(nome, sz, c)][3]) if (nome, sz, c) in ic else "-" for c in CLASSES]
            out.append(f"| {nome} | {sz} | {br(g[3])} ({br(g[4])}-{br(g[5])}) | " + " | ".join(cls) + " |")
    out += ["", "## Comparacoes pareadas (B - A)", "",
            "| A | B | Classe | Dif. | IC 95% | % amostras B melhor | Significativa |", "|---|---|---|---|---|---|---|"]
    for a, b, c, _, _, d, lo, hi, frac, sig in linhas_c:
        if c in ("geral", "gloves", "boots"):
            out.append(f"| {a} | {b} | {c} | {br(d)} | {br(lo)} a {br(hi)} | {br(100 * float(frac), 0)}% | {sig} |")
    with open(os.path.join(SAIDA, "RESUMO.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"resultados em {SAIDA}/")


if __name__ == "__main__":
    main()
