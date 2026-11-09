"""
Avalia estrategias de inferencia nos 25 frames do teste real
(novos_frames_dataset/real_world_test), usando o mesmo pipeline do app
(epi/detector.py). Nenhuma estrategia exige retreino: muda so COMO o
modelo ja treinado e aplicado ao frame.

Estrategias (CONFIGS abaixo):
  frame_640 / frame_960 / frame_1280  - frame inteiro em cada tamanho
  frame_960_tta                      - test-time augmentation do Ultralytics
  frame_960+recortes                 - + modelo de EPI em cada pessoa recortada
  frame_960+fatias2x2                - + 4 fatias do frame (estilo SAHI)
  ...

Para cada modelo x estrategia grava:
  metricas.csv        P, R, F1, mAP50, mAP50-95 por classe (padrao model.val)
  ic_bootstrap.csv    IC 95% do mAP50 geral e por classe (reamostrando imagens)
  comparacao.csv      bootstrap pareado contra a linha de base de cada modelo
  contagem.csv        TP/FP/FN em conf 0.25 e 0.40 (o que o app mostra)
  tempos.csv          ms por imagem (CPU, PyTorch) de cada etapa
  RESUMO.md           tabelas prontas para o TCC

Uso (a partir da pasta TCC):
    venv_win\\Scripts\\python scripts/avaliar_teste_real.py
    venv_win\\Scripts\\python scripts/avaliar_teste_real.py --modelos v6=models/best_v6.pt --configs frame_960 frame_960+recortes
"""

import argparse
import csv
import glob
import os
import sys
import time
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("YOLO_OFFLINE", "true")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epi import CLASSES  # noqa: E402
from epi.detector import Detector  # noqa: E402
from epi.metricas import (amostras_bootstrap, contagem_limiar, estatisticas_imagem, ler_rotulos,  # noqa: E402
                          metricas)

TESTE = "novos_frames_dataset/real_world_test"
MODELO_PESSOA = "yolo11n.pt"
CONF_MAP = 0.001  # confianca minima para a curva PR (igual ao model.val)
IOU_NMS = 0.6

CONFIGS = {
    "frame_640": dict(imgsz=640),
    "frame_960": dict(imgsz=960),
    "frame_1280": dict(imgsz=1280),
    "frame_960_tta": dict(imgsz=960, tta=True),
    "frame_640+recortes": dict(imgsz=640, recortes=True),
    "frame_960+recortes": dict(imgsz=960, recortes=True),
    "frame_960+recortes960": dict(imgsz=960, recortes=True, imgsz_recorte=960),
    "frame_960+fatias2x2": dict(imgsz=960, fatias=(2, 2, 0.2), imgsz_recorte=960),
}
BASE = "frame_960"


def carregar_teste():
    imgs = sorted(glob.glob(os.path.join(TESTE, "images", "*")))
    frames, gts = [], []
    for p in imgs:
        im = cv2.imread(p)
        h, w = im.shape[:2]
        lbl = os.path.join(TESTE, "labels", os.path.splitext(os.path.basename(p))[0] + ".txt")
        frames.append(im)
        gts.append(ler_rotulos(lbl, w, h))
    return imgs, frames, gts


def rodar(det, frames):
    preds, tempos = [], []
    for f in frames[:2]:  # aquecimento
        det.processar(f)
    for f in frames:
        t = time.perf_counter()
        r = det.processar(f)
        tempos.append({**r["tempos"], "total_ms": (time.perf_counter() - t) * 1000})
        preds.append(r["epis"])
    return preds, tempos


def fmt(x):
    return f"{x:.3f}"


def br(x, casas=3):
    return f"{float(x):.{casas}f}".replace(".", ",")


def escrever_resumo(saida, linhas_ic, linhas_cmp, linhas_cont, linhas_t, linhas_mod):
    """RESUMO.md com as tabelas prontas para o TCC (decimais com virgula)."""
    ic = {(m, e, c): (v, lo, hi) for m, e, c, v, lo, hi in linhas_ic}
    tempos = {(m, e): t for m, e, *_, t in linhas_t}
    modelos = list(dict.fromkeys(l[0] for l in linhas_ic))
    estrategias = list(dict.fromkeys(l[1] for l in linhas_ic))
    out = ["# Teste real (25 frames, 102 caixas)", "",
           "mAP50 com intervalo de confianca de 95% (bootstrap, 1000 reamostragens das imagens). "
           "Tempo = mediana por imagem na CPU (PyTorch), so para comparar as estrategias entre si.", ""]
    for m in modelos:
        out += [f"## Modelo {m}", "", "| Estrategia | mAP50 geral (IC 95%) | Luvas | Botas | ms/imagem |", "|---|---|---|---|---|"]
        for e in estrategias:
            if (m, e, "geral") not in ic:
                continue
            g, lu, bo = ic[(m, e, "geral")], ic.get((m, e, "gloves")), ic.get((m, e, "boots"))
            out.append(f"| {e} | {br(g[0])} ({br(g[1])}-{br(g[2])}) | {br(lu[0]) if lu else '-'} | "
                       f"{br(bo[0]) if bo else '-'} | {br(tempos.get((m, e), 0), 0)} |")
        out.append("")
    out += ["## Diferenca para a linha de base (bootstrap pareado)", "",
            "Dif. = estrategia - linha de base, nas mesmas reamostragens. Se o IC 95% inclui 0, "
            "a diferenca nao e significativa com 25 imagens.", "",
            "| Modelo | Estrategia | Classe | Dif. media | IC 95% | % amostras melhor |", "|---|---|---|---|---|---|"]
    for m, base, e, c, med, lo, hi, frac in linhas_cmp:
        if c in ("geral", "gloves", "boots"):
            out.append(f"| {m} | {e} | {c} | {br(med)} | {br(lo)} a {br(hi)} | {br(100 * float(frac), 0)}% |")
    out += ["", "## Contagem no limiar do app (conf 0,40; IoU >= 0,5)", "",
            "| Modelo | Estrategia | Classe | GT | TP | FP | FN | Precisao | Recall | F1 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for m, e, limiar, c, g, tp, fp, fn, p, r, f1 in linhas_cont:
        if float(limiar) == 0.40 and e == BASE:
            out.append(f"| {m} | {e} | {c} | {g} | {tp} | {fp} | {fn} | {br(p)} | {br(r)} | {br(f1)} |")
    if linhas_mod:
        out += ["", "## Comparacao entre modelos (bootstrap pareado)", "",
                "Dif. = modelo - primeiro modelo da lista, na mesma estrategia e nas mesmas reamostragens.", "",
                "| Estrategia | Base | Modelo | Classe | Dif. media | IC 95% | % amostras melhor |",
                "|---|---|---|---|---|---|---|"]
        for e, base, m, c, med, lo, hi, frac in linhas_mod:
            if c in ("geral", "gloves", "boots"):
                out.append(f"| {e} | {base} | {m} | {c} | {br(med)} | {br(lo)} a {br(hi)} | {br(100 * float(frac), 0)}% |")
    with open(os.path.join(saida, "RESUMO.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", nargs="+", default=["v6=models/best_v6.pt", "v5=models/best.pt"])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--saida", default="results/avaliacao_teste_real")
    args = ap.parse_args()
    os.makedirs(args.saida, exist_ok=True)

    _, frames, gts = carregar_teste()
    print(f"teste real: {len(frames)} imagens, {sum(len(g[1]) for g in gts)} caixas")

    linhas_m, linhas_ic, linhas_cmp, linhas_cont, linhas_t = [], [], [], [], []
    todas = {}  # (modelo, estrategia) -> reamostragens, para comparar modelos no fim
    for spec in args.modelos:
        nome_m, caminho = spec.split("=")
        amostras_base = None
        for nome_c in args.configs:
            cfg = CONFIGS[nome_c]
            usa_pessoa = cfg.get("recortes", False)
            det = Detector(caminho, MODELO_PESSOA if usa_pessoa else None, conf=CONF_MAP, iou=IOU_NMS, **cfg)
            preds, tempos = rodar(det, frames)
            stats = [estatisticas_imagem(p, g) for p, g in zip(preds, gts)]
            m = metricas(stats, CLASSES)
            # mesma semente em todas as configs = mesmas reamostragens (bootstrap pareado)
            amostras = amostras_bootstrap(stats, CLASSES, args.bootstrap, 0, "mAP50")
            todas[(nome_m, nome_c)] = amostras
            if nome_c == BASE:
                amostras_base = amostras

            for classe, v in m.items():
                linhas_m.append([nome_m, nome_c, classe, v["n"], fmt(v["P"]), fmt(v["R"]), fmt(v["F1"]),
                                 fmt(v["mAP50"]), fmt(v["mAP50-95"])])
            for classe in ["geral"] + CLASSES:
                lo, hi = np.nanpercentile(amostras[classe], [2.5, 97.5])
                linhas_ic.append([nome_m, nome_c, classe, fmt(m.get(classe, {}).get("mAP50", np.nan)), fmt(lo), fmt(hi)])
            for limiar in (0.25, 0.40):
                tp, fp, fn, ngt = contagem_limiar(preds, gts, limiar, len(CLASSES))
                for i, classe in enumerate(CLASSES + ["TOTAL"]):
                    t_, f_, n_, g_ = (tp.sum(), fp.sum(), fn.sum(), ngt.sum()) if classe == "TOTAL" else (tp[i], fp[i], fn[i], ngt[i])
                    p_ = t_ / (t_ + f_) if t_ + f_ else 0
                    r_ = t_ / g_ if g_ else 0
                    f1 = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0
                    linhas_cont.append([nome_m, nome_c, limiar, classe, g_, t_, f_, n_, fmt(p_), fmt(r_), fmt(f1)])
            med = {k: float(np.median([t[k] for t in tempos])) for k in tempos[0]}
            linhas_t.append([nome_m, nome_c] + [f"{med[k]:.1f}" for k in ("pessoas_ms", "epis_ms", "refino_ms", "total_ms")])
            if amostras_base is not None and nome_c != BASE:
                for classe in ["geral"] + CLASSES:
                    d = amostras[classe] - amostras_base[classe]
                    d = d[~np.isnan(d)]
                    lo, hi = np.percentile(d, [2.5, 97.5])
                    linhas_cmp.append([nome_m, BASE, nome_c, classe, fmt(np.mean(d)), fmt(lo), fmt(hi),
                                       f"{np.mean(d > 0):.3f}"])
            g = m["geral"]
            print(f"{nome_m:4s} {nome_c:24s} mAP50={g['mAP50']:.3f} mAP50-95={g['mAP50-95']:.3f} "
                  f"P={g['P']:.3f} R={g['R']:.3f} | luvas={m.get('gloves', {}).get('mAP50', 0):.3f} "
                  f"botas={m.get('boots', {}).get('mAP50', 0):.3f} | {med['total_ms']:.0f} ms/img")

    def gravar(nome, cab, linhas):
        with open(os.path.join(args.saida, nome), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cab)
            w.writerows(linhas)

    gravar("metricas.csv", ["modelo", "estrategia", "classe", "instancias", "P", "R", "F1", "mAP50", "mAP50-95"], linhas_m)
    gravar("ic_bootstrap.csv", ["modelo", "estrategia", "classe", "mAP50", "ic95_inf", "ic95_sup"], linhas_ic)
    gravar("comparacao.csv", ["modelo", "base", "estrategia", "classe", "dif_media", "dif_ic95_inf",
                              "dif_ic95_sup", "frac_amostras_melhor"], linhas_cmp)
    gravar("contagem.csv", ["modelo", "estrategia", "conf", "classe", "GT", "TP", "FP", "FN", "precisao", "recall", "F1"], linhas_cont)
    gravar("tempos.csv", ["modelo", "estrategia", "pessoas_ms", "epis_ms", "refino_ms", "total_ms"], linhas_t)
    # cada modelo contra o primeiro da lista, na mesma estrategia (mesmas reamostragens)
    linhas_mod = []
    nomes_m = [m.split("=")[0] for m in args.modelos]
    for nome_c in args.configs:
        for nome_m in nomes_m[1:]:
            a, b = todas.get((nomes_m[0], nome_c)), todas.get((nome_m, nome_c))
            if a is None or b is None:
                continue
            for classe in ["geral"] + CLASSES:
                d = b[classe] - a[classe]
                d = d[~np.isnan(d)]
                if len(d):
                    lo, hi = np.percentile(d, [2.5, 97.5])
                    linhas_mod.append([nome_c, nomes_m[0], nome_m, classe, fmt(np.mean(d)), fmt(lo), fmt(hi),
                                       f"{np.mean(d > 0):.3f}"])
    if linhas_mod:
        gravar("comparacao_modelos.csv", ["estrategia", "base", "modelo", "classe", "dif_media", "dif_ic95_inf",
                                          "dif_ic95_sup", "frac_amostras_melhor"], linhas_mod)
    escrever_resumo(args.saida, linhas_ic, linhas_cmp, linhas_cont, linhas_t, linhas_mod)
    print(f"\nresultados em {args.saida}/ (tabelas prontas em RESUMO.md)")


if __name__ == "__main__":
    main()
