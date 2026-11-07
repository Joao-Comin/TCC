"""
Metricas de deteccao no mesmo padrao do `model.val()` do Ultralytics:
- casamento predicao x rotulo por IoU, guloso, nos 10 limiares 0.50:0.95;
- AP por classe pela curva PR interpolada em 101 pontos (padrao COCO),
  calculada pela propria funcao ap_per_class do Ultralytics;
- P, R e F1 no limiar de confianca que maximiza o F1 medio das classes.

Alem disso:
- contagem TP/FP/FN num limiar de confianca fixo (o que o app mostra);
- intervalo de confianca de 95% por bootstrap, reamostrando as imagens,
  porque o teste real tem so 25 frames e uma ou duas deteccoes mudam muito
  o numero de uma classe;
- bootstrap pareado para comparar duas configuracoes nas MESMAS amostras.

Cada imagem vira uma tupla (acertos (N, 10), conf (N,), classe_pred (N,),
classe_real (M,)) - ver estatisticas_imagem().
"""

import numpy as np
from ultralytics.utils.metrics import ap_per_class

IOUV = np.linspace(0.5, 0.95, 10)


def iou_matriz(a, b):
    """IoU entre todas as caixas de a (N, 4) e b (M, 4), em xyxy -> (N, M)."""
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def casar(pred_caixas, pred_cls, gt_caixas, gt_cls):
    """Matriz (N, 10) de acertos por limiar de IoU (mesmo algoritmo guloso do Ultralytics)."""
    correto = np.zeros((len(pred_caixas), len(IOUV)), bool)
    if len(pred_caixas) == 0 or len(gt_caixas) == 0:
        return correto
    iou = iou_matriz(gt_caixas, pred_caixas) * (gt_cls[:, None] == pred_cls[None, :])
    for i, limiar in enumerate(IOUV):
        m = np.array(np.nonzero(iou >= limiar)).T
        if m.shape[0]:
            if m.shape[0] > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correto[m[:, 1].astype(int), i] = True
    return correto


def ler_rotulos(caminho_txt, w, h):
    """Rotulo YOLO (classe xc yc w h normalizados) -> (caixas xyxy em pixels, classes)."""
    caixas, classes = [], []
    with open(caminho_txt) as f:
        for linha in f:
            p = linha.split()
            if len(p) >= 5:
                c, xc, yc, bw, bh = int(p[0]), *map(float, p[1:5])
                caixas.append(((xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h))
                classes.append(c)
    return np.array(caixas, np.float32).reshape(-1, 4), np.array(classes, int)


def estatisticas_imagem(pred, gt):
    """pred = (caixas, scores, classes); gt = (caixas, classes)."""
    pc, ps, pk = pred
    return casar(pc, pk, *gt), ps, pk, gt[1]


def metricas(stats, classes):
    """Metricas agregadas de uma lista de estatisticas por imagem.

    Devolve dict com 'geral' e uma entrada por nome de classe presente no
    rotulo: P, R, F1, mAP50, mAP50-95 e n (instancias).
    """
    tp = np.concatenate([s[0] for s in stats]) if stats else np.zeros((0, 10), bool)
    conf = np.concatenate([s[1] for s in stats]) if stats else np.zeros(0)
    pcls = np.concatenate([s[2] for s in stats]) if stats else np.zeros(0)
    tcls = np.concatenate([s[3] for s in stats]) if stats else np.zeros(0)
    if len(tcls) == 0:
        return {}
    _, _, p, r, f1, ap, idx, *_ = ap_per_class(tp, conf, pcls, tcls)
    n = np.bincount(tcls.astype(int), minlength=len(classes))
    out = {"geral": {"P": p.mean(), "R": r.mean(), "F1": f1.mean(), "mAP50": ap[:, 0].mean(),
                     "mAP50-95": ap.mean(), "n": int(len(tcls))}}
    for i, c in enumerate(idx):
        out[classes[c]] = {"P": p[i], "R": r[i], "F1": f1[i], "mAP50": ap[i, 0],
                           "mAP50-95": ap[i].mean(), "n": int(n[c])}
    return out


def contagem_limiar(preds, gts, limiar, nc):
    """TP/FP/FN por classe considerando so as predicoes com conf >= limiar (IoU >= 0.5)."""
    tp, fp, ngt = np.zeros(nc, int), np.zeros(nc, int), np.zeros(nc, int)
    for (pc, ps, pk), (gc, gk) in zip(preds, gts):
        ok = ps >= limiar
        acerto = casar(pc[ok], pk[ok], gc, gk)[:, 0]
        for c in range(nc):
            tp[c] += int(acerto[pk[ok] == c].sum())
            fp[c] += int((~acerto[pk[ok] == c]).sum())
            ngt[c] += int((gk == c).sum())
    return tp, fp, ngt - tp, ngt


def amostras_bootstrap(stats, classes, n, semente, metrica):
    """Valor da metrica em n reamostragens (com reposicao) das imagens -> {grupo: array(n)}.

    A semente fixa faz duas configuracoes avaliadas nas mesmas imagens
    receberem as MESMAS reamostragens (base do bootstrap pareado).
    """
    rng = np.random.default_rng(semente)
    grupos = ["geral"] + list(classes)
    vals = {g: np.full(n, np.nan) for g in grupos}
    for k, linha in enumerate(rng.integers(0, len(stats), (n, len(stats)))):
        m = metricas([stats[i] for i in linha], classes)
        for g in grupos:
            if g in m:
                vals[g][k] = m[g][metrica]
    return vals


def bootstrap(stats, classes, n=1000, semente=0, metrica="mAP50"):
    """IC 95% (percentis 2,5 e 97,5) da metrica, por grupo ('geral' e cada classe)."""
    return {g: np.nanpercentile(v, [2.5, 97.5]) for g, v in amostras_bootstrap(stats, classes, n, semente, metrica).items()}


def bootstrap_pareado(stats_a, stats_b, classes, n=1000, semente=0, metrica="mAP50"):
    """Diferenca B - A nas mesmas reamostragens, por grupo: (IC 95% da diferenca, fracao com B > A)."""
    va = amostras_bootstrap(stats_a, classes, n, semente, metrica)
    vb = amostras_bootstrap(stats_b, classes, n, semente, metrica)
    out = {}
    for g in va:
        d = vb[g] - va[g]
        out[g] = (np.nanpercentile(d, [2.5, 97.5]), float(np.nanmean(d[~np.isnan(d)] > 0)) if (~np.isnan(d)).any() else np.nan)
    return out
