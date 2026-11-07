"""
Visibilidade de cada parte do corpo, a partir dos pontos do modelo de pose
(YOLO pose, 17 pontos do padrao COCO):

  0 nariz, 1-2 olhos, 3-4 orelhas, 5-6 ombros, 7-8 cotovelos,
  9-10 punhos, 11-12 quadris, 13-14 joelhos, 15-16 tornozelos

Um EPI so pode ser cobrado se a parte do corpo dele aparece no frame. Sem
isso o app acusaria "sem oculos" em quem esta de costas e "sem bota" em
quem tem os pes fora do quadro:

  helmet   cabeca: nariz, olhos ou orelhas
  vest     tronco: pelo menos um ombro e um quadril
  gloves   pelo menos um punho
  boots    pelo menos um tornozelo
  goggles  rosto de frente: os dois olhos, ou o nariz e um olho

A decisao de conformidade (usando / ausente / nao visivel) fica no
navegador (app/static/app.js), porque depende da confianca minima e dos EPIs
escolhidos no boneco, que o usuario muda com o video rodando.
"""

import numpy as np

KP_CONF = 0.5


def partes_visiveis(kps, w, h, conf=KP_CONF):
    """kps (17, 3) = (x, y, conf) em pixels -> 5 bools na ordem de CLASSES."""
    ok = (kps[:, 2] >= conf) & (kps[:, 0] >= 0) & (kps[:, 0] < w) & (kps[:, 1] >= 0) & (kps[:, 1] < h)
    cabeca = ok[0:5].any()
    rosto = (ok[1] and ok[2]) or (ok[0] and (ok[1] or ok[2]))
    tronco = ok[5:7].any() and ok[11:13].any()
    return [bool(cabeca), bool(tronco), bool(ok[9:11].any()), bool(ok[15:17].any()), bool(rosto)]


def partes_visiveis_caixa(caixa, w, h, borda=0.01):
    """Sem pontos do corpo (detector comum): so olha se a caixa foi cortada na borda do frame."""
    x1, y1, x2, y2 = caixa
    topo_ok = y1 > borda * h
    base_ok = y2 < (1 - borda) * h
    return [bool(topo_ok), True, True, bool(base_ok), bool(topo_ok)]


def mascara(visiveis):
    """Lista de bools -> inteiro (bit i = classe i visivel), formato enviado ao navegador."""
    return int(sum(1 << i for i, v in enumerate(visiveis) if v))


def mascaras_pessoas(caixas, kps, w, h):
    if kps is not None and len(kps):
        return [mascara(partes_visiveis(k, w, h)) for k in kps]
    return [mascara(partes_visiveis_caixa(c, w, h)) for c in caixas]


def resumo_visibilidade(mascaras, nc=5):
    """Fracao de pessoas com cada parte visivel (usado nos relatorios)."""
    m = np.array(mascaras, int)
    if len(m) == 0:
        return np.zeros(nc)
    return np.array([((m >> i) & 1).mean() for i in range(nc)])
