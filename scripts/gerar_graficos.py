"""
Gera as figuras do TCC a partir dos CSVs de results/ (rode depois dos
scripts de avaliacao e benchmark). Fundo branco, para impressao.

Figuras em results/figuras/:
  velocidade_app.png        frames do video que recebem analise: app antigo x novo (teste de sessao)
  latencia_modelos.png      ms por frame: PyTorch CPU x OpenVINO CPU x OpenVINO GPU
  estrategias_teste_real.png mAP50 (IC 95%) das estrategias de inferencia, v5 e v6
  evolucao_versoes.png      mAP50 (IC 95%) por classe, v2 -> v6, imgsz 960
  erros_por_classe.png      o que aconteceu com cada caixa anotada (v6, 960)
  rastreadores.png          IDs por pessoa e trilhas-relampago de cada rastreador

Cores: paleta categorica validada (azul, laranja, agua) e, so onde a cor
significa resultado bom/ruim, a paleta de status (verde, ambar, vermelho).
Todo valor plotado esta tambem nos CSVs (a "tabela" de cada figura).

Uso (a partir da pasta TCC):
    venv_win\\Scripts\\python scripts/gerar_graficos.py
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

R = "results"
SAIDA = os.path.join(R, "figuras")
SERIE = ["#2a78d6", "#eb6834", "#1baf7a"]
STATUS = {"bom": "#0ca30c", "atencao": "#fab219", "critico": "#d03b3b"}
TINTA, TINTA_2, MUDO = "#0b0b0b", "#52514e", "#898781"
GRADE, EIXO = "#e1e0d9", "#c3c2b7"
NOMES = {"geral": "Geral", "helmet": "Capacete", "vest": "Colete", "gloves": "Luvas", "boots": "Botas", "goggles": "Óculos"}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "text.color": TINTA,
    "axes.edgecolor": EIXO, "axes.labelcolor": TINTA_2, "axes.linewidth": 1, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": TINTA,
    "xtick.color": MUDO, "ytick.color": MUDO, "xtick.labelcolor": TINTA_2, "ytick.labelcolor": TINTA_2,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200, "legend.frameon": False,
})


def ler(caminho):
    with open(caminho, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def br(x, casas=2):
    return f"{x:.{casas}f}".replace(".", ",")


def limpar_eixos(ax, grade="x"):
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.spines["left" if grade == "x" else "bottom"].set_visible(True)
    ax.grid(axis=grade, color=GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def virgula(ax, eixo="x", casas=2):
    """Decimais com virgula nos eixos (padrao pt-BR do texto)."""
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: br(v, casas))
    (ax.xaxis if eixo == "x" else ax.yaxis).set_major_formatter(fmt)


def salvar(fig, nome, subtitulo=None):
    if subtitulo:  # nota abaixo de tudo (o bbox "tight" inclui o texto)
        fig.text(0.01, -0.03, subtitulo, color=MUDO, fontsize=8.5, ha="left", va="top")
    fig.savefig(os.path.join(SAIDA, nome), bbox_inches="tight")
    plt.close(fig)
    print("  ", os.path.join(SAIDA, nome))


def video_curto(nome):
    return "Vídeo 480p, 30 fps (Stemac)" if "Stemac" in nome else "Vídeo 1080p, 24 fps (alumínio)"


def fig_velocidade():
    antigo = {video_curto(r["video"]): float(r["pct_frames_exibidos"]) for r in ler(f"{R}/benchmark_velocidade/loop_app_antigo.csv")}
    # app novo: teste de sessao (o app de verdade, com a folga do passo adaptativo)
    novo = {video_curto(r["video"]): 100 / float(r["passo_medio"]) for r in ler(f"{R}/benchmark_velocidade/sessao_tempo_real.csv")}
    videos = sorted(antigo)
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    alt = 0.32
    for k, (serie, dados, rot) in enumerate([("antigo", antigo, "App antigo (Streamlit): frames mostrados"),
                                             ("novo", novo, "App novo: frames analisados (o vídeo mostra 100%)")]):
        ys = [i + (k - 0.5) * (alt + 0.04) for i in range(len(videos))]
        vals = [dados.get(v, 0) for v in videos]
        ax.barh(ys, vals, height=alt, color=SERIE[k], label=rot)
        for y, v in zip(ys, vals):
            ax.text(v + 1.2, y, f"{br(v, 0)}%", va="center", color=TINTA, fontsize=9.5)
    ax.set_yticks(range(len(videos)), videos)
    ax.set_xlim(0, 110)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.invert_yaxis()
    limpar_eixos(ax)
    ax.set_title("Frames do vídeo que recebem a análise do modelo")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2, fontsize=8.5)
    salvar(fig, "velocidade_app.png")


def fig_latencia():
    linhas = ler(f"{R}/benchmark_velocidade/latencia_modelos.csv")
    modelos = list(dict.fromkeys(r["modelo"] for r in linhas))
    fig, axs = plt.subplots(1, len(modelos), figsize=(7.6, 2.4), sharex=True)
    vmax = max(float(r["ms_mediana"]) for r in linhas)
    for ax, m in zip(axs, modelos):
        rs = [r for r in linhas if r["modelo"] == m]
        rotulos = [f"{r['formato']} · {r['dispositivo']}" for r in rs]
        vals = [float(r["ms_mediana"]) for r in rs]
        ax.barh(range(len(rs)), vals, height=0.5, color=SERIE[0])
        for i, (v, r) in enumerate(zip(vals, rs)):
            ax.text(v + vmax * 0.02, i, f"{br(v, 1)} ms ({br(float(r['fps']), 0)} FPS)", va="center", fontsize=9, color=TINTA)
        ax.set_yticks(range(len(rs)), rotulos)
        ax.invert_yaxis()
        ax.set_xlim(0, vmax * 1.55)
        limpar_eixos(ax)
        ax.set_title(m, fontsize=10)
        ax.set_xlabel("ms por frame (mediana)")
    fig.tight_layout(w_pad=2.5)
    salvar(fig, "latencia_modelos.png")


def fig_estrategias():
    ic = ler(f"{R}/avaliacao_teste_real/estrategias/ic_bootstrap.csv")
    geral = [r for r in ic if r["classe"] == "geral"]
    modelos = list(dict.fromkeys(r["modelo"] for r in geral))
    estr = list(dict.fromkeys(r["estrategia"] for r in geral))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for k, m in enumerate(sorted(modelos)):
        rs = {r["estrategia"]: r for r in geral if r["modelo"] == m}
        for i, e in enumerate(estr):
            r = rs[e]
            y = i + (k - 0.5) * 0.28
            v, lo, hi = float(r["mAP50"]), float(r["ic95_inf"]), float(r["ic95_sup"])
            ax.plot([lo, hi], [y, y], color=SERIE[k], linewidth=2, solid_capstyle="round")
            ax.plot(v, y, "o", color=SERIE[k], markersize=7, markeredgecolor="white", markeredgewidth=1.5,
                    label=f"Modelo {m}" if i == 0 else None)
            ax.text(hi + 0.008, y, br(v, 3), va="center", fontsize=8, color=TINTA_2)
    ax.set_yticks(range(len(estr)), [e.replace("_", " ").replace("+", " + ") for e in estr])
    for t in ax.get_yticklabels():
        if t.get_text() == "frame 960":
            t.set_fontweight("bold")
            t.set_color(TINTA)
    ax.invert_yaxis()
    limpar_eixos(ax)
    ax.set_xlabel("mAP50 no teste real (25 frames) — ponto = valor, barra = IC 95% (bootstrap)")
    ax.set_title("Estratégias de inferência sem retreino", pad=22)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2, fontsize=9)
    virgula(ax, "x", 2)
    salvar(fig, "estrategias_teste_real.png", "Linha de base em negrito: frame inteiro em 960 px. Todas as barras se sobrepõem: nenhuma diferença é significativa.")


def fig_versoes():
    ic = ler(f"{R}/avaliacao_teste_real/versoes/ic_bootstrap.csv")
    rs = [r for r in ic if r["estrategia"] == "frame_960"]
    versoes = list(dict.fromkeys(r["modelo"] for r in rs))
    classes = ["geral", "helmet", "vest", "gloves", "boots", "goggles"]
    fig, axs = plt.subplots(2, 3, figsize=(7.6, 4.4), sharey=True)
    for ax, c in zip(axs.ravel(), classes):
        pts = [next((r for r in rs if r["modelo"] == v and r["classe"] == c), None) for v in versoes]
        xs = range(len(versoes))
        vals = [float(p["mAP50"]) if p else float("nan") for p in pts]
        ax.plot(xs, vals, color=SERIE[0], linewidth=2, zorder=3)
        for x, p in zip(xs, pts):
            if p:
                ax.plot([x, x], [float(p["ic95_inf"]), float(p["ic95_sup"])], color=SERIE[0], alpha=0.35, linewidth=6,
                        solid_capstyle="round", zorder=2)
                ax.plot(x, float(p["mAP50"]), "o", color=SERIE[0], markersize=6.5, markeredgecolor="white",
                        markeredgewidth=1.5, zorder=4)
        ax.text(xs[-1] + 0.15, vals[-1], br(vals[-1], 2), va="center", fontsize=8.5, color=TINTA)
        ax.set_xticks(list(xs), versoes)
        ax.set_xlim(-0.3, len(versoes) - 0.3)
        ax.set_ylim(0, 1)
        virgula(ax, "y", 1)
        ax.set_title(NOMES[c], fontsize=10)
        limpar_eixos(ax, grade="y")
    axs[0, 0].set_ylabel("mAP50")
    axs[1, 0].set_ylabel("mAP50")
    fig.suptitle("Evolução por versão no teste real (imgsz 960)", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    salvar(fig, "evolucao_versoes.png", "Ponto = mAP50; faixa clara = IC 95% por bootstrap (25 imagens).")


def fig_erros():
    linhas = [r for r in ler(f"{R}/analise_erros/categorias.csv") if r["modelo"] == "v6"]
    grupos = [("Acerto", ["acerto"], STATUS["bom"]),
              ("Quase (conf. baixa, posição ou classe)", ["conf_baixa", "localizacao", "outra_classe"], STATUS["atencao"]),
              ("Perdido (nada no lugar)", ["perdido"], STATUS["critico"])]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    for i, r in enumerate(linhas):
        total = sum(int(r[c]) for _, cs, _ in grupos for c in cs)
        esq = 0
        for nome, cs, cor in grupos:
            n = sum(int(r[c]) for c in cs)
            frac = n / total if total else 0
            ax.barh(i, frac, left=esq, height=0.55, color=cor, edgecolor="white", linewidth=2,
                    label=nome if i == 0 else None)
            if frac >= 0.12:
                ax.text(esq + frac / 2, i, f"{n}", ha="center", va="center", fontsize=9,
                        color="white" if cor != STATUS["atencao"] else TINTA)
            esq += frac
        ax.text(1.02, i, f"{total} caixas", va="center", fontsize=8.5, color=TINTA_2)
    ax.set_yticks(range(len(linhas)), [NOMES[r["classe"]] for r in linhas])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
    limpar_eixos(ax)
    ax.set_title("O que o modelo v6 fez com cada caixa anotada (conf ≥ 0,25)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=3, fontsize=8.5)
    salvar(fig, "erros_por_classe.png")


def fig_rastreadores():
    import inspect
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from epi.detector import Detector  # o rastreador do app e o padrao do Detector
    usado = os.path.basename(inspect.signature(Detector).parameters["rastreador"].default).replace(".yaml", "")
    linhas = sorted(ler(f"{R}/rastreadores/resumo.csv"), key=lambda r: float(r["ids_por_pessoa"]))
    nomes = {"tracktrack": "TrackTrack", "tracktrack_sem_gmc": "TrackTrack sem GMC", "bytetrack": "ByteTrack",
             "botsort": "BoT-SORT", "ocsort": "OC-SORT"}
    fig, axs = plt.subplots(1, 3, figsize=(8.6, 2.5), sharey=True)
    for ax, campo, titulo, fmt_ in [(axs[0], "ids_por_pessoa", "IDs por pessoa real (ideal = 1)", lambda v: br(v, 2)),
                                    (axs[1], "trilhas_curtas", "Trilhas-relâmpago (< 0,5 s)", lambda v: f"{v:.0f}"),
                                    (axs[2], "ms_rastreador_mediana", "Custo extra por frame (ms)", lambda v: br(v, 1))]:
        vals = [float(r[campo]) for r in linhas]
        cores = [SERIE[0] if r["rastreador"] == usado else "#9aa4b5" for r in linhas]
        ax.barh(range(len(linhas)), vals, height=0.5, color=cores)
        for i, v in enumerate(vals):
            ax.text(v + max(vals) * 0.02, i, fmt_(v), va="center", fontsize=9, color=TINTA)
        ax.set_xlim(0, max(vals) * 1.25)
        ax.set_title(titulo, fontsize=10)
        limpar_eixos(ax)
        if campo == "ids_por_pessoa":
            ax.axvline(1, color=EIXO, linewidth=1)
            virgula(ax, "x", 1)
    axs[0].set_yticks(range(len(linhas)), [nomes.get(r["rastreador"], r["rastreador"]) for r in linhas])
    axs[0].invert_yaxis()
    fig.suptitle("Rastreadores em 8 planos contínuos (13 pessoas reais, 145 s)", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    salvar(fig, "rastreadores.png", "Mesmas detecções (YOLO26n-pose) para todos; destaque = o usado no app.")


def main():
    os.makedirs(SAIDA, exist_ok=True)
    for fn, precisa in [(fig_velocidade, "benchmark_velocidade/sessao_tempo_real.csv"),
                        (fig_latencia, "benchmark_velocidade/latencia_modelos.csv"),
                        (fig_estrategias, "avaliacao_teste_real/estrategias/ic_bootstrap.csv"),
                        (fig_versoes, "avaliacao_teste_real/versoes/ic_bootstrap.csv"),
                        (fig_erros, "analise_erros/categorias.csv"),
                        (fig_rastreadores, "rastreadores/resumo.csv")]:
        if os.path.exists(os.path.join(R, precisa)):
            fn()
        else:
            print(f"   (pulada: falta {precisa})")


if __name__ == "__main__":
    main()
