"""
Simula uma pessoa assistindo um video no app novo (app/servidor.py precisa
estar rodando) e mede se a analise acompanha o video em tempo real.

O "espectador" abre o video, avanca o relogio na velocidade normal (como o
<video> do navegador) e manda a posicao a cada 250 ms, igual ao app.js. Para
cada frame que seria exibido, confere se ja existe deteccao "fresca" para ele
(frame analisado no maximo meio segundo antes, a mesma regra do app.js).

Mede:
  cobertura   % dos frames exibidos que tinham deteccao fresca
  atraso      quantos frames a deteccao usada esta atras do frame exibido
  fps_analise frames analisados por segundo (servidor)
  passo       1 frame analisado a cada k

Saida: results/benchmark_velocidade/sessao_tempo_real.csv (acrescenta linhas).

Uso (com o servidor rodando):
    venv_win\\Scripts\\python scripts/testar_sessao_tempo_real.py --segundos 30 --inicio 120
"""

import argparse
import asyncio
import bisect
import csv
import json
import os
import time
import urllib.request

import websockets

URL = "http://127.0.0.1:8000"
SAIDA = "results/benchmark_velocidade/sessao_tempo_real.csv"


async def sessao(video, modelo, recortes, inicio, segundos):
    async with websockets.connect(URL.replace("http", "ws") + "/ws", max_size=None) as ws:
        await ws.send(json.dumps({"tipo": "abrir", "video": video, "modelo": modelo, "recortes": recortes,
                                  "t": inicio, "g": 1}))
        fps = None
        while fps is None:  # espera o modelo carregar
            m = json.loads(await ws.recv())
            if m["tipo"] == "pronto":
                fps = m["fps"]
            elif m["tipo"] == "erro":
                raise RuntimeError(m["msg"])
        analisados, status = [], []

        async def receber():
            async for txt in ws:
                m = json.loads(txt)
                if m["tipo"] == "det":
                    analisados.append(m["i"])
                elif m["tipo"] == "status":
                    status.append(m)

        tarefa = asyncio.create_task(receber())
        # como o navegador: comeca pausado, da play quando ha deteccao para o frame inicial
        while not analisados:
            await ws.send(json.dumps({"tipo": "tempo", "t": inicio, "pausado": True}))
            await asyncio.sleep(0.05)
        t0 = time.perf_counter()
        exibidos, cobertos, atrasos = 0, 0, []
        ultimo_envio = 0
        limite = round(0.5 * fps)
        while (agora := time.perf_counter() - t0) < segundos:
            t_video = inicio + agora
            if agora - ultimo_envio >= 0.25:
                await ws.send(json.dumps({"tipo": "tempo", "t": t_video, "pausado": False}))
                ultimo_envio = agora
            f = round(t_video * fps)
            ja = sorted(analisados)
            k = bisect.bisect_right(ja, f) - 1
            exibidos += 1
            if k >= 0 and f - ja[k] <= limite:
                cobertos += 1
                atrasos.append(f - ja[k])
            await asyncio.sleep(1 / fps)  # um frame de video
        tarefa.cancel()
        fps_an = [s["fps"] for s in status[1:]] or [s["fps"] for s in status]
        passos = [s["passo"] for s in status[1:]] or [1]
        ms = [s["ms"].get("total_ms", 0) for s in status[1:]] or [0]
        return {
            "video": video[:40], "modelo": modelo, "recortes": recortes, "fps_video": round(fps, 2),
            "segundos": segundos, "frames_exibidos": exibidos, "cobertura_pct": round(100 * cobertos / exibidos, 1),
            "atraso_medio_frames": round(sum(atrasos) / max(1, len(atrasos)), 2),
            "fps_analise": round(sum(fps_an) / len(fps_an), 1), "passo_medio": round(sum(passos) / len(passos), 2),
            "ms_por_frame": round(sum(ms) / len(ms), 1),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segundos", type=float, default=30)
    ap.add_argument("--inicio", type=float, default=120)
    ap.add_argument("--modelo", default="v5")
    ap.add_argument("--recortes", action="store_true")
    args = ap.parse_args()
    est = json.load(urllib.request.urlopen(URL + "/api/estado"))
    os.makedirs(os.path.dirname(SAIDA), exist_ok=True)
    novo = not os.path.exists(SAIDA)
    with open(SAIDA, "a", newline="", encoding="utf-8") as f:
        w = None
        for v in est["videos"]:
            r = asyncio.run(sessao(v["nome"], args.modelo, args.recortes, args.inicio, args.segundos))
            print(r)
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(r))
                if novo:
                    w.writeheader()
            w.writerow(r)


if __name__ == "__main__":
    main()
