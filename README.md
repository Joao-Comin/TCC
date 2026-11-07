# Detecção de EPI com YOLO — TCC (Unesc)

**Aplicação de Inteligência Artificial na Detecção de Equipamentos de Proteção Individual para Prevenção de Acidentes em Ambientes Metalúrgicos** — João Victor Comin, orientador Sergio Coral.

O modelo detecta 5 classes de EPI (capacete, colete, luvas, botas e óculos). O app de demonstração roda o modelo sobre vídeo ou câmera, identifica cada pessoa com um ID que se mantém enquanto ela anda (rastreamento) e mostra quais EPIs cada uma está usando.

- Histórico das versões do modelo (v1 a v6): [CHANGELOG.md](CHANGELOG.md)

## Rodar o app

Dois cliques em **`iniciar_app.bat`** — ou, no terminal, a partir desta pasta:

```
venv_win\Scripts\python scripts\exportar_openvino.py   (só na primeira vez, ~2 min)
venv_win\Scripts\python app\servidor.py
```

e abrir <http://localhost:8000>. Os vídeos ficam na pasta `video/` (também dá para enviar pela interface). Na aba **Câmera ao vivo** o app usa a webcam.

| Na tela | O que faz |
|---|---|
| Boneco | clique numa parte do corpo para ligar/desligar a cobrança daquele EPI |
| Caixas das pessoas | verde = com todos os EPIs marcados; vermelho = falta algum; laranja = verificando |
| Clique numa pessoa | o boneco mostra o estado dela (✓ usando, ✕ falta, ? parte do corpo não visível) |
| Confiança mínima | corte de confiança das detecções (padrão 0,40) |
| Memória do EPI | um EPI conta como presente se foi visto nos últimos N segundos (padrão 2 s) |
| Ocorrências | "Pessoa N sem X" quando a falta dura 1 s; clique para voltar ao trecho; exporta CSV |

## Estrutura

```
app/            servidor.py (FastAPI) + static/ (interface); streamlit_app.py = app antigo
epi/            núcleo usado pelo app e pelas avaliações: detector.py, conformidade.py, metricas.py
scripts/        preparo de dados, avaliações, benchmarks, exportação, gráficos
notebooks/      treinos no Google Colab (v1 a v7) e pré-anotação com Grounding DINO
models/         pesos .pt (não versionados) e models/openvino/ (exportados, não versionados)
results/        métricas e figuras de cada versão e de cada avaliação
```

## Scripts de avaliação (números do TCC)

| Script | Mede | Saída |
|---|---|---|
| `scripts/avaliar_teste_real.py` | mAP50, P, R, F1 no teste real, com IC 95% por bootstrap e comparação pareada | `results/avaliacao_teste_real/*/RESUMO.md` |
| `scripts/analisar_erros.py` | o que o modelo fez com cada caixa anotada (acerto, perdido...) | `results/analise_erros/` |
| `scripts/benchmark_velocidade.py` | ms por frame (PyTorch x OpenVINO, CPU x GPU), app antigo x novo | `results/benchmark_velocidade/` |
| `scripts/comparar_detectores_pessoa.py` | detectores de pessoa (comum x pose): cobertura das caixas de EPI | `results/detectores_pessoa/` |
| `scripts/testar_sessao_tempo_real.py` | com o app aberto: % dos frames exibidos com detecção sincronizada | `results/benchmark_velocidade/sessao_tempo_real.csv` |
| `scripts/comparar_rastreadores.py` | estabilidade dos IDs: ByteTrack x BoT-SORT x TrackTrack (com/sem GMC) x OC-SORT | `results/rastreadores/` |
| `scripts/mapear_videos.py` | de qual vídeo saiu cada frame anotado; planos sem corte de cena | `results/mapa_videos/` |
| `scripts/gerar_graficos.py` | figuras a partir dos CSVs acima | `results/figuras/` |

## Ambiente

Python 3.14 no `venv_win`; dependências em [requirements.txt](requirements.txt). Treino no Google Colab (GPU T4): `notebooks/train_colab_v7.ipynb`.
