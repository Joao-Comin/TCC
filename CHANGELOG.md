# Histórico de versões do modelo

## v1 — 2026-08-19/20

- **Dataset**: `yolo.zip` (5 classes: helmet, vest, gloves, boots, goggles — sem `person`). 14.634 imagens, split 80/10/10 (11.940 treino / 1.346 val / 1.348 teste).
- **Modelo**: YOLO11n, pré-treinado em COCO.
- **Config**: 100 épocas planejadas, batch=16, imgsz=640, otimizador auto (MuSGD).
- **Treino**: interrompido na **época 78/100** por limite de cota de GPU grátis do Colab (mAP já em platô desde ~época 70, sem sinal de que completar as 22 épocas restantes mudaria muito o resultado).
- **Resultado**: mAP50=0.699, mAP50-95=0.485 (média das 5 classes).
  - Ótimo: `vest` (mAP50 0.969), `helmet` (0.947), `boots` (0.942)
  - Fraco: `goggles` (mAP50 0.421)
  - Muito fraco: `gloves` (mAP50 0.218, recall 5,7%)
- **Causa identificada**: forte desbalanceamento de classes no dataset (helmet=25.175 instâncias vs. goggles=904, gloves=2.452).
- **Limitação conhecida**: bug em [scripts/augment_dataset.py](scripts/augment_dataset.py) — os labels não são reajustados ao aplicar `HorizontalFlip`/`Rotate`, então o augmentation atual (se usado) gera anotações desalinhadas.
- **Arquivos**: [models/v1/](models/v1/), [results/v1/](results/v1/) (inclui `resultados_treino_yolo.xlsx` com todas as métricas e o glossário de termos).

## v2 — 2026-09-06

- **Dataset**: `yolo_v2.zip` (mesmas 5 classes da v1). Adicionados os datasets `Gloves.v1i` e `Goggles.v3i` ao `scripts/unify_datasets.py` para reforçar as duas classes mais fracas da v1; corrigido também um bug de case-sensitivity que descartava a classe "Person" de um dos datasets de origem antes do filtro. Split de teste final: 2.146 imagens / 5.980 instâncias.
- **Modelo**: YOLO11n, pré-treinado em COCO (mesma base da v1).
- **Config**: 100 épocas planejadas, batch=16, imgsz=640.
- **Treino**: sessão do Colab caiu por esgotamento de cota de GPU grátis durante a época ~62/100. Não foi retomada: o mAP50 já estava estabilizado em 0.851–0.852 nas 3 últimas épocas antes da queda, sinal de convergência (o próprio `patience=20` provavelmente pararia o treino sozinho em breve). `best.pt` da época de melhor mAP foi salvo direto no Drive durante o treino (`project=` apontando pro Drive), então nada foi perdido com a queda de sessão.
- **Resultado** (validação final no split de teste, com `best.pt`): mAP50=0.853, mAP50-95=0.617 (média geral).
  - Ótimo: `goggles` (mAP50 **0.984**, +0.563 vs. v1 — o dataset extra resolveu essa classe), `vest` (0.970), `helmet` (0.944), `boots` (0.926)
  - Ainda fraco: `gloves` (mAP50 0.440, recall 26,3% — melhorou vs. v1 mas segue sendo o gargalo do modelo)
- **Comparação com v1**: melhora em todas as classes; ganho mais expressivo em `goggles`. `gloves` continua sendo a limitação principal — precisaria de mais exemplos e/ou anotações mais consistentes entre datasets de origem para melhorar de fato.
- **Correção aplicada**: bug do `augment_dataset.py` (labels não reajustados no `HorizontalFlip`/`Rotate`) foi corrigido nesta versão.
- **Arquivos**: [notebooks/train_colab.ipynb](notebooks/train_colab.ipynb), [results/v2/](results/v2/), pesos em `MyDrive/TCC/runs/train-colab-v2/weights/best.pt`.

## Conjunto de teste real (a partir da v3)

Da v3 em diante, o veredito de cada versão vem do **conjunto de teste real**: 25 frames do vídeo de demonstração (oficina metalúrgica), anotados à mão no Roboflow (`epi-frames-reai`), que nunca entram em nenhum treino. São 102 caixas: helmet 34, boots 19, vest 18, gloves 17, goggles 14. Esse conjunto fica em `novos_frames_dataset/real_world_test/`.

O split de teste dá números otimistas porque vem da mesma distribuição do treino. A v2 tinha mAP50 de 0.853 no split de teste e só **0.235** no vídeo real. Nesse vídeo os EPIs são pequenos: com imgsz=640, a mediana de uma luva fica em ~29 px e a de uma bota em ~16 px. Por isso as tabelas abaixo mostram o resultado em 640 e em 960, que é o `IMGSZ` usado no app.

## v3 — 2026-09-06

- **Dataset**: v2 + ~100 frames reais do vídeo anotados à mão (`scripts/merge_novos_frames.py`: remapeia as classes pelo nome, separa 25 frames como teste real e faz oversample 8x do restante, chegando a ~800 cópias no treino).
- **Modelo**: fine-tuning a partir do `best.pt` da v2.
- **Resultado**: não registrado. Os pesos existem só no Drive (`MyDrive/TCC/runs/train-colab-v3/weights/best.pt`) e não foram avaliados localmente. ⚠️ A pasta local `results/v3/` **não** tem a v3: `best.pt` e `last.pt` ali são cópias dos pesos da v2 (os checkpoints indicam `train-colab-v2`, época 60).

## v4 — 2026-09-24

- **Dataset**: v3 + Construction-PPE (dataset público da Ultralytics, 1.416 imagens, `scripts/baixar_construction_ppe.py`), para reforçar `gloves` (+1.445 instâncias) e `goggles` (+518).
- **Modelo**: fine-tuning a partir do `best.pt` da v3, 40 épocas, patience=15, batch=16, imgsz=640. Completou as 40 épocas.
- **Resultado no split de validação**: mAP50=0.828, mAP50-95=0.567. Não dá para comparar com a v2, porque a validação passou a incluir o Construction-PPE.
- **Resultado no teste real (640)**: mAP50 0.293 (v2: 0.235). Melhorou em quase tudo, mas **gloves continuou em ~0** (mAP50 0.012, recall 0%), mesmo com +1.445 luvas.
- **Arquivos**: `models/best_24-09.pt`, [notebooks/train_colab.ipynb](notebooks/train_colab.ipynb), `MyDrive/TCC/runs/train-colab-v4/`.

## v5 — 2026-09-28

- **Diagnóstico que motivou a versão**: os datasets de origem anotam só parte das classes. O EPI dataset, por exemplo, nunca anotou luva. Uma auditoria com Grounding DINO (detector de vocabulário aberto, independente do nosso modelo; `scripts/auditar_rotulos.py`), com os recortes conferidos a olho, estimou que **~55% das imagens do EPI dataset, ~28% do PPE-CV e ~19% do seafty_goggles têm luva visível sem caixa**. Esses casos ensinam o modelo que luva é fundo, o que explica o recall ~0 de `gloves` mesmo com mais dados.
- **Dataset novo** (`datasets/novos_unificado`, 10.283 imagens de 5 datasets, `scripts/unify_novos_datasets.py`):
  - removidas as pré-anotações erradas do best.pt: 873 "helmet" no seafty_goggles (eram boné/cabelo) e 198 "gloves" no EPI dataset (eram abafadores);
  - Grounding DINO rodado em todas as imagens ([notebooks/pre_anotar_luvas_colab.ipynb](notebooks/pre_anotar_luvas_colab.ipynb)). Limiares calibrados conferindo a olho 30 recortes por faixa de score (`scripts/aplicar_dino.py`): **gloves ≥ 0.5 → +5.357 caixas** (~90% corretas), **goggles ≥ 0.6 → +2.013 caixas** (~85% corretas);
  - upload para o Roboflow (projeto `tcc-epi`, `scripts/subir_roboflow.py`) com tags de prioridade. Revisão manual parcial: ~100 imagens da faixa duvidosa de luva, com luva real encontrada em 96 delas (+168 caixas);
  - as imagens da faixa duvidosa não revisadas (1.624, tags `talvez-*`) foram **tiradas** do treino em vez de ficarem sem caixa (`scripts/preparar_export_v5.py`). Ficaram 8.600 imagens.
- **Treino**: v4 + dataset novo = 25.871 imagens de treino (7.205 caixas de gloves). Fine-tuning a partir da v4, 40 épocas planejadas ([notebooks/train_colab_v5.ipynb](notebooks/train_colab_v5.ipynb)). A cota de GPU do Colab acabou durante a época 30; o `best.pt` usado é o da **época 29**. O modelo já estava em platô (mAP50 de validação 0.859 → 0.872 entre as épocas 19 e 29).
- **Resultado na validação**: mAP50=0.872, mAP50-95=0.617.
- **Resultado no teste real** (mAP50):

  | Classe | v2 (640) | v4 (640) | **v5 (640)** | v2 (960) | v4 (960) | **v5 (960)** |
  |---|---|---|---|---|---|---|
  | **Geral** | 0.235 | 0.293 | **0.333** | 0.301 | 0.323 | **0.402** |
  | helmet | 0.778 | 0.815 | 0.831 | 0.908 | 0.847 | 0.873 |
  | vest | 0.063 | 0.176 | 0.253 | 0.076 | 0.163 | 0.302 |
  | **gloves** | 0.000 | 0.012 | **0.114** | 0.000 | 0.008 | **0.187** |
  | boots | 0.032 | 0.063 | 0.065 | 0.127 | 0.191 | 0.133 |
  | goggles | 0.304 | 0.399 | 0.400 | 0.396 | 0.408 | 0.516 |

  - **Gloves detectado pela primeira vez no vídeo real**: recall de 0% para 12% (640) / 19% (960).
  - Melhor versão até aqui no teste real, nos dois tamanhos.
- **Limitações que continuam**:
  - luvas e botas no vídeo são muito pequenas: 29 px e 16 px em 640;
  - luvas pretas em close-up ainda não são detectadas (os datasets têm sobretudo luvas claras);
  - falso positivo de `vest` em máquina laranja;
  - o dataset da v2 (~12 mil imagens) provavelmente tem o mesmo problema de luva sem caixa e não foi auditado;
  - o teste real é pequeno (17 luvas), então as variações por classe são indicativas.
- **Próximos passos possíveis**:
  - treinar com imgsz=960;
  - rodar a mesma auditoria/pré-anotação do DINO no dataset da v2;
  - adicionar exemplos de luvas escuras;
  - terminar as 10 épocas restantes (resume no mesmo notebook).
- **Arquivos**: `models/best.pt` (= v5), [results/v5/](results/v5/) (curva de treino, comparação no teste real, frames com as detecções), `MyDrive/TCC/runs/train-colab-v5/`. `models/best_antigo.pt` = v2.

## App novo, avaliação com intervalo de confiança e preparação da v7 — 2026-10-21

- **App novo** ([app/servidor.py](app/servidor.py), FastAPI + página própria; o Streamlit fica como referência):
  - o navegador toca o vídeo na velocidade normal e desenha as detecções por cima, sincronizadas por frame;
  - frames que recebem análise: de 25% (480p) e 9% (1080p) no app antigo para 81% e 88%, com 100% dos frames exibidos com detecção sincronizada;
  - cada pessoa ganha um ID que se mantém (YOLO26n-pose + rastreador TrackTrack sem compensação de movimento da câmera);
  - conformidade por pessoa, só cobrando o EPI quando a parte do corpo aparece; ocorrências exportáveis em CSV;
  - boneco clicável para escolher os EPIs; câmera ao vivo.
- **OpenVINO na GPU integrada** ([scripts/exportar_openvino.py](scripts/exportar_openvino.py)): modelo de EPI em 960 de 70 para 23 ms por frame, sem mudar a precisão (v6: mAP50 0,392 nos dois).
- **Avaliação no mesmo pipeline do app** ([scripts/avaliar_teste_real.py](scripts/avaliar_teste_real.py)), com IC 95% por bootstrap e comparação pareada:
  - v5 − v2 (960): **+0,102** de mAP50 (IC +0,025 a +0,167), significativo;
  - luvas, v5 − v2: **+0,179** (IC +0,015 a +0,336), significativo;
  - v6 x v5: diferença não significativa.
- **Estratégias sem retreino** (recorte por pessoa, fatias, TTA, 1280): nenhuma melhora significativa.
- **Análise de erros** ([scripts/analisar_erros.py](scripts/analisar_erros.py)): as luvas perdidas são, na mediana, maiores que as acertadas (118 px contra 89 px no 1080p) — o problema é domínio (luvas escuras, de couro, coloridas; botas escuras e borradas), não tamanho.
- **Modelo padrão do app: v5.** No limiar do app (conf 0,40), F1 0,536 contra 0,497 da v6.
- **Rastreadores** ([scripts/comparar_rastreadores.py](scripts/comparar_rastreadores.py)), IDs por pessoa real:
  - TrackTrack sem GMC: 1,15, com +0,9 ms por frame — o usado no app;
  - TrackTrack padrão: 1,23, com +6,6 ms (+16,5 ms em 1080p, por causa da compensação de movimento da câmera);
  - BoT-SORT 1,54, ByteTrack 1,62, OC-SORT 1,62.
- **Vazamento verificado** ([scripts/mapear_videos.py](scripts/mapear_videos.py)): 8 dos 25 frames de teste vêm dos mesmos vídeos de 22 frames de treino. Não há quase-duplicatas (correlação máxima 0,68), mas há frames da mesma cena.
- **Preparação da v7:**
  - 47 frames escolhidos por aprendizado ativo, já pré-anotados ([scripts/selecionar_frames_anotacao.py](scripts/selecionar_frames_anotacao.py)) — falta revisar no Roboflow;
  - notebook [notebooks/train_colab_v7.ipynb](notebooks/train_colab_v7.ipynb): YOLO26s, augmentations de vídeo, reamostragem de luvas e botas.