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

<!-- ## v3 — data
- O que mudou em relação à v2 (dataset, hiperparâmetros, correções aplicadas)
- Resultado e comparação com v2
-->
