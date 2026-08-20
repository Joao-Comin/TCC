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

<!-- ## v2 — data
- O que mudou em relação à v1 (dataset, hiperparâmetros, correções aplicadas)
- Resultado e comparação com v1
-->
