# Teste real (25 frames, 102 caixas)

mAP50 com intervalo de confianca de 95% (bootstrap, 1000 reamostragens das imagens). Tempo = mediana por imagem na CPU (PyTorch), so para comparar as estrategias entre si.

## Modelo v5_pytorch_cpu

| Estrategia | mAP50 geral (IC 95%) | Luvas | Botas | ms/imagem |
|---|---|---|---|---|
| frame_960 | 0,420 (0,343-0,553) | 0,173 | 0,156 | 67 |

## Modelo v5_openvino_gpu

| Estrategia | mAP50 geral (IC 95%) | Luvas | Botas | ms/imagem |
|---|---|---|---|---|
| frame_960 | 0,419 (0,343-0,552) | 0,174 | 0,156 | 24 |

## Diferenca para a linha de base (bootstrap pareado)

Dif. = estrategia - linha de base, nas mesmas reamostragens. Se o IC 95% inclui 0, a diferenca nao e significativa com 25 imagens.

| Modelo | Estrategia | Classe | Dif. media | IC 95% | % amostras melhor |
|---|---|---|---|---|---|

## Contagem no limiar do app (conf 0,40; IoU >= 0,5)

| Modelo | Estrategia | Classe | GT | TP | FP | FN | Precisao | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|
| v5_pytorch_cpu | frame_960 | helmet | 34 | 28 | 4 | 6 | 0,875 | 0,824 | 0,848 |
| v5_pytorch_cpu | frame_960 | vest | 18 | 4 | 4 | 14 | 0,500 | 0,222 | 0,308 |
| v5_pytorch_cpu | frame_960 | gloves | 17 | 2 | 1 | 15 | 0,667 | 0,118 | 0,200 |
| v5_pytorch_cpu | frame_960 | boots | 19 | 1 | 0 | 18 | 1,000 | 0,053 | 0,100 |
| v5_pytorch_cpu | frame_960 | goggles | 14 | 6 | 1 | 8 | 0,857 | 0,429 | 0,571 |
| v5_pytorch_cpu | frame_960 | TOTAL | 102 | 41 | 10 | 61 | 0,804 | 0,402 | 0,536 |
| v5_openvino_gpu | frame_960 | helmet | 34 | 28 | 4 | 6 | 0,875 | 0,824 | 0,848 |
| v5_openvino_gpu | frame_960 | vest | 18 | 4 | 4 | 14 | 0,500 | 0,222 | 0,308 |
| v5_openvino_gpu | frame_960 | gloves | 17 | 2 | 1 | 15 | 0,667 | 0,118 | 0,200 |
| v5_openvino_gpu | frame_960 | boots | 19 | 1 | 0 | 18 | 1,000 | 0,053 | 0,100 |
| v5_openvino_gpu | frame_960 | goggles | 14 | 6 | 1 | 8 | 0,857 | 0,429 | 0,571 |
| v5_openvino_gpu | frame_960 | TOTAL | 102 | 41 | 10 | 61 | 0,804 | 0,402 | 0,536 |

## Comparacao entre modelos (bootstrap pareado)

Dif. = modelo - primeiro modelo da lista, na mesma estrategia e nas mesmas reamostragens.

| Estrategia | Base | Modelo | Classe | Dif. media | IC 95% | % amostras melhor |
|---|---|---|---|---|---|---|
| frame_960 | v5_pytorch_cpu | v5_openvino_gpu | geral | -0,001 | -0,004 a 0,001 | 30% |
| frame_960 | v5_pytorch_cpu | v5_openvino_gpu | gloves | 0,001 | -0,001 a 0,003 | 81% |
| frame_960 | v5_pytorch_cpu | v5_openvino_gpu | boots | -0,000 | -0,001 a 0,000 | 12% |
