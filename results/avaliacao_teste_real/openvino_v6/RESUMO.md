# Teste real (25 frames, 102 caixas)

mAP50 com intervalo de confianca de 95% (bootstrap, 1000 reamostragens das imagens). Tempo = mediana por imagem na CPU (PyTorch), so para comparar as estrategias entre si.

## Modelo v6_pytorch_cpu

| Estrategia | mAP50 geral (IC 95%) | Luvas | Botas | ms/imagem |
|---|---|---|---|---|
| frame_960 | 0,392 (0,326-0,518) | 0,153 | 0,050 | 77 |

## Modelo v6_openvino_gpu

| Estrategia | mAP50 geral (IC 95%) | Luvas | Botas | ms/imagem |
|---|---|---|---|---|
| frame_960 | 0,392 (0,326-0,519) | 0,153 | 0,050 | 24 |

## Diferenca para a linha de base (bootstrap pareado)

Dif. = estrategia - linha de base, nas mesmas reamostragens. Se o IC 95% inclui 0, a diferenca nao e significativa com 25 imagens.

| Modelo | Estrategia | Classe | Dif. media | IC 95% | % amostras melhor |
|---|---|---|---|---|---|

## Contagem no limiar do app (conf 0,40; IoU >= 0,5)

| Modelo | Estrategia | Classe | GT | TP | FP | FN | Precisao | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|
| v6_pytorch_cpu | frame_960 | helmet | 34 | 24 | 3 | 10 | 0,889 | 0,706 | 0,787 |
| v6_pytorch_cpu | frame_960 | vest | 18 | 3 | 1 | 15 | 0,750 | 0,167 | 0,273 |
| v6_pytorch_cpu | frame_960 | gloves | 17 | 2 | 2 | 15 | 0,500 | 0,118 | 0,190 |
| v6_pytorch_cpu | frame_960 | boots | 19 | 1 | 1 | 18 | 0,500 | 0,053 | 0,095 |
| v6_pytorch_cpu | frame_960 | goggles | 14 | 7 | 3 | 7 | 0,700 | 0,500 | 0,583 |
| v6_pytorch_cpu | frame_960 | TOTAL | 102 | 37 | 10 | 65 | 0,787 | 0,363 | 0,497 |
| v6_openvino_gpu | frame_960 | helmet | 34 | 24 | 3 | 10 | 0,889 | 0,706 | 0,787 |
| v6_openvino_gpu | frame_960 | vest | 18 | 3 | 1 | 15 | 0,750 | 0,167 | 0,273 |
| v6_openvino_gpu | frame_960 | gloves | 17 | 2 | 2 | 15 | 0,500 | 0,118 | 0,190 |
| v6_openvino_gpu | frame_960 | boots | 19 | 1 | 1 | 18 | 0,500 | 0,053 | 0,095 |
| v6_openvino_gpu | frame_960 | goggles | 14 | 7 | 3 | 7 | 0,700 | 0,500 | 0,583 |
| v6_openvino_gpu | frame_960 | TOTAL | 102 | 37 | 10 | 65 | 0,787 | 0,363 | 0,497 |

## Comparacao entre modelos (bootstrap pareado)

Dif. = modelo - primeiro modelo da lista, na mesma estrategia e nas mesmas reamostragens.

| Estrategia | Base | Modelo | Classe | Dif. media | IC 95% | % amostras melhor |
|---|---|---|---|---|---|---|
| frame_960 | v6_pytorch_cpu | v6_openvino_gpu | geral | -0,000 | -0,001 a 0,002 | 33% |
| frame_960 | v6_pytorch_cpu | v6_openvino_gpu | gloves | -0,000 | -0,001 a 0,001 | 23% |
| frame_960 | v6_pytorch_cpu | v6_openvino_gpu | boots | 0,000 | 0,000 a 0,000 | 0% |
