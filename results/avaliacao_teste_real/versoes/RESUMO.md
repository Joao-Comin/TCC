# Teste real (25 frames, 102 caixas)

mAP50 com intervalo de confianca de 95% (bootstrap, 1000 reamostragens das imagens). Tempo = mediana por imagem na CPU (PyTorch), so para comparar as estrategias entre si.

## Modelo v2

| Estrategia | mAP50 geral (IC 95%) | Luvas | Botas | ms/imagem |
|---|---|---|---|---|
| frame_640 | 0,264 (0,207-0,362) | 0,000 | 0,036 | 35 |
| frame_960 | 0,317 (0,251-0,438) | 0,000 | 0,139 | 70 |

## Modelo v4

| Estrategia | mAP50 geral (IC 95%) | Luvas | Botas | ms/imagem |
|---|---|---|---|---|
| frame_640 | 0,304 (0,233-0,424) | 0,013 | 0,060 | 37 |
| frame_960 | 0,324 (0,257-0,455) | 0,015 | 0,154 | 68 |

## Modelo v5

| Estrategia | mAP50 geral (IC 95%) | Luvas | Botas | ms/imagem |
|---|---|---|---|---|
| frame_640 | 0,339 (0,269-0,475) | 0,077 | 0,059 | 36 |
| frame_960 | 0,420 (0,343-0,553) | 0,173 | 0,156 | 67 |

## Modelo v6

| Estrategia | mAP50 geral (IC 95%) | Luvas | Botas | ms/imagem |
|---|---|---|---|---|
| frame_640 | 0,323 (0,242-0,455) | 0,106 | 0,055 | 37 |
| frame_960 | 0,392 (0,326-0,518) | 0,153 | 0,050 | 67 |

## Diferenca para a linha de base (bootstrap pareado)

Dif. = estrategia - linha de base, nas mesmas reamostragens. Se o IC 95% inclui 0, a diferenca nao e significativa com 25 imagens.

| Modelo | Estrategia | Classe | Dif. media | IC 95% | % amostras melhor |
|---|---|---|---|---|---|

## Contagem no limiar do app (conf 0,40; IoU >= 0,5)

| Modelo | Estrategia | Classe | GT | TP | FP | FN | Precisao | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|
| v2 | frame_960 | helmet | 34 | 27 | 3 | 7 | 0,900 | 0,794 | 0,844 |
| v2 | frame_960 | vest | 18 | 1 | 3 | 17 | 0,250 | 0,056 | 0,091 |
| v2 | frame_960 | gloves | 17 | 0 | 0 | 17 | 0,000 | 0,000 | 0,000 |
| v2 | frame_960 | boots | 19 | 1 | 0 | 18 | 1,000 | 0,053 | 0,100 |
| v2 | frame_960 | goggles | 14 | 3 | 0 | 11 | 1,000 | 0,214 | 0,353 |
| v2 | frame_960 | TOTAL | 102 | 32 | 6 | 70 | 0,842 | 0,314 | 0,457 |
| v4 | frame_960 | helmet | 34 | 26 | 6 | 8 | 0,812 | 0,765 | 0,788 |
| v4 | frame_960 | vest | 18 | 0 | 2 | 18 | 0,000 | 0,000 | 0,000 |
| v4 | frame_960 | gloves | 17 | 0 | 0 | 17 | 0,000 | 0,000 | 0,000 |
| v4 | frame_960 | boots | 19 | 0 | 0 | 19 | 0,000 | 0,000 | 0,000 |
| v4 | frame_960 | goggles | 14 | 4 | 0 | 10 | 1,000 | 0,286 | 0,444 |
| v4 | frame_960 | TOTAL | 102 | 30 | 8 | 72 | 0,789 | 0,294 | 0,429 |
| v5 | frame_960 | helmet | 34 | 28 | 4 | 6 | 0,875 | 0,824 | 0,848 |
| v5 | frame_960 | vest | 18 | 4 | 4 | 14 | 0,500 | 0,222 | 0,308 |
| v5 | frame_960 | gloves | 17 | 2 | 1 | 15 | 0,667 | 0,118 | 0,200 |
| v5 | frame_960 | boots | 19 | 1 | 0 | 18 | 1,000 | 0,053 | 0,100 |
| v5 | frame_960 | goggles | 14 | 6 | 1 | 8 | 0,857 | 0,429 | 0,571 |
| v5 | frame_960 | TOTAL | 102 | 41 | 10 | 61 | 0,804 | 0,402 | 0,536 |
| v6 | frame_960 | helmet | 34 | 24 | 3 | 10 | 0,889 | 0,706 | 0,787 |
| v6 | frame_960 | vest | 18 | 3 | 1 | 15 | 0,750 | 0,167 | 0,273 |
| v6 | frame_960 | gloves | 17 | 2 | 2 | 15 | 0,500 | 0,118 | 0,190 |
| v6 | frame_960 | boots | 19 | 1 | 1 | 18 | 0,500 | 0,053 | 0,095 |
| v6 | frame_960 | goggles | 14 | 7 | 3 | 7 | 0,700 | 0,500 | 0,583 |
| v6 | frame_960 | TOTAL | 102 | 37 | 10 | 65 | 0,787 | 0,363 | 0,497 |

## Comparacao entre modelos (bootstrap pareado)

Dif. = modelo - primeiro modelo da lista, na mesma estrategia e nas mesmas reamostragens.

| Estrategia | Base | Modelo | Classe | Dif. media | IC 95% | % amostras melhor |
|---|---|---|---|---|---|---|
| frame_640 | v2 | v4 | geral | 0,045 | 0,010 a 0,088 | 99% |
| frame_640 | v2 | v4 | gloves | 0,020 | 0,002 a 0,067 | 100% |
| frame_640 | v2 | v4 | boots | 0,045 | -0,003 a 0,171 | 95% |
| frame_640 | v2 | v5 | geral | 0,078 | 0,035 a 0,154 | 100% |
| frame_640 | v2 | v5 | gloves | 0,095 | 0,017 a 0,241 | 100% |
| frame_640 | v2 | v5 | boots | 0,030 | -0,045 a 0,127 | 82% |
| frame_640 | v2 | v6 | geral | 0,062 | -0,012 a 0,152 | 95% |
| frame_640 | v2 | v6 | gloves | 0,136 | 0,027 a 0,325 | 100% |
| frame_640 | v2 | v6 | boots | 0,028 | -0,101 a 0,159 | 73% |
| frame_960 | v2 | v4 | geral | 0,010 | -0,028 a 0,052 | 72% |
| frame_960 | v2 | v4 | gloves | 0,023 | 0,002 a 0,075 | 100% |
| frame_960 | v2 | v4 | boots | 0,019 | -0,138 a 0,203 | 60% |
| frame_960 | v2 | v5 | geral | 0,102 | 0,025 a 0,167 | 99% |
| frame_960 | v2 | v5 | gloves | 0,179 | 0,015 a 0,336 | 100% |
| frame_960 | v2 | v5 | boots | -0,002 | -0,159 a 0,101 | 54% |
| frame_960 | v2 | v6 | geral | 0,076 | -0,003 a 0,156 | 97% |
| frame_960 | v2 | v6 | gloves | 0,167 | 0,009 a 0,341 | 100% |
| frame_960 | v2 | v6 | boots | -0,112 | -0,447 a 0,062 | 17% |
