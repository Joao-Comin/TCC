# Teste real (25 frames, 102 caixas) - estrategias de inferencia

mAP50 com intervalo de confianca de 95% (bootstrap, 1000 reamostragens das imagens). Tempo = mediana por imagem na CPU (PyTorch), so para comparar as estrategias entre si.

## Modelo v6

| Estrategia | mAP50 geral (IC 95%) | Luvas | Botas | ms/imagem |
|---|---|---|---|---|
| frame_640 | 0,323 (0,242-0,455) | 0,106 | 0,055 | 38 |
| frame_960 | 0,392 (0,326-0,518) | 0,153 | 0,050 | 70 |
| frame_1280 | 0,397 (0,329-0,509) | 0,142 | 0,119 | 129 |
| frame_960_tta | 0,364 (0,282-0,496) | 0,142 | 0,059 | 159 |
| frame_640+recortes | 0,389 (0,312-0,540) | 0,134 | 0,111 | 182 |
| frame_960+recortes | 0,407 (0,339-0,551) | 0,144 | 0,102 | 217 |
| frame_960+recortes960 | 0,399 (0,324-0,539) | 0,116 | 0,138 | 579 |
| frame_960+fatias2x2 | 0,354 (0,304-0,469) | 0,100 | 0,149 | 505 |

## Modelo v5

| Estrategia | mAP50 geral (IC 95%) | Luvas | Botas | ms/imagem |
|---|---|---|---|---|
| frame_640 | 0,339 (0,269-0,475) | 0,077 | 0,059 | 46 |
| frame_960 | 0,420 (0,343-0,553) | 0,173 | 0,156 | 70 |
| frame_1280 | 0,377 (0,311-0,507) | 0,070 | 0,114 | 133 |
| frame_960_tta | 0,425 (0,362-0,552) | 0,151 | 0,164 | 154 |
| frame_640+recortes | 0,352 (0,286-0,503) | 0,062 | 0,126 | 163 |
| frame_960+recortes | 0,405 (0,334-0,540) | 0,162 | 0,143 | 227 |
| frame_960+recortes960 | 0,386 (0,315-0,518) | 0,161 | 0,129 | 297 |
| frame_960+fatias2x2 | 0,364 (0,305-0,473) | 0,080 | 0,132 | 370 |

## Diferenca para a linha de base (bootstrap pareado)

Dif. = estrategia - linha de base, nas mesmas reamostragens. Se o IC 95% inclui 0, a diferenca nao e significativa com 25 imagens.

| Modelo | Estrategia | Classe | Dif. media | IC 95% | % amostras melhor |
|---|---|---|---|---|---|
| v6 | frame_1280 | geral | 0,002 | -0,060 a 0,062 | 51% |
| v6 | frame_1280 | gloves | -0,025 | -0,135 a 0,068 | 34% |
| v6 | frame_1280 | boots | 0,066 | -0,034 a 0,275 | 78% |
| v6 | frame_960_tta | geral | -0,031 | -0,087 a 0,012 | 10% |
| v6 | frame_960_tta | gloves | -0,015 | -0,052 a 0,018 | 22% |
| v6 | frame_960_tta | boots | 0,011 | -0,000 a 0,033 | 96% |
| v6 | frame_640+recortes | geral | 0,005 | -0,079 a 0,107 | 52% |
| v6 | frame_640+recortes | gloves | 0,002 | -0,205 a 0,203 | 49% |
| v6 | frame_640+recortes | boots | 0,078 | -0,016 a 0,214 | 95% |
| v6 | frame_960+recortes | geral | 0,019 | -0,031 a 0,081 | 72% |
| v6 | frame_960+recortes | gloves | -0,002 | -0,100 a 0,106 | 47% |
| v6 | frame_960+recortes | boots | 0,065 | -0,007 a 0,228 | 94% |
| v6 | frame_960+recortes960 | geral | 0,009 | -0,052 a 0,083 | 59% |
| v6 | frame_960+recortes960 | gloves | -0,032 | -0,122 a 0,050 | 23% |
| v6 | frame_960+recortes960 | boots | 0,087 | -0,032 a 0,335 | 78% |
| v6 | frame_960+fatias2x2 | geral | -0,029 | -0,091 a 0,037 | 17% |
| v6 | frame_960+fatias2x2 | gloves | -0,046 | -0,129 a 0,021 | 14% |
| v6 | frame_960+fatias2x2 | boots | 0,101 | -0,013 a 0,349 | 87% |
| v5 | frame_1280 | geral | -0,037 | -0,073 a 0,002 | 4% |
| v5 | frame_1280 | gloves | -0,088 | -0,203 a 0,030 | 8% |
| v5 | frame_1280 | boots | -0,037 | -0,110 a 0,051 | 8% |
| v5 | frame_960_tta | geral | 0,008 | -0,037 a 0,062 | 62% |
| v5 | frame_960_tta | gloves | -0,009 | -0,073 a 0,064 | 37% |
| v5 | frame_960_tta | boots | 0,014 | -0,050 a 0,106 | 61% |
| v5 | frame_640+recortes | geral | -0,058 | -0,098 a -0,001 | 2% |
| v5 | frame_640+recortes | gloves | -0,106 | -0,240 a 0,040 | 12% |
| v5 | frame_640+recortes | boots | -0,021 | -0,094 a 0,052 | 26% |
| v5 | frame_960+recortes | geral | -0,010 | -0,042 a 0,025 | 26% |
| v5 | frame_960+recortes | gloves | -0,010 | -0,047 a 0,014 | 20% |
| v5 | frame_960+recortes | boots | -0,004 | -0,059 a 0,078 | 27% |
| v5 | frame_960+recortes960 | geral | -0,029 | -0,065 a 0,002 | 3% |
| v5 | frame_960+recortes960 | gloves | -0,011 | -0,055 a 0,026 | 27% |
| v5 | frame_960+recortes960 | boots | -0,027 | -0,122 a 0,015 | 16% |
| v5 | frame_960+fatias2x2 | geral | -0,053 | -0,092 a -0,017 | 0% |
| v5 | frame_960+fatias2x2 | gloves | -0,091 | -0,178 a -0,004 | 2% |
| v5 | frame_960+fatias2x2 | boots | -0,019 | -0,087 a 0,008 | 10% |

## Contagem no limiar do app (conf 0,40; IoU >= 0,5)

| Modelo | Estrategia | Classe | GT | TP | FP | FN | Precisao | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|
| v6 | frame_960 | helmet | 34 | 24 | 3 | 10 | 0,889 | 0,706 | 0,787 |
| v6 | frame_960 | vest | 18 | 3 | 1 | 15 | 0,750 | 0,167 | 0,273 |
| v6 | frame_960 | gloves | 17 | 2 | 2 | 15 | 0,500 | 0,118 | 0,190 |
| v6 | frame_960 | boots | 19 | 1 | 1 | 18 | 0,500 | 0,053 | 0,095 |
| v6 | frame_960 | goggles | 14 | 7 | 3 | 7 | 0,700 | 0,500 | 0,583 |
| v6 | frame_960 | TOTAL | 102 | 37 | 10 | 65 | 0,787 | 0,363 | 0,497 |
| v5 | frame_960 | helmet | 34 | 28 | 4 | 6 | 0,875 | 0,824 | 0,848 |
| v5 | frame_960 | vest | 18 | 4 | 4 | 14 | 0,500 | 0,222 | 0,308 |
| v5 | frame_960 | gloves | 17 | 2 | 1 | 15 | 0,667 | 0,118 | 0,200 |
| v5 | frame_960 | boots | 19 | 1 | 0 | 18 | 1,000 | 0,053 | 0,100 |
| v5 | frame_960 | goggles | 14 | 6 | 1 | 8 | 0,857 | 0,429 | 0,571 |
| v5 | frame_960 | TOTAL | 102 | 41 | 10 | 61 | 0,804 | 0,402 | 0,536 |
