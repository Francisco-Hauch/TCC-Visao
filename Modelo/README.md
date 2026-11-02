# Modelo: YOLO26-OBB

Detector de caixas orientadas usado para extrair três classes das ortofotos 2019.

| id | Classe | Avaliação na revisão |
|---|---|---|
| 0 | `area_NãoEdificada` (mata + baldio, sem separação) | Presença por patch |
| 1 | `ponto_onibus` | Contagem |
| 2 | `faixa_pedestre` | Contagem |

Na ortofoto, mata e terreno baldio têm o mesmo papel para a análise. Por isso a
métrica mede área não edificada visível, o que inclui verde dentro de lote construído.

## Dataset

`dataset/`: 348 patches 512×512 rotulados à mão, 272 de treino e 76 de validação.
Não há patch repetido entre os dois conjuntos.

## Treino

| Parâmetro | Valor |
|---|---|
| Modelo base | `yolo26x-obb.pt` |
| imgsz | 1024 |
| Épocas / patience | 100 / 30 (rodou as 100) |
| Batch | 4 |
| Optimizer | auto |
| Workers | 0 |
| AMP / seed / deterministic | True / 0 / True |

O `best.pt` é o checkpoint da época 83, a de maior mAP50-95. Foi treinado com
Ultralytics 8.4.163 e não está no repositório (126 MB). Curvas, matriz de confusão
e `results.csv` estão em `runs/yolo26x_obb_v4_xl/`.

## Desempenho na validação

Revalidação do `best.pt` nos 76 patches de validação (`validar_modelo.py`):

| Classe | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **Geral** | 0,787 | 0,690 | **0,735** | 0,484 |
| `area_NãoEdificada` | 0,611 | 0,612 | 0,666 | 0,310 |
| `ponto_onibus` | 0,843 | 0,578 | 0,657 | 0,465 |
| `faixa_pedestre` | 0,907 | 0,881 | 0,883 | 0,676 |

No conjunto de treino, o mAP50 é 0,970.

## Inferência em lote

`inferir_lote.py` roda o `best.pt` nos 45.523 patches com conf ≥ 0,25 e
georreferencia cada caixa: **48.345 detecções**.

`filtro_tamanho.py` descarta a caixa cujo lado menor ou maior fique abaixo do
mínimo da classe nas caixas rotuladas, e restam 48.286 detecções. Os mínimos, em
px do patch 512 (lado menor × lado maior), são fixos no script:

| Classe | Mínimo |
|---|---|
| `area_NãoEdificada` | 21,2 × 25,6 |
| `ponto_onibus` | 6,5 × 15,0 |
| `faixa_pedestre` | 5,0 × 18,6 |

A contagem por patch usa conf ≥ 0,25 sobre o lote filtrado. A revisão mostrou
precisão alta já na faixa 0,25–0,5, e um corte de 0,5 descartaria ~28% dos pontos
de ônibus reais. Nesse corte, a área não edificada aparece em 54,7% dos patches.

## Revisão manual de 400 patches

A amostra tem 400 patches com detecção, espalhados pelas faixas de confiança de
cada classe, a pelo menos ~200 m de qualquer imagem do dataset. Foram alterados 54.

Precisão por presença no patch, por faixa da maior confiança da classe (entre
parênteses, o número de patches):

| Confiança | Área | Ônibus | Faixa |
|---|---|---|---|
| 0,25–0,4 | 0,96 (52) | 0,78 (37) | 0,96 (24) |
| 0,4–0,5 | 1,00 (32) | 0,73 (15) | 0,67 (12) |
| 0,5–0,7 | 1,00 (51) | 0,79 (28) | 0,76 (21) |
| 0,7–0,9 | 0,99 (80) | 0,92 (63) | 0,96 (73) |
| 0,9–1,0 | 1,00 (38) | 1,00 (13) | 0,95 (22) |
| **Total** | **0,99** | **0,86** | **0,92** |

Por contagem: ônibus com 157 acertos, 26 falsos positivos e 4 falsos negativos;
faixa com 229, 20 e 1. Área, por presença: 251, 3 e 0.

**Limites:** o recall só é medido dentro dos 400, todos com alguma detecção, e os
346 patches sem alteração podem refletir ancoragem no pré-rótulo.

## Limitação conhecida: export ONNX

Usar o `.pt`. O `.onnx` exporta sem erro, mas perde acurácia: em um teste com uma
versão anterior do modelo, o mAP50 caiu de 0,72 para 0,52. A geometria das caixas
se mantém, mas a confiança sai mais baixa e quase-duplicatas não são suprimidas. O
problema aparenta ser do cabeçalho `OBB26` do Ultralytics.

## Ambiente

Dependências em `requirements.txt`. No Windows, usar `workers=0`: com subprocessos,
o DataLoader falha com o erro 1455.
