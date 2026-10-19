# Modelo — YOLO26-OBB (área não edificada / ponto de ônibus / faixa de pedestre)

Diário de bordo do treino, inferência e revisão. Contexto geral do TCC no
`CLAUDE.md` da raiz.

## Estado (2026-10-18)

- **Modelo treinado e em uso: v4 XL** — `runs/yolo26x_obb_v4_xl/weights/best.pt`
  (yolo26x-obb, 348 imagens). Ver "Treino do v4 XL" abaixo.
- Inferência em lote nos 45.523 patches: feita, com filtro de tamanho mínimo.
  Corte de confiança em uso: **0,25**.
- Revisão de 400: **concluída** (precisão por presença: área 0,99, faixa 0,92,
  ônibus 0,86).
- Export ONNX com bug de acurácia: usar o `.pt`.
- As seções de 2026-09-27 a 2026-10-04 são do **v1 nano** (150 imagens), o
  primeiro ciclo; ficam como registro. O run do v1 está em `D:\CityVision_lixeira`.

Classes (`dataset/data.yaml`): `0 area_NãoEdificada` (**mata + baldio, sem
separação — decisão travada**), `1 ponto_onibus`, `2 faixa_pedestre`.
Ambiente: Python 3.13 global com `torch==2.14.0+cu130` (RTX 5060 Ti 16 GB) e
`ultralytics` — ver `requirements.txt`.

## v1 nano — hiperparâmetros (2026-09-27)

**Treino** (`scripts/treinar.py::CONFIG`):

| métrica | valor | por quê |
|---|---|---|
| modelo base | `yolo26n-obb.pt` | nano — dataset pequeno (117 imgs), menor risco de overfitting |
| imgsz | 1024 | upscale 2x de 512 nativo — grid de features maior p/ objetos pequenos |
| epochs | 100 | |
| batch | 16 | |
| patience | 30 | tolerância maior que o usual — val set pequeno (33 imgs) é mais ruidoso |
| optimizer | `auto` | Ultralytics escolhe (tipicamente AdamW) e ajusta lr0/momentum sozinho |
| lr0 | default (não sobrescrito) | deixa o `optimizer=auto` decidir |
| device / workers / seed / amp | 0 / 0 / 0 / True | defaults de ambiente, não perguntados como "métrica" |

**Export ONNX** (`scripts/exportar_onnx.py::CONFIG`):

| parâmetro | valor | por quê |
|---|---|---|
| imgsz | 1024 | igual ao treino |
| half (fp16) | False | fp32, compatível com CPU e qualquer GPU |
| dynamic | False | grafo estático, batch=1, 1024×1024 fixo |
| simplify | True | roda onnxslim no grafo |
| opset | default (não fixado) | deixa o Ultralytics escolher a versão compatível |
| nms embutido | False | pós-processamento fica no código Python, não no grafo |

## v1 nano — crashes no treino (2026-09-27)

Primeira tentativa de treino (`python scripts/treinar.py`, rodada pelo
usuário) crashou no epoch 3, na validação, com:

```
RuntimeError: Couldn't open shared file mapping: <torch_...>, error code: <1455>
...
RuntimeError: Pin memory thread exited unexpectedly
```

Erro 1455 do Windows = `ERROR_COMMITMENT_LIMIT` — bug conhecido do
DataLoader multiprocessing do PyTorch no Windows quando `workers > 0`
(tenta compartilhar tensores entre processos via memory-mapped file e o
sistema recusa). **Corrigido:** `workers` mudou de 4 para 0 no `CONFIG` de
`scripts/treinar.py` — com um dataset de 117 imagens, carregar no processo
principal (sem subprocessos) não tem custo de performance relevante.
O aviso de "OOM" da CUDA que apareceu no epoch 2 foi transitório (o
treino continuou normalmente logo em seguida) — não foi a causa do crash.

A run parcial gerada antes do crash (`runs/yolo26_obb_baldio_onibus_faixa/`,
só 2 épocas, mAP50≈0.03) ainda está em disco, não é um resultado
utilizável — apagar antes de re-treinar (senão o Ultralytics cria uma
pasta nova numerada ao lado, `..._faixa2`, em vez de sobrescrever).

**Segunda tentativa:** apagada a run acima, reiniciado o treino (mesmo
`workers=0`) e chegou ao epoch 34 com resultado razoável (mAP50≈0,69)
antes de crashar de novo — desta vez por `numpy._core._exceptions.
_ArrayMemoryError` (RAM do sistema esgotada). Causa: o usuário tinha um
jogo aberto na máquina consumindo memória em paralelo, não um bug no
script. Corrigido fechando o jogo (RAM livre voltou de ~7,5GB para
~18GB de 31,9GB totais).

**Retomada:** `scripts/retomar_treino.py` (novo) — usa `model.train
(resume=True)` a partir do `last.pt` da run interrompida, preservando as
34 épocas já treinadas em vez de reiniciar do zero.

## v1 nano — resultado do treino (2026-09-27)

Treino terminou por **early stopping** no epoch 90/100 (patience=30, sem
melhora desde o epoch 60 — `best.pt` é o checkpoint do epoch 60).

| classe | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **geral** | 0,897 | 0,600 | **0,718** | 0,437 |
| area_NãoEdificada (baldio) | 1,000 | 0,469 | 0,660 | 0,237 |
| ponto_onibus | 0,789 | 0,545 | 0,614 | 0,423 |
| faixa_pedestre | 0,901 | 0,786 | 0,880 | 0,650 |

**Baldio é a classe mais fraca** (recall baixo, mAP50-95 baixo — quando
detecta, a caixa não encaixa bem) — esperado dado o tamanho do dataset
(73 instâncias de treino) e a ambiguidade intrínseca do contorno de
terreno vago. É a classe-alvo principal do TCC (ver `CLAUDE.md`), então
vale mais dados rotulados ou complementar com SAM3 zero-shot antes de
considerar este resultado suficiente.

**Export ONNX:** `runs/yolo26_obb_baldio_onibus_faixa/weights/best.onnx`
(9,7 MB, opset 18, imgsz 1024, fp32, estático, sem NMS embutido) — exportou
sem erro, mas **falhou na validação de paridade** (ver seção abaixo).

## ⚠️ Bug de exportação ONNX encontrado (2026-09-27)

Ao validar o `.onnx` contra o `.pt` (script `scripts/validar_onnx.py`),
o resultado **não bate**:

| | mAP50 | mAP50-95 |
|---|---|---|
| `best.pt` | 0,720 | 0,438 |
| `best.onnx` | 0,522 | 0,354 |

Diagnóstico feito (script `validar_onnx.py` + inspeção manual de caixas):
- A **geometria das caixas exporta certinho** (mesma posição/tamanho/
  ângulo entre `.pt` e `.onnx` pra um mesmo objeto).
- A **confiança sai sistematicamente mais baixa no `.onnx`**, e
  caixas quase-duplicadas que o `.pt` suprime (NMS) não são suprimidas
  no `.onnx`.
- Confirmado no código-fonte instalado do Ultralytics
  (`ultralytics/nn/modules/head.py`, classe `OBB26`) que o YOLO26-OBB é
  uma arquitetura **end2end (NMS-free)** — a seleção top-300 já acontece
  dentro do `forward()`, igual para os dois caminhos. Ou seja, não é o
  parâmetro `nms=False` que escolhemos que causa isso.
- **Não é o `onnxslim`**: reexportado com `simplify=False`, resultado
  idêntico (0,5223/0,3539 nas duas execuções, até a 4ª casa decimal).
- **Não é versão desatualizada**: atualizado `ultralytics` 8.4.150 →
  8.4.163 (mais recente disponível) e reexportado — resultado idêntico.

**Conclusão: aparenta ser um bug real de exportação ONNX do Ultralytics,
específico do cabeçalho novo `OBB26` (YOLO26 pra caixas orientadas) —
não algo causado pela nossa configuração.** Não investigado mais a fundo
(exigiria comparar op a op dentro do grafo ONNX, ex. o decode do ângulo).

Nota lateral (não é a causa do bug de acurácia, só afeta desempenho):
o `onnxruntime-gpu` avisa `No registered plugin EP device found for
'CUDAExecutionProvider'` e parece cair pra CPU — possível
incompatibilidade entre `onnxruntime-gpu==1.30.0` e o CUDA 13.0 do
driver instalado. Não resolvido.

**Enquanto isso, usar o `.pt` para inferência** (validado, mAP50=0,72,
funciona perfeitamente via `ultralytics`/`torch`) — o `.onnx` existe mas
não é confiável para uso até este bug ser entendido/corrigido.

## v1 nano — inferência em lote (2026-09-27/2026-10-04)

`scripts/inferir_lote.py` roda `best.pt` sobre qualquer pasta de patches
(padrão: `TCC-data/Patches512`, os 45.523 recortes 512×512/z19 de toda a
cobertura) e georreferencia cada detecção (pixel→lat/lon via `tile2deg()`,
mesma convenção de `TCC/rotular_tiles.py`). Rodado no corpus inteiro:
**45.523 imagens, 48.575 detecções em 687s** (~15ms/img). Saída:
`runs/yolo26_obb_baldio_onibus_faixa/deteccoes_lote.csv`.

Por classe (todas confianças): `area_NãoEdificada` 33.901 (conf. média
0,377) — implausível (42% dos patches da cidade), sinal de muito falso
positivo em confiança baixa; `ponto_onibus` 7.699 (0,509); `faixa_pedestre`
6.975 (0,658). **Filtrado para conf≥0,5** (`deteccoes_lote_conf50.csv`):
cai para 12.313 detecções — `area_NãoEdificada` 4.129 (8,3% dos patches,
bem mais plausível), `ponto_onibus` 3.330, `faixa_pedestre` 4.854.

## v1 nano — validação por bairro/renda (2026-10-04)

`scripts/validar_confiabilidade_renda.py` cruza as 150 imagens rotuladas
com bairro/CEP (`TCC-data/tiles_rotulos_quadra.csv`) e renda (censo
2010/2022, `VetoresFeatures/setor_censitario_*.geojson`), comparando
rótulo manual × predição do próprio modelo nessas mesmas imagens, e
depois olhando a densidade de detecção no restante da cidade (exclui as
150) por faixa de renda.

**Achado 1 — confiabilidade do detector de baldio muda de direção por
renda:** em bairro de renda baixa, o modelo **perde** mais da metade dos
baldios reais (12 falso-negativo vs 10 acerto, taxa de acerto 0,67); em
bairro de renda alta, o modelo **inventa** baldio que não existe (8
falso-positivo vs 10 acerto, taxa 0,74). `ponto_onibus` e
`faixa_pedestre` ficam estáveis (0,81–0,97) em todas as faixas — o
problema é específico da classe `area_NãoEdificada`.

**Achado 2 — densidade de `area_NãoEdificada` no restante da cidade
aumenta com a renda** (6,8/100 patches em renda baixa → 9,7/100 em renda
alta) — à primeira vista parecia bater com o viés do Achado 1 (sobredetecção
em renda alta), mas **ver a ressalva abaixo, isso mudou depois que o
usuário esclareceu a definição da classe.**

### ⚠️ Classe `area_NãoEdificada` inclui mata, não é só baldio (2026-10-04)

O usuário esclareceu: em Curitiba é comum haver área de mata/vegetação
que **tem o mesmo valor comparativo que terreno baldio** pro objetivo do
TCC (ambos são "área não edificada"), então a rotulagem manual das 150
imagens **deliberadamente inclui mata dentro da classe `area_NãoEdificada`**
— não é só lote vago. Isso explica o nome da classe no `classes.txt`.

**Isso revisa o Achado 2 acima.** O `CLAUDE.md` do projeto já documenta
(linha de base estatística, §3.6) que **cobertura arbórea em Curitiba
correlaciona com renda** (bairro rico = mais verde). Como a classe
mistura baldio com mata, o aumento de `area_NãoEdificada` com a renda no
restante da cidade **pode ser (parte ou todo) um padrão real de mais
mata em bairro rico, não só viés de detector**. As duas explicações
(confusão real mata×renda, e o viés de detector achado separadamente no
Achado 1) não são mutuamente exclusivas e **não foram desembaraçadas**
com os dados atuais — registrar essa ambiguidade em qualquer relato do
TCC, não apresentar o Achado 2 como prova isolada de viés do detector.

## Treino do v4 XL (modelo em uso)

`scripts/treinar.py` com `yolo26x-obb.pt` (pesos pré-treinados oficiais) sobre
`dataset/data.yaml`. **Dataset: 348 imagens 512×512 (272 treino / 76 val)** —
as 150 do v1 mais as rotuladas depois; checagem de 2026-10-18: nenhum patch
repetido entre treino e val, 348/348 conferem com `Patches512`.

| parâmetro | valor |
|---|---|
| modelo base | `yolo26x-obb.pt` |
| imgsz | 1024 (upscale 2× do patch 512) |
| epochs / patience | 100 / 30 (rodou as 100) |
| batch | 4 (XL em 16 GB de VRAM) |
| optimizer | auto |
| workers | 0 (bug 1455 do Windows, ver v1) |
| amp / seed / deterministic | True / 0 / True |

Treino de ~75 min na RTX 5060 Ti. **Melhor época: 82** (`best.pt`):

| métrica | v4 XL | v1 nano |
|---|---|---|
| mAP50 | **0,736** | 0,718 |
| mAP50-95 | **0,482** | 0,437 |
| Precision | 0,775 | 0,897 |
| Recall | **0,726** | 0,600 |
| mAP50 `area_NãoEdificada` | 0,666 | 0,660 |

O XL troca precisão por recall em relação ao v1. Curvas, matriz de confusão e
`results.csv` em `runs/yolo26x_obb_v4_xl/`. `.gitignore` exclui `*.pt` (126 MB):
o peso vai como anexo de Release.

## Inferência em lote com o v4 XL (2026-10-13)

`scripts/inferir_lote.py` apontando para `runs/yolo26x_obb_v4_xl/weights/best.pt`
(yolo26x-obb), conf ≥0,25, batch 8, nos 45.523 patches de `TCC-data/Patches512`.
Teste com `--limite 200`: 100,8 ms/img com o carregamento do modelo, ~63 ms/img
em regime → rodada completa estimada em 50–75 min. Saída:
`runs/yolo26x_obb_v4_xl/deteccoes_lote.csv` — **48.345 detecções**, 28.702
patches com alguma (63,0%).

| classe | detecções | conf ≥0,5 | patches com ≥1 (conf ≥0,5) |
|---|---|---|---|
| area_NaoEdificada | 37.676 | 23.418 | 18.972 (41,7%) |
| faixa_pedestre | 6.825 | 5.585 | 3.018 (6,6%) |
| ponto_onibus | 3.844 | 2.595 | 2.303 (5,1%) |

**⚠️ Comparação com o v1 nano:** com o mesmo corte (≥0,5), `area_NãoEdificada`
aparecia em 8,3% dos patches; no XL, 41,7%. Não sabemos ainda se é mata real
(a classe inclui mata) ou o XL confiante demais (aí o corte de 0,5 deixa de
filtrar). Por isso a revisão de 400 abaixo. A validação por renda de
2026-10-04 foi feita com o v1 e **precisa ser refeita com o XL**.

Observação: os 200 primeiros patches (ordem alfabética) são todos de
`_sem_regional` — franja com muita vegetação, sem endereço no cadastro —,
então o teste de 200 não representa a cidade.

## Dataset objeto × patch × rua e revisão de 400 (2026-10-13)

**Patch → rua.** `TCC/rotular_tiles.py --zoom 19` rodado por pasta de regional
de `Patches512` e depois `TCC/rotular_quadras.py --zoom 19`, sem alterar os
scripts → `TCC-data/patches512_rotulos_quadra.csv` (45.523 patches; 3,8% sem
endereço, 95% desses em `_sem_regional`). Na parte resolvida "via quadra", a
rua confrontante fica a 114 m na mediana e a >150 m em 2.859 patches — por
isso a rua do patch não deve ser carimbada no objeto (CLAUDE.md §7-b).

**Contagem por patch.** `scripts/contar_objetos_por_patch.py` →
`TCC-data/patches512_rua_objetos.csv`: as colunas de rua do patch + contagem
de `ponto_onibus`, `faixa_pedestre`, `area_NaoEdificada` (conf ≥0,5 por
padrão, `--conf` para mudar).

**Uma linha por objeto.** `scripts/montar_dataset_objetos.py` →
`TCC-data/deteccoes_objetos_rua.csv` + `.geojson` (caixa como polígono, para
o QGIS). Valores do modelo sem filtro (48.345), com info do patch e da caixa
(centro e 4 cantos em lat/lon, largura/altura/área em m). Centro recalculado
confere com o lat/lon do modelo (erro máx. 2×10⁻⁷ graus).

NMS extra, opcional (`--nms-iou`, caixa girada via shapely, por patch e
classe, maior confiança fica) → `_nms.csv/.geojson`:

| critério | removidas | area_NaoEdificada | faixa | ônibus |
|---|---|---|---|---|
| IoU >0,7 | 0 | 0 | 0 | 0 |
| **IoU >0,5 (gerado)** | **610** | 568 | 33 | 9 |
| IoU >0,3 | 2.808 | 2.665 | 117 | 26 |
| contenção >0,7 (`--nms-contencao`, não aplicado) | 1.862 | 1.715 | 120 | 27 |

IoU 0,7 não remove nada porque o modelo já aplica NMS interno a 0,7. Depois do
NMS a 0,5 sobra um patch com 27 `area_NaoEdificada`
(`bairro_novo/19/190397/300630`) — provável caixa pequena dentro de grande,
caso que só a contenção pega.

**Revisão de 400.** `scripts/preparar_revisao.py` adaptado para o XL (exclui as
348 imagens do dataset; guarda as predições originais em
`<dest>/predicoes_modelo/`) e rodado com `--n 400 --dest revisao_400_xl`.
Seleção espalhada pela faixa de confiança (≥0,25) por classe; caixas: 396
`area_NãoEdificada`, 299 `faixa_pedestre`, 174 `ponto_onibus`. Mede precisão
por faixa de confiança; recall só dentro dessas imagens (todas têm detecção) e
não é amostra proporcional da cidade. Concluída — resultado na última seção.

## Notas

- `dataset/treinar.py` (dentro da pasta do dataset) é o script de exemplo
  que veio no zip do rotulador — mantido como referência histórica, mas o
  script real a usar é `scripts/treinar.py` (mais completo/parametrizado).
- O zip original do rotulador (`dataset_yolo.zip`) foi para `D:\\CityVision_lixeira` em 2026-10-18.

## Filtro de tamanho e corte de confiança (2026-10-18)

`scripts/filtro_tamanho.py` descarta caixa do modelo cujo lado menor ou maior
seja menor que o mínimo da classe nas caixas rotuladas do dataset (em px do
patch 512): `area_NaoEdificada` 21,2 × 25,6; `ponto_onibus` 6,5 × 15,0;
`faixa_pedestre` 5,0 × 18,6. Mínimos fixos no script (não mudam se o dataset
crescer). `--lote` grava `runs/yolo26x_obb_v4_xl/deteccoes_lote_filtrado.csv`.

`scripts/contar_objetos_por_patch.py` passou a usar **conf ≥0,25 sobre o lote
filtrado** (antes 0,5 sobre o bruto): a revisão mostrou precisão alta já em
0,25–0,5, e o corte 0,5 descartava ~28% dos pontos de ônibus reais.

## Resultado da revisão de 400 (concluída em 2026-10-18)

`scripts/estado_revisao.py` → `revisao_400_xl/estado_revisao.csv`.
`area_NãoEdificada` é avaliada como presença por patch (binária);
ônibus e faixa por contagem (acerto = min(modelo, revisão)). Caixas do modelo
filtradas por tamanho antes de contar; a revisão não é filtrada.

400/400 revisados, 54 alterados. Precisão por presença no patch, por faixa da
maior confiança da classe no patch:

| conf | área | ônibus | faixa |
|---|---|---|---|
| 0,25–0,4 | 0,96 (52) | 0,78 (37) | 0,96 (24) |
| 0,4–0,5 | 1,00 (32) | 0,73 (15) | 0,67 (12) |
| 0,5–0,7 | 1,00 (51) | 0,79 (28) | 0,76 (21) |
| 0,7–0,9 | 0,99 (80) | 0,92 (63) | 0,96 (73) |
| 0,9–1,0 | 1,00 (38) | 1,00 (13) | 0,95 (22) |
| **total** | **0,99** (251/254) | **0,86** | **0,92** |

Por contagem: ônibus 157 acertos / 26 FP / 4 FN; faixa 229 / 20 / 1; área
(presença) 251 / 3 / 0.

Leitura: os 41,7% de patches com área não edificada são a classe (mata +
baldio) presente de fato, não excesso de confiança. Limites: recall só dentro
dos 400 (todos têm alguma detecção); 346/400 sem mudança pode refletir
ancoragem no pré-rótulo — uma amostra aleatória revisada sem pré-rótulo mediria
isso. As 400 ainda não entraram no treino.
