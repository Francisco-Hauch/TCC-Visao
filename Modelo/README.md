# Modelo — treino YOLO26-OBB (baldio / ponto de ônibus / faixa de pedestre)

Preparado em 2026-09-27. Ambiente pronto e hiperparâmetros já decididos
com o usuário (perguntados 1 a 1) — mas **treino ainda NÃO EXECUTADO**.
Rodar só quando o usuário pedir explicitamente. Ver `CLAUDE.md` na raiz do
projeto para o contexto geral do TCC.

## O que já está pronto

- **Dataset** (`dataset/`) — extraído de `TCC-data/dataset_yolo.zip`
  (gerado pelo Rotulador YOLO em 2026-09-27). Detecção OBB (caixa
  orientada), 3 classes:
  - `0 area_NãoEdificada` — **não é só terreno baldio, inclui também área
    de mata/vegetação** (decisão deliberada de rotulagem — ver seção
    "Classe area_NãoEdificada inclui mata" mais abaixo) — 73 instâncias
    treino / 17 val
  - `1 ponto_onibus` — 81 treino / 33 val
  - `2 faixa_pedestre` — 161 treino / 42 val
  - 117 imagens de treino (15 negativas, sem objeto) / 33 de validação (4
    negativas) — split 78%/22%, já vem pronto do rotulador.
  - `dataset/data.yaml` tem `path` absoluto (`D:/CityVision/Modelo/dataset`)
    + `train`/`val` relativos — passar este arquivo direto pro Ultralytics.
- **Pesos pré-treinados** (`weights/`) — `yolo26n-obb.pt` já baixado
  (checkpoint oficial Ultralytics, release v8.4.0). Variantes maiores
  (s/m/l/x-obb) baixam automaticamente na primeira vez que forem referenciadas
  pelo nome — não pré-baixadas ainda.
- **Ambiente Python** — sem venv dedicada: usa o Python 3.13 global da
  máquina, que já tinha `torch==2.14.0+cu130` funcionando com a RTX 5060 Ti
  (16 GB) e `ultralytics==8.4.150` (já traz suporte a YOLO26). Instalado
  agora: `onnx`, `onnxslim`, `onnxruntime-gpu` (providers disponíveis:
  TensorRT, CUDA, CPU) — ver `requirements.txt`.
- **Scripts** (`scripts/`):
  - `treinar.py` — treino, com um bloco `CONFIG` no topo listando cada
    hiperparâmetro a decidir (modelo/imgsz/epochs/batch/patience/
    optimizer/lr0); recusa executar (`SystemExit`) enquanto restar campo
    não definido.
  - `exportar_onnx.py` — export do `best.pt` do treino para ONNX, também
    com `CONFIG` próprio (imgsz/half/dynamic/opset/nms) a decidir depois
    que o treino terminar.
- **Saída de treino** (`runs/`) — vazia, é onde o Ultralytics vai gravar
  cada run (`runs/<name>/weights/best.pt` etc.).

## Hiperparâmetros decididos (2026-09-27)

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

Nada foi executado ainda — só configurado.

## Bug encontrado ao rodar (2026-09-27)

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

## Resultado do treino (2026-09-27)

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

## Inferência em lote na cidade toda (2026-09-27/2026-10-04)

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

## Validação por bairro/renda (2026-10-04)

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

## Notas

- `dataset/treinar.py` (dentro da pasta do dataset) é o script de exemplo
  que veio no zip do rotulador — mantido como referência histórica, mas o
  script real a usar é `scripts/treinar.py` (mais completo/parametrizado).
- Fonte do dataset (`TCC-data/dataset_yolo.zip`) não foi alterada.
