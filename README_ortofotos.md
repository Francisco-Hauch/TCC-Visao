# Baixador de ortofotos IPPUC — TCC "Quadra a Quadra"

Script **`baixar_ortofotos_ippuc.py`** — baixa o cache de tiles das ortofotos do
GeoCuritiba (IPPUC) para disco. Estágio 1 do pipeline: montar a **base primária**
de imagens. Só usa a biblioteca padrão do Python (a costura opcional usa Pillow).

> **Por que roda na sua máquina e não na nuvem:** o ambiente de nuvem do Claude
> está atrás de um proxy que **bloqueia** `geocuritiba.ippuc.org.br`. Além disso o
> dataset (dezenas de GB) precisa ficar do lado da sua GPU. Então o download roda
> aqui no seu Windows, com internet plena.

## Como o dataset fica organizado

```
dataset/
  Ortofotos2019/
    21/                     # nível de zoom
      761538/               # coluna (x)
        1201824.jpg         # linha (y)  -> tile JPEG 256x256
      ...
  manifesto_Ortofotos2019_centro_z21.json   # bbox, faixa de tiles, contadores
  mosaico_Ortofotos2019_centro_z21.png      # (se usar --stitch) mosaico + .pgw + .prj
```

Cada tile é um recorte fixo do chão (EPSG:3857, Web Mercator). O mesmo `z/x/y`
cobre exatamente o mesmo terreno em qualquer ano — por isso dá pra comparar 2012
e 2019 sem trabalho de registro.

## Passo a passo (Windows)

Abra o **PowerShell** na pasta onde estão os arquivos.

**1. Confira o Python (precisa 3.8+):**
```powershell
python --version
```

**2. Veja o que seria baixado, SEM tocar na rede (recomendado começar aqui):**
```powershell
python baixar_ortofotos_ippuc.py --regiao centro --zoom 21 --dry-run
```

**3. Baixe o piloto de verdade e monte um mosaico pra olhar:**
```powershell
pip install Pillow            # só na 1a vez, e só pra --stitch
python baixar_ortofotos_ippuc.py --regiao centro --zoom 21 --stitch
python baixar_ortofotos_ippuc.py --regiao sitio_cercado --zoom 21 --stitch
```
Isso leva ~1 min cada. Abra `mosaico_Ortofotos2019_centro_z21.png` para conferir a
qualidade. O `.pgw` + `.prj` ao lado deixam o PNG **georreferenciado** — arraste pro
QGIS e ele cai no lugar certo do mapa.

**4. Só depois de validar o piloto, a cidade inteira:**
```powershell
python baixar_ortofotos_ippuc.py --regiao curitiba --zoom 21
```
Leva horas (~18 h no z21 para o bbox cheio; ~4,5 h no z20). Pode interromper com
`Ctrl+C` e rodar o **mesmo comando** de novo: o *resume* pula o que já baixou.

## Opções úteis

| Flag | Para quê |
|---|---|
| `--regiao {centro,sitio_cercado,curitiba}` | Área pré-definida. |
| `--bbox LAT_S LON_O LAT_N LON_L` | Um bounding box arbitrário (graus decimais). |
| `--acervo Ortofotos2012` | Outra época (2012 é o `t1` do painel). |
| `--zoom 20` | Metade da resolução, um quarto dos tiles. |
| `--threads 8` | Paralelismo (8 foi o medido como ideal). |
| `--stitch` | Monta o mosaico PNG georreferenciado. |
| `--grade N` | Corta o mosaico numa grade N×N (ex.: `--grade 2` = 4 partes). Requer `--stitch`. |
| `--dry-run` | Só calcula faixa/tamanho/tempo; não baixa. |
| `--insecure` | Ignora verificação de certificado TLS (se der erro de SSL). |
| `--saida PASTA` | Onde gravar (padrão `./dataset`). |

## Estimativas (medidas: ~38 tiles/s, ~11,8 KB/tile)

| Alvo | Tiles | Tamanho | Tempo |
|---|---|---|---|
| Piloto Centro z21 | ~2.346 | ~26 MB | ~1 min |
| Piloto Sítio Cercado z21 | ~2.208 | ~25 MB | ~1 min |
| Curitiba z20 | ~618 k | ~6,8 GB | ~4,5 h |
| Curitiba z21 | ~2,47 M | ~27 GB | ~18 h |

## Estágio 2a/2b — nome de rua por tile (feito)

Cada tile z20 cobre ~34,5 m de lado, ou seja, um pedaço de uma ou duas ruas.
Para saber **o que cada tile representa** sem 180 mil chamadas a um geocoder,
o cruzamento é feito offline contra os eixos de logradouro do próprio IPPUC.

```
# 2a) baixa os eixos uma vez (~42 mil trechos, 24 MB, ~25 s)
python baixar_eixos_logradouro.py

# 2b) rotula todos os tiles que estão no disco (~10 s para 180 mil tiles)
python rotular_tiles.py --formato ambos
```

Camada de origem: `GeoCuritiba/Publico_GeoCuritiba_MapaCadastral/MapServer/11`
("Trecho Logradouro"), que traz nome oficial, bairro, regional, CEP e
hierarquia viária de cada trecho.

Método: os eixos são projetados para coordenada de tile fracionária no mesmo
zoom, cada segmento é recortado (Liang-Barsky) contra o quadrado de cada tile
que ele cruza, e o comprimento recortado é acumulado por logradouro. A
`rua_principal` é a de **maior comprimento de eixo dentro do tile** — critério
de dominância, não de mera presença; a coluna `ruas` lista todas com os metros
de cada uma, o que identifica esquinas (`n_ruas >= 2`).

Tiles sem nenhum eixo dentro (miolo de quadra, parque, pátio) caem no fallback
`--raio-vizinho` (padrão 60 m), que atribui a rua mais próxima do centro do
tile e marca `origem = vizinho_NNm`. Quem ficar sem nada sai como `sem_rua`.

Saída em `D:\TCC-data\tiles_rotulos.csv` (`;` como separador, UTF-8 com BOM
para abrir direto no Excel) e `.jsonl`. Colunas: `z, x, y, arquivo,
lat_centro, lon_centro, rua_principal, rua_abrev, cobertura_m, n_ruas, ruas,
bairro, regional, cep, hierarquia_viaria, origem`.

Cobertura na base atual (180.319 tiles): 45,2% por intersecção, 33,1% por
proximidade, 21,8% sem rua. Validação: 10 tiles sorteados conferidos contra
uma consulta espacial no servidor do IPPUC — 10/10.

## Estágio 3 — quadra cadastral e rua confrontante (feito)

O critério do estágio 2b é estrito e correto, mas responde a pergunta errada
para o miolo de quadra: um tile z20 tem ~34,5 m de lado, **menor que a maioria
das quadras de Curitiba**. Tile que cai em fundo de lote, pátio de indústria ou
telhado de galpão não tem eixo dentro e sai `sem_rua` — mesmo estando numa
quadra perfeitamente endereçada, e mesmo quando a imagem mostra via asfaltada
(via interna de lote privado não é logradouro público e não existe no cadastro
do IPPUC).

A correção é trocar a pergunta: em vez de *"que rua passa dentro deste tile?"*,
perguntar *"a que quadra este tile pertence, e quais ruas confrontam essa
quadra?"*.

```
python baixar_eixos_logradouro.py --camada 17   # 16.606 quadras cadastrais
python rotular_quadras.py
```

O script localiza o centro do tile dentro de uma quadra (ray casting),
pré-computa as ruas que confrontam cada quadra (eixos a até `--confrontante`
metros da borda, padrão 30 m) e escolhe como `rua_quadra` a confrontante mais
próxima do **centro do tile** — num quarteirão comprido, o fundo do lote
pertence à rua que está do seu lado, não à primeira da lista da quadra.

O estágio 2b não é alterado: suas colunas mantêm o significado original. Este
script apenas acrescenta a `D:{BS}TCC-data{BS}tiles_rotulos_quadra.csv` as colunas
`quadra, rua_quadra, dist_rua_quadra_m, n_ruas_quadra, ruas_quadra, endereco,
endereco_origem`. A coluna `endereco` é a de conveniência: melhor endereço
disponível, com `endereco_origem` dizendo de onde veio (`eixo` > `vizinho` >
`quadra` > `indefinido`).

Resultado: 75,1% dos tiles caem dentro de uma quadra cadastral; **32.544 dos
39.220 `sem_rua` recuperados**; sobram 3,7% sem endereço. Validação: 11 tiles
conferidos contra a consulta espacial da camada 17 no servidor — 11/11 na
quadra correta.

Atenção ao usar: `dist_rua_quadra_m` costuma passar de 100 m em lote
industrial grande. A atribuição continua correta (o tile é daquela quadra), mas
filtre por essa distância se o seu uso exigir proximidade real da via.

## Achado — tiles em branco

797 tiles (0,4%) são imagem vazia, branco puro (desvio-padrão de pixel 0,00,
arquivo < 2 KB): borda do voo / fora da cobertura da ortofoto. Todos caem em
`sem_rua`, mas explicam só 2% dele. Vale excluir do dataset antes de treinar
qualquer coisa — o filtro por tamanho de arquivo < 2 KB os isola sem precisar
abrir a imagem.

## Próximo (Estágio 4, quando a base estiver no disco)

Gerar os **recortes 1024×1024 por quadra** a partir dos polígonos de quadra
(shapefile `ARRUAMENTO_QUADRAS_SIRGAS.zip` ou a API `Publico_GeoCuritiba_MapaCadastral`
para ter o identificador de cada quadra). Esse é o próximo script.

## Nota de uso

O IPPUC declara caráter "meramente informativo" dos arquivos, sem licença aberta
explícita nem termo proibindo coleta. Para uso acadêmico (TCC) está tranquilo; para
produto, peça autorização a `geoprocessamento@ippuc.org.br`. O script já manda um
`User-Agent` identificando uso acadêmico e respeita o serviço (8 threads, retry com
backoff, sem martelar tile ausente).
