# TCC "Quadra a Quadra" — contexto do projeto

Este arquivo existe para que uma sessão local do Claude Code, aberta nesta
pasta (`D:\CityVision`), tenha contexto completo sem precisar perguntar do
zero. Escrito em 2026-09-13. Se algo aqui ficar desatualizado, atualize este
arquivo — ele é o ponto de entrada.

Autor/orientando: Francisco (fran.hauch@gmail.com). Há também um projeto
"TCC" no Claude com documentos mais longos (metodologia, artigos, notas) —
ver seção final.

> **Git / GitHub (2026-10-12).** Repositório: `https://github.com/Francisco-Hauch/TCC-Visao`
> (raiz = `D:\CityVision`). Todo commit tem autor **Francisco-Hauch**
> (`user.name` configurado só neste repositório). **Nunca** colocar
> `Co-Authored-By: Claude` nem "Generated with Claude Code" em commit ou
> PR — pedido explícito do autor. Trabalhar em branch separada, não direto
> na `main`.

> **Status em 2026-10-12 — mais recente que o de 2026-10-04 abaixo.**
> Modelo atual = **v4 XL** (`Modelo/runs/yolo26x_obb_v4_xl/weights/best.pt`,
> yolo26x-obb, dataset de 348 imgs dedup, batch 4): mAP50 geral 0,735,
> `area_NãoEdificada` 0,666. `scripts/inferir_lote.py` já aponta para ele
> (batch 8), mas a inferência em lote **ainda não rodou** com o XL. O
> diretório foi limpo para subir ao GitHub: execuções antigas (v1/v2/v3),
> pesos pré-treinados, `last.pt` e scripts de validação/experimento
> (SAM3, GroundedSAM2, Tile2Net, YOLO DOTA, ONNX, comparação v3×v4,
> validação por renda) foram movidos para `D:\CityVision_lixeira` (fora
> do repositório). Referências a esses arquivos no §3.10/§4 abaixo são
> históricas. `.gitignore` na raiz exclui `TCC-data/` e `*.pt`.

> **Status em 2026-10-04 — leia isto primeiro, antes do banner de
> 2026-09-13 abaixo (mantido como registro histórico, não mais o estado
> atual).** O primeiro ciclo completo de modelo já rodou, numa pasta nova
> `CityVision\Modelo\` que o banner de 2026-09-13 abaixo não conhece:
> dataset rotulado manualmente (150 imagens, 3 classes em caixa orientada
> OBB — `area_NãoEdificada`, `ponto_onibus`, `faixa_pedestre`; sobre a
> primeira classe incluir mata, ver nota logo mais abaixo), YOLO26n-obb
> treinado do zero até convergir (mAP50 geral 0,718 — `area_NãoEdificada`
> é a classe mais fraca), inferência em lote nos 45.523 patches
> (`Patches512`) da cidade toda, e uma primeira validação estratificada
> por renda. **Detalhe completo em §3.10 (nova).** Achado central dessa
> validação: o detector de `area_NãoEdificada` erra em direções opostas
> por faixa de renda — subdetecta em bairro pobre, sobredetecta em bairro
> rico — e a tendência de mais detecção em bairro rico no restante da
> cidade é ambígua entre viés de detector e cobertura arbórea real (ver
> §3.6 e a nota sobre mata abaixo). Export para ONNX tem um bug de
> acurácia não resolvido (Ultralytics, cabeçalho `OBB26`) — usar o `.pt`
> para qualquer inferência, não o `.onnx` (detalhe em §3.10). **Isso NÃO
> invalida o banner de 2026-09-13 abaixo** (ele ainda descreve o raciocínio
> original e as decisões travadas) — só significa que a "extração de
> imagem" que ele descreve como passo futuro já tem um primeiro resultado
> real, com limitações conhecidas, documentadas em §3.10.

> **Foco atual (corrigido em 2026-09-13) — leia isto primeiro.** O trabalho
> agora é **extração de features nas ortofotos de 2019 que já estão em
> disco** — nada de baixar 2012 nem montar análise temporal por enquanto,
> isso não é prioridade hoje. Três classes-alvo: **terreno baldio
> (rotulado na prática como "área não edificada" — inclui mata/vegetação,
> ver nota de 2026-10-04 logo abaixo do banner), ponto de ônibus, calçada**.
> Mas atenção à divisão de trabalho decidida em
> 2026-09-13, que muda tudo:
>
> - **Ponto de ônibus e calçada — decisão original abaixo, mas EM REVISÃO
>   desde mais tarde em 2026-09-13, ver bloco mais abaixo e §3.4.** A ideia
>   original era: já existem como vetor pronto no cadastro do IPPUC/URBS,
>   com mais precisão do que qualquer detector entregaria, então **puxar o
>   vetor direto** — nada de treinar/rodar modelo nelas. Isso foi revisto
>   no mesmo dia (ver decisão logo abaixo) — não assumir que ainda vale sem
>   checar a revisão.
> - **Terreno baldio é a única classe que vai por extração de imagem de
>   verdade.** É onde o esforço de modelo se concentra agora — 100% dele.
> - A **comparação com o dataset de ocorrências (Sigesguarda)** continua
>   sendo **só a etapa final** do TCC (decisão anterior, ainda de pé) — não
>   trabalhar nisso agora.
>
> Isso deixa o §5 abaixo bem mais enxuto do que a versão anterior deste
> arquivo (que incluía baixar 2012 e análise temporal como prioridade —
> **isso estava errado, foi revertido**).
>
> **Decisão travada em 2026-09-13 (2ª vez) — núcleo do projeto, não
> reabrir:** achamos vetores/cadastro fiscal que PARECEM substituir o
> trabalho de visão computacional para terreno baldio (ver §5, item 6 —
> campo `gtm_desc_natureza` = "Territorial" no Lote Cadastral). **Isso não
> muda o plano.** A extração por imagem acontece de qualquer forma — o cadastro
> fiscal/vetor não é confiável ou preciso o bastante sozinho (é rótulo
> administrativo, desatualiza, tem motivo cartorial que não bate com o que
> está no chão) para substituir a mensuração visual. O papel desses dados é
> **gabarito**: rótulo fraco para pré-rotular/validar/estratificar o
> modelo, nunca para pular a etapa de modelo. Não repetir a pergunta "dado
> que já temos o vetor, ainda precisamos do modelo?" — a resposta é sempre
> sim para terreno baldio.
>
> **Generalização EM REVISÃO (2026-09-13, mesmo dia, mais tarde) — esta
> parte NÃO está travada, ao contrário do parágrafo acima.** O usuário
> estendeu o princípio: **todas** as classes (não só baldio) passam por
> modelo — ponto de ônibus, calçada, rua etc. incluídos. Vetor é gabarito
> pra validar, nunca substituto, em qualquer classe. Ele foi explícito que
> pode mudar de ideia e que vai falar com o orientador sobre essa divisão
> de trabalho antes de fechar. **Enquanto não houver confirmação dele,
> tratar como hipótese de trabalho atual, não como decisão travada** — se
> ele voltar dizendo que o orientador prefere a divisão original (§3.4,
> vetor puro pra ponto de ônibus/calçada), isso NÃO é reabrir uma decisão
> travada, é a decisão pendente sendo fechada.

> **Classe "terreno baldio" inclui mata — decisão de rotulagem (registrada
> em 2026-10-04).** Em Curitiba é comum haver área de mata/vegetação que,
> para o objetivo do TCC, tem o mesmo valor comparativo que terreno baldio
> em si (ambos são "área não edificada" do ponto de vista do ambiente
> construído). Por isso a classe rotulada manualmente no dataset de
> treino (`CityVision/Modelo/dataset`, 150 imagens, YOLO26-OBB — ver
> `Modelo/README.md` para o pipeline de treino/inferência em lote, não
> descrito neste arquivo) se chama `area_NãoEdificada` e deliberadamente
> **mistura terreno baldio com área de mata** — não é só "lote vago".
> Consequência prática: qualquer métrica dessa classe (contagem, taxa por
> bairro, correlação com renda/ocorrência) está medindo os dois fenômenos
> juntos. Isso é diretamente relevante para o confundidor de cobertura
> arbórea × renda já citado no §3.6 — ver a ressalva lá. Não reabrir esta
> decisão de rotulagem sem reclassificar o dataset inteiro.

## 1. Objetivo final

Cruzar **ocorrências registradas pela Guarda Municipal de Curitiba
(Sigesguarda)** com **características do ambiente construído extraídas de
ortofoto e de cadastro municipal (IPPUC)**, na escala de quadra/trecho de
rua, para medir o quanto o ambiente físico (terreno baldio, iluminação,
vegetação, densidade construtiva) explica variação espacial de
crime/desordem em Curitiba — e, se possível, mostrar isso com um desenho
**temporal** (2012 → 2019), não só uma correlação estática.

Não é "treinar um detector e printar mAP". O detector é insumo. O produto é
a métrica por unidade espacial cruzada com o dado de crime, com desenho
estatístico que aguente a banca (confusão por renda, autocorrelação
espacial, MAUP, viés sistemático do detector por classe social do bairro).

Achado estrutural que orienta tudo: **metade do que pareceria "tarefa de
visão computacional" já está resolvida como vetor pronto no cadastro do
IPPUC** (poste, edificação, árvore isolada, passeio, meio-fio, parada de
ônibus — todos com API espacial, sem precisar detectar nada em imagem). O
que sobra genuinamente para modelo é **condição**, não inventário: terreno
baldio e seu estado, mato alto, entulho, pátio de veículos, telhado
degradado — coisas que ninguém cadastra porque mudam o tempo todo, e que
por isso mesmo servem à análise temporal.

## 2. As duas bases de dados

**Sigesguarda** (ocorrências da Guarda Municipal) — duas eras incompatíveis,
não uma série única: sistema antigo (2009–2022, 429.346 registros, `;`,
latin-1, logradouro sem tipo) e sistema novo (2022–2026, 223.071 registros,
`,`, UTF-8, logradouro com tipo). **Nenhuma das eras tem coordenada** — só
`bairro + logradouro`. Isso define o teto de granularidade do projeto:
dá pra chegar em rua dentro de bairro, não em lote. Recorte recomendado:
2009–2020, sistema antigo, arquivo de qualquer data entre 2023-01-01 e
2024-04-01 (congelado, todos idênticos).

**GeoCuritiba/IPPUC** — 8 acervos de imagem aérea/satélite co-registrados
(mesmo tile z/x/y cobre o mesmo chão em qualquer época), cache XYZ público
sem autenticação, mais serviços de vetor com Query liberado (sem token):
`Publico_GeoCuritiba_MapaCadastral` = quadra/lote/logradouro;
`Publico_Interno_GeoCuritiba_BaseCartografica_para_BC` = poste (camada 71),
edificação (72), árvore isolada (65), passeio (80), meio-fio (75), campo
quadra (14), piscina (16), maciço arbóreo 2019 (64), entre outras; e
`URBS_Transporte_Publico` = parada de ônibus (`MapServer/1`). Nome do
serviço de base cartográfica confirmado em 2026-09-13 (o nome sem
"_Interno_" usado em versões antigas deste arquivo dá 404/token). Há
também `MDS_2019` e `MDT_2019` como `ImageServer` (modelo de superfície e
de terreno, útil para altura acima do solo = MDS−MDT). `Ortofotos2012` é o
t₁ do painel, `Ortofotos2019` (+ NIR) é o t₂.

## 3. O que já está pronto

### 3.1 Ortofotos baixadas (Estágio 1)
180.319 tiles em `Ortofotos2019\20\<x>\<y>.jpg` (2,4 GB), zoom 20
(~0,135 m/px, 13,5 cm/px), cobrindo o núcleo urbano de 8 regionais (Matriz,
Boa Vista, Santa Felicidade, Portão, Boqueirão, Cajuru, CIC, Pinheirinho) —
franjas rurais e Bairro Novo/Tatuquara ficaram de fora de propósito. z=21
fica reservado para quando entrar detecção de ponto de ônibus/estação-tubo
(resolução extra ajuda em objeto pequeno). 797 tiles (0,4%) são imagem
branca pura (borda de voo) — descartar antes de treinar (arquivo < 2 KB).

### 3.2 Rótulo de rua por tile (Estágio 2)
Cada tile cruzado offline (sem geocoder, sem chamada de rede por foto)
contra os 42.236 trechos de logradouro do IPPUC: recorte geométrico
(Liang-Barsky) mede quantos metros de cada rua caem dentro do tile,
`rua_principal` = maior cobertura. 45,2% por interseção direta, 33,1% por
vizinhança até 60 m, 21,8% sem rua (miolo de quadra menor que o tile não
tem eixo perto). Validado 10/10 contra o servidor.

### 3.3 Quadra cadastral e rua confrontante (Estágio 3)
Resolve o miolo de quadra: localiza o centro do tile dentro de uma das
16.606 quadras cadastrais e escolhe a rua confrontante mais próxima do
centro do tile. Recuperou 32.544 dos 39.220 tiles sem rua → cobertura total
sobe para 75,1% dentro de quadra, restando 3,7% sem endereço algum.
Validado 11/11.

**Arquivo final:** `tiles_rotulos_quadra.csv` (73 MB, `;`, UTF-8 BOM) — uma
linha por foto, com coordenada, rua principal, bairro, CEP, hierarquia
viária, quadra, rua confrontante, distância até ela, e um campo `endereco`
já consolidado (melhor fonte disponível). Ver `README_ortofotos.md` e
`COMO_FUNCIONA.txt` nesta pasta para o dicionário completo de colunas.

### 3.4 Ponto de ônibus e calçada já são vetor pronto — não é tarefa de CV
> **EM REVISÃO (2026-09-13, mesmo dia, mais tarde) — não travada.** O
> usuário disse depois que TODAS as classes (inclusive ponto de ônibus e
> calçada) devem passar por modelo — vetor é só **gabarito** (referência
> pra validar/estratificar a saída do modelo), nunca um substituto que
> pula a etapa de CV, nem aqui nem em baldio (ver banner do topo do
> arquivo). Ele mesmo disse que pode mudar de ideia e que vai conversar
> com o orientador sobre isso — **não tratar como definitivo até ele
> confirmar**. Enquanto isso, a leitura original abaixo (puxar só vetor,
> sem modelo, pra estas duas classes) fica pausada, não descartada.

**Decisão de 2026-09-13 (versão original, ver revisão acima):** para estas
duas classes, puxar o cadastro em vez de detectar na imagem. Nenhuma das
duas entra no modelo de extração.

- **Ponto de ônibus** — `GeoCuritiba/URBS_Transporte_Publico/MapServer/1`,
  **7.252 pontos reais com coordenada E tipo** (chapéu-chinês 2.364, novo
  mobiliário 1.931, domus 735, sem demarcação 600, placa em cano 487,
  plataforma 339, estação-tubo 289, placa em poste 234, especial
  madrugueiro 185, linha turismo 25, outros 45). `MapServer/0` = terminais,
  `MapServer/2` = linhas/trajeto.
- **Calçada** — camada `Área de Passeio` (nº 80 do serviço
  `Publico_Interno_GeoCuritiba_BaseCartografica_para_BC`, ver §3 do
  relatório-mãe), **74.046 polígonos** com atributos `largura`, `calcada` e
  `pavimentacao`. Confirmado acessível em 2026-09-13 (74.047 dentro do bbox
  das ortofotos — bate com a contagem citada aqui).

**Atualização: ambas já foram baixadas** (antes de 2026-09-20, ver §3.9 —
`pontos_onibus.geojson` e `area_passeio.geojson` em
`TCC-data/VetoresFeatures/`) e cruzadas com a grade de tiles/quadra (§5,
itens 1/2, marcados como feitos). O padrão de download usado foi o mesmo
de `baixar_eixos_logradouro.py` (Query da API MapServer, paginado, saída
GeoJSON), só trocando de camada/serviço.

### 3.5 Ligação ocorrência ↔ geometria (fora desta pasta, análise já feita)
Casamento de `bairro + logradouro` do Sigesguarda contra os 42.234 trechos:
89,2% no sistema antigo (com normalização — remover tipo, expandir
abreviação: MAL→MARECHAL etc.), 91,6% no novo. O que sobra sem casar é
praça, via sem denominação cadastral (ex. "Linha Verde" = BR-116 urbana) e
equipamento urbano nomeado — todos com correção conhecida, ~95% alcançável.
**Trecho é a unidade estatística confiável**; quadra é agregação derivada
com peso proporcional à testada, não o contrário.

### 3.6 Linha de base estatística (sem visão computacional)
Cruzamento Spearman de 9 características do cadastro contra taxa de
ocorrência por bairro (n=73). Achados: a maioria das naturezas de ocorrência
correlaciona 0,71–0,89 com o volume total (estão medindo alocação de
efetivo/propensão a denunciar, não risco); **Invasão** é o desfecho mais
limpo (ρ baixo com volume total); usar **proporção** em vez de taxa (ex.
"share de Invasão" em vez de "Invasões por km") limpa boa parte do viés —
share de Invasão × ocupação irregular chega a ρ=+0,51; cobertura arbórea ×
Roubo/Furto é a associação mais robusta (ρ≈ -0,44/-0,48), mas confunde com
renda (verde em Curitiba também é bairro rico). **(2026-10-04: esse
confundidor deixou de ser só teórico — a classe de "terreno baldio" do
modelo de CV inclui mata/vegetação, ver nota no topo do arquivo. Qualquer
correlação entre detecção de baldio e renda pode estar medindo este mesmo
efeito de cobertura arbórea, não (só) terreno vago nem viés de
detector.)** Teto de efeito é modesto
(|ρ|≤0,51) — é a régua que o detector precisa superar para se justificar.

### 3.7 Desenho do estudo (o que salva o TCC de virar correlação fraca)
Painel temporal 2012→2019 com efeito fixo de unidade (absorve renda,
cultura de denúncia — tudo que é constante no lugar) e efeito fixo de
época. Problema real: no nível de trecho de rua a contagem é esparsa (rua
mediana = 1 ocorrência/ano; só 75 ruas têm Invasão nos três anos
2010/2012/2019). Três correções, em ordem de retorno: (1) agrupar anos em
janelas em torno de cada voo (2009–2011/2012–2013/2018–2020) — ruas com
dado suficiente sobem de 184 para 695; (2) usar uma cesta "desordem e
zeladoria" (Dano+Invasão+Perturbação+Depósito irregular+Crime
ambiental+Furto+Substância ilícita) em vez de uma natureza isolada; (3) se
ainda faltar contagem, subir a unidade para os 2.395 setores censitários de
2010 (casa com a época da imagem de 2010) antes de ir para bairro. Tratar
2010→2012 como teste de tendência prévia (paralela) e 2012→2019 como a
diferença que importa — evita comparar imagem 2010 (satélite off-nadir) com
ortofoto.

### 3.8 Modelos avaliados e orçamento de GPU (RTX 5060 Ti 16 GB)
**Atualização 2026-10-04: o caminho escolhido na prática, pra `terreno
baldio`/`area_NãoEdificada`, foi nenhum dos dois abaixo — foi rotulagem
manual (150 imagens) + fine-tuning supervisionado do YOLO26n-obb, feito e
concluído, ver §3.10.** SAM 3 zero-shot (§5, item antigo) não chegou a ser
tentado porque a rotulagem manual + treino supervisionado já couberam no
tempo disponível. O texto abaixo é o levantamento original de opções
(2026-09-13), mantido como referência caso valha complementar depois.

Não é preciso treinar do zero. Pré-treinados aplicáveis, por tarefa:
**SAM 3** (zero-shot por prompt de texto — "baldio", "mato", "telhado") para
segmentação de condição; **YOLO26/YOLO11-obb** com pesos DOTA-v1 (caixa
orientada) para veículo, quadra esportiva, piscina; **DINOv3** (backbone
congelado + linear probe) para embedding por patch, treina classificador
com poucas centenas de exemplos; SAM 3 também serve como pré-rotulador
(humano só corrige, reduz custo de anotação em uma ordem de grandeza). Uma
regra geométrica de graça, antes de qualquer inferência: lote sem polígono
de edificação sobreposto + NDVI alto e uniforme + altura quase nula
(MDS−MDT) já é candidato forte a baldio — reserva o modelo caro só para o
que essa regra deixar ambíguo. Estimativa (derivada de benchmark público,
não medida na 5060 Ti local): YOLO-obb e DINOv3 nas duas épocas custam
poucas horas de GPU somadas; SAM 3 pesa mais (ordem de 13h/prompt em
TensorRT) — reservar para os casos ambíguos. **O gargalo real do projeto é
rede, não GPU**: baixar a cidade inteira já consumiu mais tempo que rodar
os modelos vai consumir.

### 3.9 Bases complementares baixadas em 2026-09-20 (pós-conversa com orientador)

Depois de checar com o orientador, foi pedido um levantamento de tudo que dá
para extrair sem CV (calçada, baldio, prédio, casa, indústria, poste,
semáforo) e se dava para ter isso de forma temporal. Achado central: **nenhuma
camada do GeoCuritiba nem do OSM tem versão histórica exposta por API** — é
sempre retrato atual. A única exceção parcial (`created_date` do Poste) foi
testada e não presta (ver abaixo). Isso significa que o painel temporal do
§3.7 continua dependendo de imagem (Ortofotos2012/2019 + CV), não de vetor.

Tudo em `TCC-data\VetoresFeatures\` — a pedido do usuário, `area_passeio.
geojson`, `pontos_onibus.geojson` e `lotes_cadastrais.geojson` (calçada,
ponto de ônibus, lote fiscal) foram MOVIDOS pra cá também (não baixados de
novo), pra juntar tudo que é vetor de feature num lugar só. `quadras_
cadastrais.geojson` e `eixos_logradouro.geojson` ficaram na raiz de
`TCC-data` — são usados por `rotular_quadras.py`/`rotular_tiles.py` com
caminho fixo, não são "feature" desta lista. Os scripts que liam
`pontos_onibus.geojson`/`area_passeio.geojson` na raiz
(`cruzar_vetores_quadra.py`, `features_vetor_por_tile.py`) e o que os gera
(`baixar_vetores_prontos.py`) já foram atualizados para o caminho novo.
Bases novas baixadas por `TCC\baixar_bases_features.py` (só biblioteca
padrão, nada para instalar):

- **`edificacoes.geojson`** — camada 72 "Edificação" do GeoCuritiba, 1.201.673
  polígonos, 970 MB. **ACHADO IMPORTANTE (validado por amostragem no arquivo
  inteiro, não é suposição pelo schema): os campos semânticos estão
  essencialmente vazios** — `finalidade` nulo em 99,7% (resto é código
  9999="a ser preenchido"), `situacaofisica` e `matconstr` são 9999 em
  99,9995%, `numeropavimentos`/`classeativecon`/`divisaoativecon`/
  `administracao`/`bairro`/`cep`/`logradouro` nulos em praticamente 100% dos
  registros. **Ou seja: essa camada só serve como geometria de contorno de
  edificação (footprint), não dá para classificar casa/prédio/indústria por
  atributo — isso teria que vir de CV sobre a ortofoto, ou de outra fonte.**
  Segue útil para: existência/contagem/área de edificação por lote e quadra
  (input direto da regra geométrica de baldio do item 6 do §5 — lote sem
  polígono de Edificação sobreposto).
- **`postes.geojson`** — camada 71 "Poste", 179.401 pontos, 54,5 MB.
  `tipoposte` está bem preenchido de verdade (154.850 Iluminação, 24.550 Rede
  elétrica, 1 Ornamental) — utilizável. `created_date`/`last_edited_date`
  **testados e descartados como sinal temporal**: só 22 valores distintos em
  179 mil registros — é data de carga em lote no GIS, não data de instalação
  de poste individual.
- **`setor_censitario_2010.geojson`** (2.395 setores, campo `renda`, 0 nulos)
  e **`setor_censitario_2022.geojson`** (3.190 setores, campo `renda_media`,
  22 nulos) — `Publico_Interno_GeoCuritiba_BaseTematica`/10 e /45. Dados reais
  e bem preenchidos (verificado). Não são feature do ambiente construído, são
  para controlar renda por época — resolve de graça o item 7-f, em duas datas
  que cobrem o período do Sigesguarda.
- **`semaforos_osm.geojson`** (1.860 nós) e **`faixa_pedestre_osm.geojson`**
  (5.094 elementos) — OpenStreetMap via Overpass API, não GeoCuritiba (**não
  existe camada de semáforo em nenhum dos 3 serviços públicos do GeoCuritiba**,
  vasculhado em 2026-09-20). Cobertura colaborativa/irregular (pior em
  periferia, mesmo viés do item 7-a mas na fonte, não no detector) — tratar
  como complemento, não como inventário completo. `faixa_pedestre_osm` é
  bônus: o GeoCuritiba só tem travessia em desnível (camada 20), não faixa
  pintada no chão.

**Próxima ação sugerida, ainda não feita:** re-baixar `Lote Cadastral`
(camada 15) incluindo os campos de zoneamento (`gtm_desc_grupo_zoneamento`,
`gtm_sigla_zoneamento` — existem no schema, não foram pedidos no download
original de `lotes_cadastrais.geojson`). Zoneamento é classificação legal por
lote, mais provável de estar preenchido de verdade que os campos de
Edificação — é a aposta mais forte agora para separar residencial/industrial/
comercial sem CV.

### 3.10 Dataset rotulado, treino YOLO26-OBB e inferência em lote (2026-09-27 a 2026-10-04)

Pasta nova, fora da árvore `TCC`/`TCC-data` descrita no §4 antigo:
`D:\CityVision\Modelo\` — ver `Modelo/README.md` para o detalhe completo
(tabela de hiperparâmetros, logs de erro, todos os números). Resumo:

**Dataset.** `TCC-data/dataset_yolo.zip` (gerado por um "Rotulador YOLO"
em página HTML, `TCC-data/rotulador-yolo.html`) — 150 imagens 512×512
(recortes de `Patches512`, ver abaixo), rótulos em caixa orientada (OBB),
split 78/22 (117 treino/33 val) já pronto, 3 classes: `area_NãoEdificada`
(**inclui mata, não é só lote vago — ver nota no topo do arquivo**),
`ponto_onibus`, `faixa_pedestre`. Extraído para `Modelo/dataset/`.

**Patches512.** `TCC-data/Patches512/<regional>/19/<x>/<y>.jpg` — 45.523
recortes 512×512 (o "Estágio 4" do §5 antigo, já executado: bloco 2×2 de
tiles z20 = 1 tile z19, mesma convenção de `foto_pai = (x//2, y//2)`).
As 150 imagens do dataset rotulado são uma amostra destes mesmos patches.

**Treino.** YOLO26n-obb (nano — dataset pequeno), imgsz 1024, epochs 100
(parou em 90 por early stopping, patience 30), batch 16, optimizer auto
(AdamW), ambiente sem venv (Python global já tinha
`torch==2.14.0+cu130` + `ultralytics`). Dois crashes corrigidos no
caminho: (1) `workers>0` no DataLoader dá erro Windows 1455
(`ERROR_COMMITMENT_LIMIT`, bug conhecido do multiprocessing do PyTorch no
Windows) — corrigido com `workers=0`; (2) RAM do sistema esgotada por um
programa externo (jogo) rodando em paralelo — não é bug do pipeline,
resolvido fechando o programa e retomando com `model.train(resume=True)`
a partir do último checkpoint. **Resultado final** (`best.pt`, checkpoint
do epoch 60):

| classe | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| geral | 0,897 | 0,600 | 0,718 | 0,437 |
| area_NãoEdificada | 1,000 | 0,469 | 0,660 | 0,237 |
| ponto_onibus | 0,789 | 0,545 | 0,614 | 0,423 |
| faixa_pedestre | 0,901 | 0,786 | 0,880 | 0,650 |

`area_NãoEdificada` é a classe mais fraca (recall baixo, caixa imprecisa)
— esperado com só 73 instâncias de treino.

**Bug de export ONNX (não resolvido).** `best.onnx` exporta sem erro mas
tem mAP50 bem mais baixo que o `.pt` (0,52 vs 0,72) — geometria das caixas
bate, mas a confiança sai sistematicamente mais baixa e near-duplicatas
não são suprimidas. Não é o `onnxslim`, não é versão do Ultralytics
(testado 8.4.150 e 8.4.163, idêntico) — aparenta ser bug real do
Ultralytics no cabeçalho novo `OBB26` (end2end/NMS-free). **Usar o `.pt`
para qualquer inferência até isso ser corrigido upstream.**

**Inferência em lote.** `best.pt` rodado nos 45.523 patches
(`scripts/inferir_lote.py`, georreferencia cada detecção via `tile2deg()`
— mesma função de `rotular_tiles.py`, ver armadilha §7-b). 48.575
detecções brutas; `area_NãoEdificada` sozinha tinha 33.901 (42% dos
patches, confiança média 0,38 — implausível, muito falso positivo em
confiança baixa). **Filtrado para confiança≥0,5**: 12.313 detecções
(`area_NãoEdificada` 4.129 = 8,3% dos patches — bem mais plausível,
`ponto_onibus` 3.330, `faixa_pedestre` 4.854).

**Validação por bairro/renda.** Cruzamento das 150 imagens rotuladas com
bairro (`tiles_rotulos_quadra.csv`, §3.2/3.3) e renda (censo 2010/2022,
§3.9) via `scripts/validar_confiabilidade_renda.py`. Comparando rótulo
manual × predição do modelo nas mesmas 150 imagens: confiabilidade de
`area_NãoEdificada` **muda de direção por renda** — em bairro de renda
baixa o modelo subdetecta (12 falso-negativo vs 10 acerto), em bairro de
renda alta sobredetecta (8 falso-positivo vs 10 acerto); `ponto_onibus` e
`faixa_pedestre` ficam estáveis em todas as faixas. Isso é a confirmação
empírica da armadilha §7-a para esta classe especificamente. No restante
da cidade, densidade de `area_NãoEdificada` também sobe com a renda
(6,8→9,7 por 100 patches) — **mas essa parte é ambígua**: como a classe
inclui mata (nota no topo do arquivo) e cobertura arbórea já correlaciona
com renda (§3.6), não dá pra separar quanto desse aumento é viés de
detector e quanto é mata real concentrada em bairro rico, com os dados
atuais.

**Pendente, não feito ainda:** cruzar as detecções de `area_NãoEdificada`
com o cadastro fiscal (`lotes_cadastrais.geojson`, `gtm_desc_natureza` =
"Territorial" — §5 antigo, item 6) como segunda validação independente;
separar mata de baldio de verdade (exigiria rotular uma 4ª classe ou
atributo); investigar o bug de export ONNX a fundo (comparar operação a
operação dentro do grafo); mais dados rotulados de `area_NãoEdificada`
especificamente (é a classe com menos instâncias e peor desempenho).

## 4. Estrutura de arquivos

```
D:\CityVision\
  TCC\                              # programas
    baixar_ortofotos_ippuc.py       # Estágio 1 — baixa tiles
    rodar_regionais.ps1             # roda o de cima nas 8 regionais
    verificar_integridade.py        # confere tiles baixados
    baixar_eixos_logradouro.py      # baixa ruas (padrão) ou quadras (--camada 17)
    rotular_tiles.py                # Estágio 2 — rua por tile
    rotular_quadras.py              # Estágio 3 — quadra + rua confrontante
    baixar_bases_features.py        # 2026-09-20 — edificação/poste/censo/OSM (§3.9)
    README_ortofotos.md             # doc técnica
    COMO_FUNCIONA.txt               # doc em linguagem simples
    COMANDOS_regionais.txt          # comandos prontos por regional
  TCC-data\                         # dados (não sobem para o projeto Claude)
    Ortofotos2019\20\<x>\<y>.jpg    # 180.319 tiles, 2,4 GB
    eixos_logradouro.geojson        # 42.236 trechos, 24 MB
    quadras_cadastrais.geojson      # 16.606 quadras, 30 MB
    tiles_rotulos.csv / .jsonl      # saída do Estágio 2
    tiles_rotulos_quadra.csv        # ARQUIVO FINAL — Estágio 2+3 combinados
    VetoresFeatures\                # TODAS as bases de vetor das features — ver §3.9
      edificacoes.geojson             # camada 72, só geometria (ver achado no §3.9)
      postes.geojson                  # camada 71
      lotes_cadastrais.geojson        # MOVIDO p/ cá 2026-09-20 (gtm_desc_natureza = baldio)
      area_passeio.geojson            # MOVIDO p/ cá — calçada, 74.046 polígonos
      pontos_onibus.geojson           # MOVIDO p/ cá — 7.252 pontos
      setor_censitario_2010/2022.geojson  # renda por setor (controle, não é feature)
      semaforos_osm.geojson, faixa_pedestre_osm.geojson  # OSM, não GeoCuritiba
    Logs\*.txt, Manifesto\*.json    # log e manifesto por regional/download
    sonda_vazao_ippuc.py, baixar_recorte.py, simular_z20.py, analisar_log.py
    Patches512\<regional>\19\<x>\<y>.jpg  # 45.523 recortes 512x512 — ver §3.10
    dataset_yolo.zip                # 150 imgs rotuladas (OBB) — ver §3.10
    rotulador-yolo.html             # ferramenta de rotulagem manual (gerou o zip acima)
  Modelo\                           # 2026-09-27+ — treino/inferência YOLO26-OBB, ver §3.10
    dataset\                        # dataset_yolo.zip extraído (images/labels train+val, data.yaml)
    weights\yolo26n-obb.pt          # peso pré-treinado baixado
    scripts\treinar.py              # treino (CONFIG com hiperparâmetros já decididos)
    scripts\retomar_treino.py       # resume de checkpoint (usado pós-crash)
    scripts\exportar_onnx.py        # export p/ onnx (tem bug conhecido, ver §3.10)
    scripts\inferir_lote.py         # inferência em lote + georreferência (tile2deg)
    scripts\validar_onnx.py         # comparação .pt x .onnx (achou o bug)
    scripts\validar_confiabilidade_renda.py  # validação por bairro/renda (§3.10)
    runs\yolo26_obb_baldio_onibus_faixa\     # best.pt/best.onnx, deteccoes_lote*.csv
    README.md                       # detalhe completo do treino/export/inferência
  CLAUDE.md                         # este arquivo
```

## 5. O que falta fazer (atualizado em 2026-10-04 — status real de cada item)

**Foco atual: refinar/validar o modelo de `area_NãoEdificada` (item 6) e
decidir se separa mata de baldio. Nada de 2012, nada de análise temporal,
nada de Sigesguarda por enquanto — essa última é a etapa final do TCC
(ver banner no topo do arquivo).**

1. ~~Baixar os dois vetores prontos (ponto de ônibus/calçada)~~ —
   **FEITO** antes de 2026-09-20 (`pontos_onibus.geojson`,
   `area_passeio.geojson` em `VetoresFeatures/`, ver §3.9).
2. ~~Cruzar os dois vetores com a grade de tiles/quadra~~ — **FEITO**
   (`cruzar_vetores_quadra.py`/`features_vetor_por_tile.py`, ver §3.9).
3. ~~Estágio 4 — recortes por quadra (512×512, bloco 2×2 de tiles z20)~~ —
   **FEITO**: `TCC-data/Patches512/`, 45.523 patches (ver §3.10).
4. **Checar se dá para baixar MDS/MDT e a banda NIR de 2019** — status
   não confirmado em nenhuma sessão registrada neste arquivo; a regra
   geométrica sem modelo (lote sem edificação + NDVI alto + altura nula)
   não foi tentada. Ainda pendente, verificar antes de assumir que não
   dá.
5. ~~SAM 3 zero-shot para terreno baldio~~ — **SUPERSEDIDO**: em vez de
   zero-shot, foi direto para rotulagem manual (150 imagens) + fine-tuning
   supervisionado do YOLO26n-obb (ver §3.10). SAM 3 fica como opção em
   aberto se precisar de mais dados pseudo-rotulados depois.
6. **Refino e validação do modelo de `area_NãoEdificada`** —
   **PARCIALMENTE FEITO**. O que já rodou (§3.10): treino supervisionado
   concluído (mAP50 0,718 geral, 0,660 na classe); inferência em lote na
   cidade toda, filtrada por confiança; validação estratificada por
   bairro/renda via censo (achou o viés por renda previsto no §7-a). **O
   que ainda falta deste item:** (a) cruzar com o cadastro fiscal
   (`lotes_cadastrais.geojson`, `gtm_desc_natureza`="Territorial", 28.834
   lotes — ver detalhe original abaixo) como segunda validação
   independente, ainda não feito; (b) decidir se separa mata de baldio
   (ver nota no topo do arquivo) — exigiria rotular uma 4ª classe ou
   atributo extra, ainda não feito; (c) mais dados rotulados de
   `area_NãoEdificada` especificamente, já que é a classe com menos
   instâncias (73 no treino) e pior desempenho; (d) resolver ou contornar
   o bug de export ONNX antes de qualquer deploy fora do Python/Ultralytics
   (§3.10) — usar o `.pt` até lá.

   Detalhe original do cadastro fiscal (item "a" acima, ainda válido):
   camada `Lote Cadastral`, nº 15 do serviço
   `Publico_GeoCuritiba_MapaCadastral`, campo `gtm_desc_natureza` =
   "Territorial" (tributado só pelo terreno) vs "Predial". 28.834 lotes
   "Territorial" dentro do bbox das ortofotos, de ~290 mil lotes. Baixado em
   `lotes_cadastrais.geojson` via `baixar_eixos_logradouro.py --camada 15`.
   É pseudo-rótulo FISCAL, não visual — lote pequeno em bairro denso também
   pode sair "Territorial" por motivo cartorial, então ainda serve só como
   pseudo-rótulo fraco a validar contra a imagem, não como verdade
   definitiva.
7. **(Etapa final do TCC, sem prazo definido ainda, NÃO INICIADO)** Fechar
   a junção ocorrência↔geometria do Sigesguarda (~95%, ver §3.5) e cruzar
   a taxa de `area_NãoEdificada` por quadra/setor com o painel de
   ocorrências. Não mexer nisso antes do item 6 estar validado (ainda não
   está — ver (a)/(b) acima).

**Fora do escopo por enquanto, não iniciar:** baixar `Ortofotos2012`,
qualquer análise temporal/diferença entre épocas.

**Nota de consistência (2026-10-04):** o dataset rotulado e o treino do
§3.10 **já incluíram `ponto_onibus` e `faixa_pedestre` como classes do
modelo**, não só `area_NãoEdificada` — ou seja, na prática, a hipótese
"generalização em revisão" do banner do topo (todas as classes passam
por modelo, vetor só como gabarito) foi a que acabou sendo seguida, não a
decisão original do §3.4 (vetor puro pra essas duas). Isso **não fecha**
a decisão pendente — o usuário ainda não confirmou com o orientador —
mas é o que de fato rodou até aqui. Se a conversa com o orientador
reverter pra decisão original, as duas classes já treinadas no YOLO26
ficam como excedente (não é retrabalho perdido, só não é mais o caminho
único: o cadastro de `ponto_onibus`/calçada do §3.4 continua disponível e
mais preciso pra uso direto).

## 6. Decisões travadas — não reabrir

1. Zoom padrão de download = **20** (~0,135 m/px). z=21 só para ponto de
   ônibus/estação-tubo.
2. Aquisição é **tile a tile**. `export`/`exportTiles` testados,
   indisponíveis, não existe atalho.
3. Download roda **nesta máquina Windows** (`D:\CityVision`), nunca na nuvem
   do Claude — proxy da nuvem bloqueia `geocuritiba.ippuc.org.br` (403) e o
   dataset precisa ficar do lado da GPU local.
4. Cobertura = **núcleo urbano de 8 regionais**, não cidade inteira nem
   limites administrativos oficiais.
5. Rótulo de rua/quadra é **geometria offline contra o cadastro do IPPUC**,
   nunca geocoder por chamada de rede — feito e validado (10/10, 11/11).
6. Recorte temporal do Sigesguarda: **2009–2020, sistema antigo**. 2026 fica
   de fora da análise temporal principal (ruptura de sistema, não deriva).

## 7. Armadilhas e avisos importantes

**(a) Erro sistemático do detector vira viés, não ruído.** Se o modelo
perde mais casas/baldio em bairro pobre, a correlação final com crime é
artefato do detector, não achado. Validar sempre estratificado por
faixa de renda/região, não só no agregado — é metade da defesa do TCC.
**Confirmado empiricamente em 2026-10-04 (§3.10) para `area_NãoEdificada`:
o detector subdetecta em bairro de renda baixa e sobredetecta em bairro
de renda alta** — não é mais só um risco teórico, é um viés medido. Isso
reforça que qualquer número agregado dessa classe precisa vir acompanhado
da quebra por renda, sempre.

**(b) Nunca carimbar uma feature detectada com a rua do tile onde ela
apareceu.** Isso erra sistematicamente em esquina (2–3 ruas no mesmo tile)
e em miolo de quadra (rua pode estar a 100+ m). Converter a posição em
pixel do objeto detectado para lat/lon (mesma conta usada no Estágio 2, ao
contrário) e só então buscar a rua mais próxima daquele ponto específico.
Funções prontas: `tile2deg()` e `dist_ponto_segmento()` em `rotular_tiles.py`.

**(c) Cuidado com contagem dupla** se houver sobreposição entre recortes —
dois pontos detectados a menos de 5 m um do outro são o mesmo objeto.

**(d) Dado da PM/Guarda é notificação, não crime.** Reflete onde se
policia e onde se denuncia; subnotificação é maior em área periférica —
enviesa na mesma direção do erro do detector (item a). Declarar isso no
texto do TCC.

**(e) MAUP e autocorrelação espacial.** Rodar em pelo menos duas escalas
espaciais e reportar o I de Moran dos resíduos; regressão comum sem isso
superestima significância.

**(f) Renda é o confundidor gigante** (baldio↔renda↔crime) — sem controlar
por renda (Censo por setor censitário) o achado pode ser só "medi renda com
passos extras".

**(g) Deriva temporal na imagem.** Estação e ângulo solar mudam entre
voos — normalizar por histograma. Só 2019 tem infravermelho (NDVI não
comparável entre épocas). Quadra não é imutável — fixar a malha cadastral
em uma época e tratar as demais como recorte sobre ela. Rodar o mesmo
detector nas duas épocas, nunca fine-tune por ano (senão "mudança" medida
é mudança de modelo, não de cidade).

## 8. Onde estão os documentos completos

Há um projeto "TCC" no Claude (claude.ai) com os documentos de origem,
mais longos que este resumo — abrir lá se precisar do detalhe completo ou
da lista de artigos citados:

- `plano-aquisicao-dataset.md` — histórico completo do download (este
  CLAUDE.md resume a versão mais atual dele).
- `estrategia-extracao-features.md` — por que "extrair features" virou
  "integrar cadastro + validar com imagem", e a ordem de ataque das 4
  classes (casa, baldio/mata, ponto de ônibus, poste).
- `nota-supervisao-fraca-pontos.md` — como usar o mapa de pontos de ônibus
  como supervisão fraca (Pointly Supervised Object Detection) em vez de
  entrada de inferência, e as 7 armadilhas a medir antes de treinar.
- `artigos-extracao-features-urbanas.md` — literatura anotada (segmentação
  de terreno vago urbano, detecção de objeto minúsculo/orientado, SAM
  urbano, etc.), com abstracts e por que cada um interessa.
- `quadra-a-quadra.html` — o relatório-mãe do projeto (8 seções): as duas
  eras do Sigesguarda, os 8 acervos do IPPUC, o inventário já resolvido por
  vetor, a costura ocorrência↔geometria, a linha de base estatística, o
  desenho temporal, os modelos e custo de GPU, e "onde eu começaria". As
  seções 3.6–3.8 e 5–7 deste CLAUDE.md são resumo direto desse relatório.
