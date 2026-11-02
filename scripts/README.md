# Scripts

Os scripts estão separados pelo papel no resultado final. Quase todos aceitam
`--help` para listar as opções.

## `essenciais/`: cadeia que produz o resultado

Em ordem de execução:

| # | Script | O que faz | Saída principal |
|---|---|---|---|
| 1 | `baixar_ortofotos_ippuc.py` | Baixa os tiles da ortofoto 2019 (z20), retomando de onde parou | `Ortofotos2019/` |
| 2 | `gerar_patches_quadra.py` | Cola blocos de 2×2 tiles z20 em patches 512×512 (z19) | `Patches512/` |
| 3 | `baixar_eixos_logradouro.py` | Baixa os eixos de rua e, com `--camada 17`, as quadras | `.geojson` |
| 4 | `rotular_tiles.py` | Identifica as ruas que cruzam cada patch | rótulos de rua |
| 5 | `rotular_quadras.py` | Dá rua aos patches de miolo de quadra | `patches512_rotulos_quadra.csv` |
| 6 | `treinar.py` | Treina o YOLO26-OBB | `best.pt` |
| 7 | `inferir_lote.py` | Roda o modelo nos 45.523 patches e georreferencia as caixas | `deteccoes_lote.csv` |
| 8 | `filtro_tamanho.py` | Descarta caixas menores que o mínimo da classe (`--lote`) | `deteccoes_lote_filtrado.csv` |
| 9 | `preparar_revisao.py` | Monta a amostra da revisão de 400 patches | pasta de revisão |
| 10 | `estado_revisao.py` | Compara modelo × revisão e calcula a precisão | `estado_revisao.csv` |
| 11 | `renda_por_patch.py` | Atribui a cada patch a renda do setor censitário 2022 | `patches512_renda.csv` |
| 12 | `contar_objetos_por_patch.py` | Conta os objetos de cada classe por patch (conf ≥ 0,25) | `patches512_rua_objetos.csv` |
| 13 | `montar_tabela_ruas.py` | Cruza características × Sigesguarda por (rua, bairro) | `ruas_caracteristicas_ocorrencias_*.csv` |
| 14 | `modelo_hipoteses_ruas.py` | Testa H1–H4 e o controle negativo, por volume e composição | `resultado_hipoteses_*.csv` |
| 15 | `robustez_r3.py` | Moran, composição, cortes de confiança, viés por renda e permutação | `resultado_r3_*.csv` |
| 16 | `todas_naturezas_r4.py` | Exploratório: área × 64 naturezas, com FDR | `resultado_r4_todas_naturezas.csv` |
| 17 | `onibus_area_r5.py` | Exploratório: ônibus + área no mesmo patch × roubo | `resultado_r5_onibus_area.csv` |

Períodos usados em `montar_tabela_ruas.py` e `modelo_hipoteses_ruas.py`:
`--anos 2009 2020` (principal) e `--anos 2009 2022` (sensibilidade). Sem esse
argumento, `montar_tabela_ruas.py` usa apenas 2019.

`rotular_tiles.py` e `rotular_quadras.py` rodam com `--zoom 19`, por pasta de
regional. Os caminhos de entrada e saída precisam ser informados, porque o padrão
desses dois scripts aponta para `D:\TCC-data`.

## `organizacao_verificacao/`: automação e conferência

Estes scripts não alteram o resultado. Parte deles importa funções de `essenciais/`.

| Script | O que faz |
|---|---|
| `rodar_regionais.ps1` | Roda o download das regionais em sequência, com até 3 passadas por regional |
| `verificar_integridade.py` | Reconta os tiles no disco e aponta o que falta |
| `classificar_regional.py` | Conta tiles e patches por regional oficial, sem mover nada |
| `reorganizar_por_regional.py` | Move tiles e patches para pastas por regional oficial |
| `retomar_treino.py` | Retoma um treino interrompido a partir do `last.pt` |
| `checar_vazamento.py` | Procura vazamento entre treino e validação |
| `conferir_dataset_patches.py` | Confere se o dataset usa os mesmos recortes de `Patches512` |
| `validar_modelo.py` | Revalida o `best.pt` e mostra as métricas por classe |
| `ler_best_pt.py` | Lê do checkpoint a versão, as métricas e a época |
| `comparar_inferencia.py` | Compara uma inferência nova com o lote original |
| `comparar_saidas.py` | Compara arquivos regerados com os originais |
| `conferir_artigo.py` | Confere os números do artigo contra as saídas dos scripts |
| `preparar_revisao_cega.py` | Sorteia patches sem pré-rótulo para medir recall e ancoragem |

## Como o endereçamento funciona

O endereçamento é geométrico e offline, sem geocoder. O caminho `z/x/y` de cada
imagem define exatamente o retângulo do chão que ela cobre.

1. **Rua que cruza o patch:** cada trecho de eixo é recortado (Liang-Barsky) contra
   o patch, e os metros de cada rua dentro dele são somados. Uma esquina conta para
   todas as ruas que a cruzam.
2. **Miolo de quadra:** patch sem rua cruzando é localizado na quadra cadastral e
   recebe a rua confrontante mais próxima do seu centro. Nesses casos a rua fica,
   na mediana, a 114 m.

Resultado: 45.523 patches, 3,8% sem endereço. A conferência contra o servidor do
IPPUC acertou 10 de 10 ruas e 11 de 11 quadras.

## Aquisição das imagens

- **Cobertura:** núcleo urbano de 8 regionais, 180.319 tiles z20 (~34,5 m de lado).
- **Patches:** um bloco de 2×2 tiles z20 cobre exatamente a área de um tile z19
  (`foto_pai = (x//2, y//2)`), com ~69 m de lado.
- **Tiles em branco:** 797 (0,4%) são borda de voo, identificáveis por terem menos de 2 KB.
- **Servidor:** cache XYZ público do GeoCuritiba, com download de uma regional por
  vez e nova tentativa em caso de erro. Repetir o comando retoma o que faltou.
