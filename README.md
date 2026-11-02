# Quadra a Quadra

Trabalho de conclusão de curso que cruza **ocorrências da Guarda Municipal de
Curitiba (Sigesguarda)** com **características do ambiente construído extraídas
da ortofoto de 2019 por visão computacional**, na escala de rua. A pergunta é o
quanto o ambiente físico, em especial a área não edificada, se associa à variação
espacial das ocorrências.

## Como funciona

```
Ortofotos 2019 → patches 512×512 → detector YOLO26-OBB → contagem por patch
                       │                                       │
                eixos de rua e quadras                         ▼
                       └───────────────► tabela (rua, bairro) × Sigesguarda → modelos
```

O detector extrai três classes: `area_NãoEdificada` (mata + baldio),
`ponto_onibus` e `faixa_pedestre`. A unidade de análise é o par (rua, bairro),
porque o Sigesguarda não tem coordenada, apenas bairro e logradouro.

## Resultados

Recorte principal 2009–2020: 323.007 registros e 5.712 pares (rua, bairro) com
pelo menos 3 patches, em 64 bairros. Os modelos usam efeito fixo de bairro,
hierarquia viária, renda do setor censitário 2022, ônibus e faixa como controles,
com erro-padrão agrupado por bairro.

Os efeitos são razões por +10 pontos percentuais da rua com área não edificada.
O critério é p < 0,0125 (Bonferroni), e a composição é o teste principal.

| Hipótese | Volume | Composição | Veredito |
|---|---|---|---|
| H1 Queima a céu aberto | 1,09 | 1,07 | apoiada |
| H2 Depósito irregular | 1,15 | 1,12 | apoiada |
| H3 Invasão | 1,03 | 1,00 | sem evidência |
| H4 Cesta de desordem | 1,03 | 0,98 | sem evidência |
| C Trânsito (controle negativo) | 1,01 | 1,00 | sem efeito |

Nos testes de robustez, H2 resiste a todos. H1 enfraquece com corte de confiança
≥ 0,5: na composição, p = 0,12, enquanto o volume segue significativo. As análises
de todas as naturezas e de ônibus + área × roubo são exploratórias.

Desempenho do detector, detalhado em [`Modelo/README.md`](Modelo/README.md):
mAP50 0,735 na validação. Na revisão manual de 400 patches, a precisão por presença
foi de 0,99 para área não edificada, 0,92 para faixa e 0,86 para ônibus.

## Limites

- O dado da Guarda é notificação, não crime: reflete onde se policia e onde se denuncia.
- A análise usa a rua do patch (~69 m de lado), não a rua de cada objeto detectado.
- A imagem é de 2019 e a renda é de 2022, aplicadas a ocorrências de 2009–2020.
- A classe mede área não edificada visível, o que inclui verde dentro de lote construído.
- O recall do detector foi medido apenas dentro dos 400 patches revisados, todos com
  alguma detecção.
- Os resultados são associações, não efeitos causais.

## Estrutura

```
scripts/
  essenciais/                cadeia que produz o resultado final
  organizacao_verificacao/   automação e conferência
Modelo/
  dataset/                   348 imagens rotuladas (272 treino, 76 validação)
  runs/yolo26x_obb_v4_xl/    curvas e métricas do treino
  verificacao_2026-10-28/    registros da reexecução independente do pipeline
TCC-data/                    tabelas finais rua × características × ocorrências
```

Cada script está descrito em [`scripts/README.md`](scripts/README.md).

**Fora do repositório**, por tamanho: ortofotos e patches, base do Sigesguarda,
camadas do GeoCuritiba, detecções em lote e pesos do modelo (`best.pt`, 126 MB).

## Dados

| Base | Uso | Origem |
|---|---|---|
| Ortofotos 2019 | Imagens (zoom 20, ~0,135 m/px) | GeoCuritiba/IPPUC |
| Eixos de logradouro e quadras | Endereçamento dos patches | GeoCuritiba/IPPUC |
| Setor censitário 2022 | Renda, como controle | GeoCuritiba/IPPUC |
| Sigesguarda, sistema antigo | Ocorrências | [Dados abertos de Curitiba](https://dadosabertos.c3sl.ufpr.br/curitiba/Sigesguarda/) |

Os dados do IPPUC são públicos e têm caráter informativo, sem licença aberta explícita.

## Ambiente

- **Detector:** Python 3.13 com GPU; dependências em
  [`Modelo/requirements.txt`](Modelo/requirements.txt).
- **Análise estatística:** Python 3.12 com statsmodels 0.15.0, pandas 3.0.6,
  numpy 2.5.4 e scipy 1.18.1.

Os scripts usam caminhos absolutos em `D:\CityVision`.
