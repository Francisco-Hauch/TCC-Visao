#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baixar_eixos_logradouro.py
==========================
Estagio 2a do pipeline "Quadra a Quadra": baixa os eixos de logradouro do
GeoCuritiba (IPPUC) para um arquivo GeoJSON local, que depois eh usado por
rotular_tiles.py para dar nome de rua a cada tile de ortofoto.

Camada: GeoCuritiba/Publico_GeoCuritiba_MapaCadastral/MapServer/11
        "Trecho Logradouro" (polilinha, ~42 mil trechos, SIRGAS2000 UTM 22S)
A consulta pede outSR=4326 (WGS84), o mesmo datum usado na matematica de
tiles slippy-map do baixar_ortofotos_ippuc.py.

Requisitos: apenas a biblioteca padrao do Python (>=3.8).

Exemplos
--------
python baixar_eixos_logradouro.py
python baixar_eixos_logradouro.py --saida D:\TCC-data --pagina 2000
python baixar_eixos_logradouro.py --camada 12   # "Logradouro" (eixo inteiro)
"""

import argparse
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HOST = "https://geocuritiba.ippuc.org.br"
SERVICE = (HOST + "/server/rest/services/GeoCuritiba/"
           "Publico_GeoCuritiba_MapaCadastral/MapServer")

# Campos por camada. Mantidos enxutos: o GeoJSON dos eixos ja passa de 20 MB
# so com a geometria.
CAMPOS_EIXO = [
    "gtm_cod_logradouro",   # codigo do logradouro (chave estavel)
    "gtm_nm_logradouro",    # nome oficial: "RUA XV DE NOVEMBRO"
    "nm_log_abrev",         # nome abreviado
    "geo_nm_bairro_e",      # bairro do lado esquerdo do eixo
    "geo_nm_bairro_d",      # bairro do lado direito
    "geo_nm_regional_e",    # regional (esq)
    "geo_nm_regional_d",    # regional (dir)
    "cep_e",
    "cep_d",
    "hierarquia_viaria",    # 1=estrutural ... 4=local (classificacao viaria)
    "sist_viario_classificado",
]

# Quadra Cadastral (poligono): usada no estagio 3 para dar endereco aos tiles
# de miolo de quadra, que nao tem eixo de rua dentro deles.
CAMPOS_QUADRA = [
    "gtm_cod_quadrafiscal",   # codigo da quadra fiscal (chave)
    "gtm_indfiscais",         # indicacoes fiscais contidas
]

# Lote Cadastral (poligono): usado na regra geometrica de baldio (Estagio
# 5, item 1/4 do CLAUDE.md). gtm_desc_natureza = "Territorial" (tributado so
# pelo terreno, sem predio declarado) vs "Predial" e o pseudo-rotulo fiscal
# de baldio/subutilizado que o CLAUDE.md previa so conseguir via LAI -
# achado em 2026-09-13: ja vem aberto nesta camada, sem precisar pedir nada.
CAMPOS_LOTE = [
    "gtm_cod_lote",            # codigo do lote (chave)
    "gtm_cod_quadra",          # quadra fiscal (liga com CAMPOS_QUADRA)
    "gtm_desc_natureza",       # "Territorial" (so terreno) | "Predial" (com predio)
    "gtm_desc_situacao",       # "Ativo" etc.
    "gtm_desc_especie",        # Normal, Condominio, Subeconomia...
    "gtm_mtr_area_terreno",    # area do terreno (m2)
    "gtm_nm_bairro",
    "gtm_nm_logradouro",
    "gtm_num_predial",
]

# camada -> (campos, nome padrao do arquivo)
CAMADAS = {
    11: (CAMPOS_EIXO, "eixos_logradouro.geojson"),
    12: (CAMPOS_EIXO, "logradouros_eixo_inteiro.geojson"),
    15: (CAMPOS_LOTE, "lotes_cadastrais.geojson"),
    17: (CAMPOS_QUADRA, "quadras_cadastrais.geojson"),
}

USER_AGENT = "TCC-QuadraAQuadra/1.0 (uso academico; contato: fran.hauch@gmail.com)"


def _ctx():
    c = ssl.create_default_context()
    # O certificado do IPPUC costuma falhar a cadeia em maquinas Windows sem
    # o intermediario instalado; o dado eh publico, entao seguimos.
    c.check_hostname = False
    c.verify_mode = ssl.CERT_NONE
    return c


def get_json(url, tentativas=4, timeout=180):
    ctx = _ctx()
    ultimo = None
    for i in range(tentativas):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:          # rede instavel: backoff simples
            ultimo = e
            espera = 2 ** i
            print("  ! falha (%s), retry em %ds" % (e, espera), file=sys.stderr)
            time.sleep(espera)
    raise RuntimeError("desisti apos %d tentativas: %s" % (tentativas, ultimo))


def contar(base):
    p = {"where": "1=1", "returnCountOnly": "true", "f": "json"}
    return get_json(base + "/query?" + urllib.parse.urlencode(p))["count"]


def baixar(base, pagina, campos):
    """Pagina a camada inteira e devolve a lista de features GeoJSON."""
    total = contar(base)
    print("Trechos na camada: %d" % total)
    feats = []
    offset = 0
    while offset < total:
        p = {
            "where": "1=1",
            "outFields": ",".join(campos),
            "returnGeometry": "true",
            "outSR": "4326",
            "resultOffset": str(offset),
            "resultRecordCount": str(pagina),
            "orderByFields": "objectid",
            "f": "geojson",
        }
        d = get_json(base + "/query?" + urllib.parse.urlencode(p))
        lote = d.get("features", [])
        if not lote:
            print("  lote vazio em offset %d — encerrando" % offset)
            break
        feats.extend(lote)
        offset += len(lote)
        print("  %6d / %6d (%4.1f%%)" % (offset, total, 100.0 * offset / total))
    return feats


def main():
    ap = argparse.ArgumentParser(description="Baixa eixos de logradouro do GeoCuritiba.")
    ap.add_argument("--saida", default=r"D:\CityVision\TCC-data",
                    help="diretorio de saida (padrao: D:\\CityVision\\TCC-data)")
    ap.add_argument("--arquivo", default=None,
                    help="nome do GeoJSON (padrao: eixos_logradouro.geojson)")
    ap.add_argument("--camada", type=int, default=11,
                    help="11=Trecho Logradouro (padrao), 12=Logradouro, 15=Lote Cadastral, 17=Quadra Cadastral")
    ap.add_argument("--pagina", type=int, default=2000,
                    help="registros por requisicao (maxRecordCount=2000)")
    args = ap.parse_args()

    base = "%s/%d" % (SERVICE, args.camada)
    if args.camada not in CAMADAS:
        raise SystemExit("camada %d nao mapeada em CAMADAS" % args.camada)
    campos, nome_padrao = CAMADAS[args.camada]
    nome = args.arquivo or nome_padrao
    destino = os.path.join(args.saida, nome)
    os.makedirs(args.saida, exist_ok=True)

    t0 = time.time()
    feats = baixar(base, args.pagina, campos)

    fc = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "metadata": {
            "fonte": base,
            "gerado_em": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_features": len(feats),
        },
        "features": feats,
    }
    tmp = destino + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False)
    os.replace(tmp, destino)

    mb = os.path.getsize(destino) / 1e6
    print("OK: %d feicoes -> %s (%.1f MB, %.1fs)"
          % (len(feats), destino, mb, time.time() - t0))


if __name__ == "__main__":
    main()
