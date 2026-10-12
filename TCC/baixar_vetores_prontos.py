#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baixar_vetores_prontos.py
==========================
Estagio 5.1 do pipeline "Quadra a Quadra": baixa os DOIS vetores que a
decisao de 2026-09-13 tirou da lista de tarefas de visao computacional
(ver CLAUDE.md, secao 3.4) - ponto de onibus e calcada. Nenhum dos dois
precisa de modelo: ja existem como vetor com coordenada e atributo no
cadastro do IPPUC/URBS.

Camadas
-------
onibus  -> GeoCuritiba/URBS_Transporte_Publico/MapServer/1  ("Parada de
           Onibus", ponto, ~7.253 registros, campo "tipo" com o gabarito
           chapeu-chines/estacao-tubo/etc.)
calcada -> GeoCuritiba/Publico_Interno_GeoCuritiba_BaseCartografica_para_BC/
           MapServer/80 ("Area de Passeio", poligono, ~74.047 registros,
           campos largura/calcada/pavimentacao)

O nome do segundo servico tem "_Interno_" no meio - o nome mais curto usado
em versoes antigas do CLAUDE.md (sem "_Interno_") devolve 404/"Token
Required". Confirmado por sondagem em 2026-09-13.

Mesmo padrao de baixar_eixos_logradouro.py: pagina via Query da API
MapServer/ArcGIS REST, grava GeoJSON. Apenas biblioteca padrao do Python
(>=3.8).

Exemplos
--------
python baixar_vetores_prontos.py                  # baixa os dois
python baixar_vetores_prontos.py --camada onibus
python baixar_vetores_prontos.py --camada calcada --pagina 1000
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

# nome curto -> (servico, camada, campos, nome do arquivo de saida)
CAMADAS = {
    "onibus": (
        "URBS_Transporte_Publico", 1,
        ["objectid", "nome_ponto", "num", "tipo"],
        "pontos_onibus.geojson",
    ),
    "calcada": (
        "Publico_Interno_GeoCuritiba_BaseCartografica_para_BC", 80,
        ["objectid", "largura", "calcada", "pavimentacao"],
        "area_passeio.geojson",
    ),
}

USER_AGENT = "TCC-QuadraAQuadra/1.0 (uso academico; contato: fran.hauch@gmail.com)"


def _ctx():
    c = ssl.create_default_context()
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
        except Exception as e:
            ultimo = e
            espera = 2 ** i
            print("  ! falha (%s), retry em %ds" % (e, espera), file=sys.stderr)
            time.sleep(espera)
    raise RuntimeError("desisti apos %d tentativas: %s" % (tentativas, ultimo))


def contar(base):
    p = {"where": "1=1", "returnCountOnly": "true", "f": "json"}
    d = get_json(base + "/query?" + urllib.parse.urlencode(p))
    if "error" in d:
        raise RuntimeError("erro do servidor: %s" % d["error"])
    return d["count"]


def baixar(base, pagina, campos):
    total = contar(base)
    print("Registros na camada: %d" % total)
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
        if "error" in d:
            raise RuntimeError("erro do servidor em offset %d: %s" % (offset, d["error"]))
        lote = d.get("features", [])
        if not lote:
            print("  lote vazio em offset %d - encerrando" % offset)
            break
        feats.extend(lote)
        offset += len(lote)
        print("  %6d / %6d (%4.1f%%)" % (offset, total, 100.0 * offset / total))
    return feats


def baixar_camada(nome, saida, pagina):
    servico, camada, campos, nome_padrao = CAMADAS[nome]
    base = "%s/server/rest/services/GeoCuritiba/%s/MapServer/%d" % (HOST, servico, camada)
    destino = os.path.join(saida, nome_padrao)
    os.makedirs(saida, exist_ok=True)

    print("\n=== %s (%s) ===" % (nome, base))
    t0 = time.time()
    feats = baixar(base, pagina, campos)

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


def main():
    ap = argparse.ArgumentParser(description="Baixa ponto de onibus e/ou calcada do GeoCuritiba/URBS.")
    ap.add_argument("--saida", default=r"D:\CityVision\TCC-data\VetoresFeatures",
                    help="diretorio de saida (padrao: D:\\CityVision\\TCC-data\\VetoresFeatures - "
                         "movido para la em 2026-09-20 junto com as outras bases de feature)")
    ap.add_argument("--camada", choices=sorted(CAMADAS), default=None,
                    help="baixar so uma camada (padrao: as duas)")
    ap.add_argument("--pagina", type=int, default=2000,
                    help="registros por requisicao (maxRecordCount=2000)")
    args = ap.parse_args()

    nomes = [args.camada] if args.camada else list(CAMADAS)
    for nome in nomes:
        baixar_camada(nome, args.saida, args.pagina)


if __name__ == "__main__":
    main()
