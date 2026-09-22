#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baixar_bases_features.py
=========================
Baixa as bases de vetor NOVAS que faltavam para as features escolhidas em
2026-09-20 (calcada, terreno baldio, predio, casa, industria, poste,
semaforo) - as que JA estavam prontas (calcada = area_passeio.geojson,
ponto de onibus = pontos_onibus.geojson, lote cadastral = lotes_cadastrais.
geojson) foram MOVIDAS (nao baixadas de novo) para a mesma pasta logo em
seguida, a pedido do usuario, pra tudo ficar junto - ver git-free "mv" em
2026-09-20. `quadras_cadastrais.geojson` e `eixos_logradouro.geojson`
continuam na raiz de TCC-data (usados por rotular_quadras.py/rotular_tiles.py
com default hardcoded, nao sao "feature" desta lista).

Saida vai para um diretorio proprio, TCC-data\\VetoresFeatures, que agora
reune TODAS as bases de vetor das features escolhidas (novas + as movidas):

  edificacoes.geojson              - camada 72 (Edificacao), ~1,1 milhao de
                                      poligonos. Resolve PREDIO na hora; CASA
                                      e INDUSTRIA sao derivados depois (nao
                                      tem camada propria):
                                        casa      = finalidade=="Residencial"
                                                    e numeropavimentos baixo
                                        industria = classeativecon/
                                                    divisaoativecon com
                                                    codigo de fabricacao
                                      Ambos os dois sao HEURISTICA sobre o
                                      mesmo arquivo, nao um download a mais.
  postes.geojson                   - camada 71 (Poste), ~dezenas de milhares
                                      de pontos. Unica camada com
                                      created_date/last_edited_date (data de
                                      edicao no GIS, NAO de instalacao -
                                      sinal temporal fraco, nao confiavel).
  setor_censitario_2010.geojson    - BaseTematica/10, com campo `renda`.
  setor_censitario_2022.geojson    - BaseTematica/45, com campo
                                      `renda_media` + populacao/domicilios.
                                      Usados para controlar renda por epoca
                                      (item 7-f do CLAUDE.md), nao sao
                                      "feature" do ambiente construido.
  semaforos_osm.geojson            - OpenStreetMap (Overpass API), tag
                                      highway=traffic_signals. NAO EXISTE
                                      camada de semaforo no GeoCuritiba (
                                      verificado em 2026-09-20, vasculhando
                                      as 3 servicos publicos existentes) -
                                      OSM e a unica fonte aberta achada.
                                      Cobertura e colaborativa/irregular,
                                      tende a ser pior em bairro periferico -
                                      mesmo tipo de viés do item 7-a, agora
                                      na fonte do dado e nao no detector.
  faixa_pedestre_osm.geojson       - OSM, node highway=crossing + way
                                      footway=crossing. BONUS: o GeoCuritiba
                                      so tem travessia em desnivel
                                      (passarela/subterranea, camada 20),
                                      NAO faixa pintada no chao. OSM cobre
                                      isso parcialmente - tratar como
                                      complemento/validacao cruzada da
                                      deteccao por imagem (Tile2Net), nunca
                                      como inventario completo sozinho.

Nenhuma das bases acima tem versao historica exposta por API (nem
GeoCuritiba nem OSM de forma confiavel pre-2013/14) - sao retrato ATUAL.
Ver a conversa de 2026-09-20 no CLAUDE.md para o porque.

Requisitos: so biblioteca padrao (urllib) - nada para instalar.

Uso
---
  python baixar_bases_features.py                 # baixa tudo
  python baixar_bases_features.py --camada poste   # so uma
  python baixar_bases_features.py --listar         # lista as chaves
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
SERVICE_BC = HOST + "/server/rest/services/GeoCuritiba/Publico_Interno_GeoCuritiba_BaseCartografica_para_BC/MapServer"
SERVICE_TEMATICA = HOST + "/server/rest/services/GeoCuritiba/Publico_Interno_GeoCuritiba_BaseTematica/MapServer"

USER_AGENT = "TCC-QuadraAQuadra/1.0 (uso academico; contato: fran.hauch@gmail.com)"

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

CAMPOS_EDIFICACAO = [
    "objectid", "finalidade", "numeropavimentos", "matconstr", "situacaofisica",
    "administracao", "classeativecon", "divisaoativecon", "grupoativecon",
    "bairro", "cep", "logradouro", "municipio", "alturaaproximada",
    "geometriaaproximada",
]

CAMPOS_POSTE = [
    "objectid", "codident", "tipoposte", "matconstr", "geometriaaproximada",
    "created_date", "last_edited_date",
]

CAMPOS_SETOR_2010 = [
    "objectid", "cd_geocodsetor", "nm_bairro", "nm_regional", "tipo",
    "pessoas", "domicilios", "dens_pop_km2", "renda",
]

CAMPOS_SETOR_2022 = [
    "objectid", "cd_setor", "nm_bairro", "area_km2", "v0001", "v0002",
    "v0003", "v0005", "v0007", "renda_media",
]

# chave -> (service, id da camada, campos, arquivo de saida)
CAMADAS_ARCGIS = {
    "edificacao": (SERVICE_BC, 72, CAMPOS_EDIFICACAO, "edificacoes.geojson"),
    "poste": (SERVICE_BC, 71, CAMPOS_POSTE, "postes.geojson"),
    "setor2010": (SERVICE_TEMATICA, 10, CAMPOS_SETOR_2010, "setor_censitario_2010.geojson"),
    "setor2022": (SERVICE_TEMATICA, 45, CAMPOS_SETOR_2022, "setor_censitario_2022.geojson"),
}

# chave -> (query Overpass QL sem o cabecalho [out:json], arquivo de saida)
CAMADAS_OSM = {
    "semaforo": (
        'node["highway"="traffic_signals"](area.a);',
        "semaforos_osm.geojson",
    ),
    "faixa_pedestre": (
        'node["highway"="crossing"](area.a);way["footway"="crossing"](area.a);',
        "faixa_pedestre_osm.geojson",
    ),
}


def _ctx():
    c = ssl.create_default_context()
    c.check_hostname = False
    c.verify_mode = ssl.CERT_NONE
    return c


def get_json(url, tentativas=4, timeout=180, dados=None, metodo=None):
    ctx = _ctx()
    ultimo = None
    for i in range(tentativas):
        try:
            req = urllib.request.Request(
                url, data=dados, method=metodo,
                headers={"User-Agent": USER_AGENT,
                         "Content-Type": "application/x-www-form-urlencoded"},
            )
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
    return get_json(base + "/query?" + urllib.parse.urlencode(p))["count"]


def baixar_arcgis(service, camada_id, campos, pagina=2000):
    base = "%s/%d" % (service, camada_id)
    total = contar(base)
    print("  registros na camada: %d" % total)
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
            print("  lote vazio em offset %d - encerrando" % offset)
            break
        feats.extend(lote)
        offset += len(lote)
        print("  %7d / %7d (%4.1f%%)" % (offset, total, 100.0 * offset / total))
    return feats, base


def baixar_osm(query_ql):
    """Overpass QL, com area de Curitiba resolvida por nome (mesma logica
    testada manualmente em 2026-09-20 via WebFetch)."""
    ql = ('[out:json][timeout:120];'
          'area["name"="Curitiba"]["boundary"="administrative"]->.a;'
          '(%s);'
          'out body geom;') % query_ql
    dados = urllib.parse.urlencode({"data": ql}).encode("utf-8")
    d = get_json(OVERPASS_URL, dados=dados, metodo="POST", timeout=150)

    feats = []
    for el in d.get("elements", []):
        tags = el.get("tags", {})
        if el["type"] == "node":
            geom = {"type": "Point", "coordinates": [el["lon"], el["lat"]]}
        elif el["type"] == "way" and "geometry" in el:
            coords = [[pt["lon"], pt["lat"]] for pt in el["geometry"]]
            geom = {"type": "LineString", "coordinates": coords}
        else:
            continue
        feats.append({
            "type": "Feature",
            "geometry": geom,
            "properties": dict(tags, osm_type=el["type"], osm_id=el["id"]),
        })
    return feats


def gravar_geojson(destino, feats, fonte):
    fc = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "metadata": {
            "fonte": fonte,
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
    print("  OK: %d feicoes -> %s (%.1f MB)" % (len(feats), destino, mb))


def main():
    todas = sorted(list(CAMADAS_ARCGIS) + list(CAMADAS_OSM))
    ap = argparse.ArgumentParser(description="Baixa bases de vetor complementares (edificacao, poste, censo, OSM).")
    ap.add_argument("--saida", default=r"D:\CityVision\TCC-data\VetoresFeatures")
    ap.add_argument("--camada", choices=todas, default=None,
                    help="baixa so uma (padrao: todas). opcoes: %s" % ", ".join(todas))
    ap.add_argument("--pagina", type=int, default=2000)
    ap.add_argument("--listar", action="store_true")
    args = ap.parse_args()

    if args.listar:
        for c in todas:
            print(c)
        return

    os.makedirs(args.saida, exist_ok=True)
    alvos = [args.camada] if args.camada else todas

    for chave in alvos:
        print("\n=== %s ===" % chave)
        t0 = time.time()
        if chave in CAMADAS_ARCGIS:
            service, camada_id, campos, arquivo = CAMADAS_ARCGIS[chave]
            feats, base = baixar_arcgis(service, camada_id, campos, args.pagina)
            gravar_geojson(os.path.join(args.saida, arquivo), feats, base)
        else:
            query_ql, arquivo = CAMADAS_OSM[chave]
            feats = baixar_osm(query_ql)
            gravar_geojson(os.path.join(args.saida, arquivo), feats, "overpass-api.de (OpenStreetMap)")
        print("  %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
