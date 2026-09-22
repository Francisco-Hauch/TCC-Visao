#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cruzar_vetores_quadra.py
=========================
Estagio 5.2 do pipeline "Quadra a Quadra": cruza os vetores prontos
(ponto de onibus, area de passeio - baixados por baixar_vetores_prontos.py)
com a grade de tiles/quadra ja pronta, por overlay espacial simples.

Para cada feicao do vetor:
  - QUADRA: reaproveita a mesma logica de rotular_quadras.py (ray casting
    do ponto/centroide dentro dos poligonos de quadras_cadastrais.geojson);
  - TILE (z20): converte lat/lon para coordenada de tile com deg2tilef() e
    trunca - eh a mesma conta de rotular_tiles.py, so que "ao contrario"
    (aqui a coordenada real ja existe no vetor, nao precisa achar rua mais
    proxima de pixel detectado - ver armadilha (b) do CLAUDE.md, que fala
    de OBJETO DETECTADO em imagem, nao de vetor com coordenada oficial);
  - TILE_EXISTE: confere se o tile calculado esta entre os 180.319 baixados
    (fora da area de 8 regionais cobertas, ou tile 100% branco descartado,
    o vetor pode apontar para um tile que nao temos foto).

Ponto de onibus (MapServer/1) ja e um ponto: usa a coordenada direto.
Area de passeio (camada 80) e poligono: usa o centroide (media dos vertices
do anel externo, ponderada por area via formula do shoelace) como
representante da feicao - overlay simples, como pedido no CLAUDE.md
(secao 5, item 2). Uma calcada comprida pode atravessar mais de um tile;
para isso this script tambem grava min/max de tile x/y do bbox da feicao,
para quem precisar saber que ela nao cabe em um tile so.

Saida: dois CSVs em --saida (padrao D:\\TCC-data):
  onibus_quadra.csv   (7.253 linhas)
  calcada_quadra.csv  (74.047 linhas)

Requisitos: apenas a biblioteca padrao do Python (>=3.8).

Exemplos
--------
python cruzar_vetores_quadra.py
python cruzar_vetores_quadra.py --amostra 10
"""

import argparse
import csv
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rotular_tiles import deg2tilef
from rotular_quadras import carregar_quadras, indexar, quadra_do_ponto, SHIFT


def centroide_anel(anel):
    """Centroide de poligono (shoelace), em coordenadas de tile fracionarias.

    anel: lista de (x, y). Cai para media simples dos vertices se a area
    calculada for ~0 (poligono degenerado - linha ou ponto duplicado).
    """
    n = len(anel)
    a = cx = cy = 0.0
    for i in range(n):
        x0, y0 = anel[i]
        x1, y1 = anel[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        a += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    a *= 0.5
    if abs(a) < 1e-12:
        xs = [p[0] for p in anel]
        ys = [p[1] for p in anel]
        return sum(xs) / n, sum(ys) / n
    cx /= (6.0 * a)
    cy /= (6.0 * a)
    return cx, cy


def tile_existe(dir_tiles, tx, ty, z):
    return os.path.isfile(os.path.join(dir_tiles, str(z), str(tx), "%d.jpg" % ty))


def processar_pontos(caminho, campos, quadras, grade_q, dir_tiles, z):
    with open(caminho, encoding="utf-8") as f:
        fc = json.load(f)
    linhas = []
    for feat in fc.get("features", []):
        g = feat.get("geometry") or {}
        p = feat.get("properties") or {}
        if g.get("type") != "Point":
            continue
        lon, lat = g["coordinates"][0], g["coordinates"][1]
        px, py = deg2tilef(lat, lon, z)
        tx, ty = int(px), int(py)
        iq = quadra_do_ponto(px, py, quadras, grade_q)
        cod = quadras[iq]["cod"] if iq is not None else ""
        linha = {c: p.get(c, "") for c in campos}
        linha.update({
            "lat": "%.7f" % lat, "lon": "%.7f" % lon,
            "tile_x": tx, "tile_y": ty,
            "tile_existe": int(tile_existe(dir_tiles, tx, ty, z)),
            "quadra": cod or "",
        })
        linhas.append(linha)
    return linhas


def processar_poligonos(caminho, campos, quadras, grade_q, dir_tiles, z):
    with open(caminho, encoding="utf-8") as f:
        fc = json.load(f)
    linhas = []
    for feat in fc.get("features", []):
        g = feat.get("geometry") or {}
        p = feat.get("properties") or {}
        t = g.get("type")
        if t == "Polygon":
            poligonos = [g.get("coordinates") or []]
        elif t == "MultiPolygon":
            poligonos = g.get("coordinates") or []
        else:
            continue
        # usa o maior anel externo entre os poligonos da feicao (caso MultiPolygon)
        melhor_anel = None
        melhor_n = -1
        todos_pts = []
        for poli in poligonos:
            if not poli:
                continue
            anel_deg = poli[0]
            pts = [deg2tilef(c[1], c[0], z) for c in anel_deg]
            todos_pts.extend(pts)
            if len(pts) > melhor_n:
                melhor_n = len(pts)
                melhor_anel = pts
        if not melhor_anel or len(melhor_anel) < 3:
            continue
        cx, cy = centroide_anel(melhor_anel)
        tx, ty = int(cx), int(cy)
        xs = [pt[0] for pt in todos_pts]
        ys = [pt[1] for pt in todos_pts]
        tx_min, tx_max = int(min(xs)), int(max(xs))
        ty_min, ty_max = int(min(ys)), int(max(ys))
        iq = quadra_do_ponto(cx, cy, quadras, grade_q)
        cod = quadras[iq]["cod"] if iq is not None else ""
        lat_c, lon_c = None, None
        # tile2deg nao importado aqui de proposito: so precisamos do lat/lon
        # do centroide para conferencia visual, calculo direto (inverso de deg2tilef)
        n = 2.0 ** z
        lon_c = cx / n * 360.0 - 180.0
        lat_c = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * cy / n))))
        linha = {c: p.get(c, "") for c in campos}
        linha.update({
            "lat_centro": "%.7f" % lat_c, "lon_centro": "%.7f" % lon_c,
            "tile_x": tx, "tile_y": ty,
            "tile_existe": int(tile_existe(dir_tiles, tx, ty, z)),
            "multi_tile": int(tx_min != tx_max or ty_min != ty_max),
            "tile_x_min": tx_min, "tile_x_max": tx_max,
            "tile_y_min": ty_min, "tile_y_max": ty_max,
            "quadra": cod or "",
        })
        linhas.append(linha)
    return linhas


def escrever(linhas, destino, campos_extra_ordem):
    if not linhas:
        print("  (0 linhas, nao escrevi %s)" % destino)
        return
    colunas = list(linhas[0].keys())
    with open(destino, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=colunas, delimiter=";")
        w.writeheader()
        w.writerows(linhas)


def main():
    ap = argparse.ArgumentParser(
        description="Cruza pontos_onibus.geojson e area_passeio.geojson com quadra/tile.")
    ap.add_argument("--dados", default=r"D:\CityVision\TCC-data")
    ap.add_argument("--quadras", default=None,
                    help="padrao: <dados>\\quadras_cadastrais.geojson")
    ap.add_argument("--tiles", default=None,
                    help="raiz dos tiles, padrao: <dados>\\Ortofotos2019")
    ap.add_argument("--zoom", type=int, default=20)
    ap.add_argument("--amostra", type=int, default=0)
    args = ap.parse_args()

    dados = args.dados
    quadras_path = args.quadras or os.path.join(dados, "quadras_cadastrais.geojson")
    dir_tiles = args.tiles or os.path.join(dados, "Ortofotos2019")
    z = args.zoom

    t0 = time.time()
    print("[1/3] carregando quadras cadastrais...")
    quadras = carregar_quadras(quadras_path, z)
    grade_q = indexar([(i, q["bbox"]) for i, q in enumerate(quadras)])
    print("      %d quadras" % len(quadras))

    print("[2/3] onibus...")
    onibus_path = os.path.join(dados, "VetoresFeatures", "pontos_onibus.geojson")
    onibus = processar_pontos(onibus_path, ["objectid", "nome_ponto", "num", "tipo"],
                               quadras, grade_q, dir_tiles, z)
    escrever(onibus, os.path.join(dados, "onibus_quadra.csv"), None)
    n_q = sum(1 for r in onibus if r["quadra"])
    n_t = sum(1 for r in onibus if r["tile_existe"])
    print("      %d pontos | em quadra: %d (%.1f%%) | tile baixado: %d (%.1f%%)"
          % (len(onibus), n_q, 100.0 * n_q / len(onibus), n_t, 100.0 * n_t / len(onibus)))

    print("[3/3] calcada...")
    calcada_path = os.path.join(dados, "VetoresFeatures", "area_passeio.geojson")
    calcada = processar_poligonos(calcada_path, ["objectid", "largura", "calcada", "pavimentacao"],
                                   quadras, grade_q, dir_tiles, z)
    escrever(calcada, os.path.join(dados, "calcada_quadra.csv"), None)
    n_q = sum(1 for r in calcada if r["quadra"])
    n_t = sum(1 for r in calcada if r["tile_existe"])
    n_m = sum(1 for r in calcada if r["multi_tile"])
    print("      %d poligonos | em quadra: %d (%.1f%%) | tile baixado: %d (%.1f%%) | multi-tile: %d (%.1f%%)"
          % (len(calcada), n_q, 100.0 * n_q / len(calcada), n_t, 100.0 * n_t / len(calcada),
             n_m, 100.0 * n_m / len(calcada)))

    print("\nOK em %.1fs" % (time.time() - t0))

    if args.amostra:
        print("\n  amostra onibus:")
        for r in onibus[:args.amostra]:
            print("   ", r)
        print("\n  amostra calcada:")
        for r in calcada[:args.amostra]:
            print("   ", r)


if __name__ == "__main__":
    main()
