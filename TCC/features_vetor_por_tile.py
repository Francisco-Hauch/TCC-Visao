#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
features_vetor_por_tile.py
============================
Estagio 5.3 do pipeline "Quadra a Quadra": vira a pergunta de
cruzar_vetores_quadra.py (Estagio 5.2) do avesso.

O 5.2 responde "este ponto de onibus / esta calcada cai em que tile?"
(indexado pelo VETOR). Este script responde a pergunta que interessa para
treinar/avaliar em cima de `tiles_rotulos_quadra.csv`: "ESTE TILE tem ponto
de onibus? tem calcada, e quanto dela?" (indexado pelo TILE).

Ponto de onibus: um ponto cai em exatamente um tile - trivial.

Calcada: um poligono de calcada e comprido, 95,6% deles cruzam mais de um
tile (visto no Estagio 5.2). Colar so no tile do centroide subestimaria
muito. Aqui cada poligono e recortado (Sutherland-Hodgman) contra o
quadrado [tx, tx+1) x [ty, ty+1) de CADA tile candidato (os que estao
dentro do bbox do poligono, em coordenada de tile), e a area recortada vira
fracao de cobertura daquele tile (1.0 = tile inteiro coberto de calcada).
Mesma familia de tecnica do recorte de reta Liang-Barsky ja usado no
Estagio 2 (rotular_tiles.py), so que para poligono em vez de segmento.

Saida: tiles_rotulos_quadra.csv + colunas novas, gravado em
`tiles_rotulos_quadra_vetor.csv` (nao sobrescreve o arquivo final anterior,
mesma convencao dos estagios anteriores). Colunas novas:
  tem_ponto_onibus, n_pontos_onibus, tipos_onibus
  tem_calcada, n_calcada_poligonos, cobertura_calcada_frac

Requisitos: apenas a biblioteca padrao do Python (>=3.8).

Exemplo
-------
python features_vetor_por_tile.py
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rotular_tiles import deg2tilef


# --------------------------------------------------------------------------
# Recorte de poligono contra retangulo (Sutherland-Hodgman)
# --------------------------------------------------------------------------
def _clip_meio(pts, dentro, corte):
    if not pts:
        return pts
    out = []
    n = len(pts)
    for i in range(n):
        cur = pts[i]
        prev = pts[i - 1]
        cur_in = dentro(cur)
        prev_in = dentro(prev)
        if cur_in:
            if not prev_in:
                out.append(corte(prev, cur))
            out.append(cur)
        elif prev_in:
            out.append(corte(prev, cur))
    return out


def clip_poligono_retangulo(pts, xmin, ymin, xmax, ymax):
    """Recorta poligono convexo-ou-nao (anel simples) contra retangulo axis-aligned."""
    def corte_x(a, b, x):
        ax, ay = a
        bx, by = b
        t = (x - ax) / (bx - ax)
        return (x, ay + t * (by - ay))

    def corte_y(a, b, y):
        ax, ay = a
        bx, by = b
        t = (y - ay) / (by - ay)
        return (ax + t * (bx - ax), y)

    pts = _clip_meio(pts, lambda p: p[0] >= xmin, lambda a, b: corte_x(a, b, xmin))
    pts = _clip_meio(pts, lambda p: p[0] <= xmax, lambda a, b: corte_x(a, b, xmax))
    pts = _clip_meio(pts, lambda p: p[1] >= ymin, lambda a, b: corte_y(a, b, ymin))
    pts = _clip_meio(pts, lambda p: p[1] <= ymax, lambda a, b: corte_y(a, b, ymax))
    return pts


def area_poligono(pts):
    n = len(pts)
    if n < 3:
        return 0.0
    a = 0.0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        a += x0 * y1 - x1 * y0
    return abs(a) * 0.5


# --------------------------------------------------------------------------
def carregar_tiles_existentes(csv_rotulos):
    """Devolve set {(x, y)} de todo tile que ja foi baixado (linhas do CSV)."""
    tiles = set()
    with open(csv_rotulos, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            tiles.add((int(r["x"]), int(r["y"])))
    return tiles


def processar_onibus(caminho, tiles_existentes, z):
    """tile (x,y) -> {'n': int, 'tipos': set(str)}"""
    with open(caminho, encoding="utf-8") as f:
        fc = json.load(f)
    agg = defaultdict(lambda: {"n": 0, "tipos": set()})
    for feat in fc.get("features", []):
        g = feat.get("geometry") or {}
        p = feat.get("properties") or {}
        if g.get("type") != "Point":
            continue
        lon, lat = g["coordinates"][0], g["coordinates"][1]
        px, py = deg2tilef(lat, lon, z)
        tx, ty = int(px), int(py)
        if (tx, ty) not in tiles_existentes:
            continue
        a = agg[(tx, ty)]
        a["n"] += 1
        tipo = p.get("tipo")
        if tipo:
            a["tipos"].add(tipo)
    return agg


def processar_calcada(caminho, tiles_existentes, z):
    """tile (x,y) -> {'n': int, 'area_frac': float}"""
    with open(caminho, encoding="utf-8") as f:
        fc = json.load(f)
    agg = defaultdict(lambda: {"n": 0, "area_frac": 0.0})
    n_feats = 0
    for feat in fc.get("features", []):
        g = feat.get("geometry") or {}
        t = g.get("type")
        if t == "Polygon":
            aneis_deg = [g.get("coordinates", [[]])[0]] if g.get("coordinates") else []
        elif t == "MultiPolygon":
            aneis_deg = [poli[0] for poli in (g.get("coordinates") or []) if poli]
        else:
            continue
        tiles_tocados = set()
        for anel_deg in aneis_deg:
            pts = [deg2tilef(c[1], c[0], z) for c in anel_deg]
            if len(pts) < 3:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            tx0, tx1 = int(min(xs)), int(max(xs))
            ty0, ty1 = int(min(ys)), int(max(ys))
            for tx in range(tx0, tx1 + 1):
                for ty in range(ty0, ty1 + 1):
                    if (tx, ty) not in tiles_existentes:
                        continue
                    recorte = clip_poligono_retangulo(pts, tx, ty, tx + 1, ty + 1)
                    a = area_poligono(recorte)
                    if a > 1e-9:
                        agg[(tx, ty)]["area_frac"] += a
                        tiles_tocados.add((tx, ty))
        for k in tiles_tocados:
            agg[k]["n"] += 1
        n_feats += 1
    return agg, n_feats


def main():
    ap = argparse.ArgumentParser(
        description="Agrega ponto de onibus e calcada por TILE (inverso de cruzar_vetores_quadra.py).")
    ap.add_argument("--dados", default=r"D:\CityVision\TCC-data")
    ap.add_argument("--rotulos", default=None,
                    help="padrao: <dados>\\tiles_rotulos_quadra.csv")
    ap.add_argument("--saida", default=None,
                    help="padrao: <dados>\\tiles_rotulos_quadra_vetor.csv")
    ap.add_argument("--zoom", type=int, default=20)
    args = ap.parse_args()

    dados = args.dados
    rotulos_path = args.rotulos or os.path.join(dados, "tiles_rotulos_quadra.csv")
    saida_path = args.saida or os.path.join(dados, "tiles_rotulos_quadra_vetor.csv")
    z = args.zoom

    t0 = time.time()
    print("[1/4] carregando tiles existentes...")
    tiles_existentes = carregar_tiles_existentes(rotulos_path)
    print("      %d tiles" % len(tiles_existentes))

    print("[2/4] ponto de onibus...")
    onibus_path = os.path.join(dados, "VetoresFeatures", "pontos_onibus.geojson")
    agg_onibus = processar_onibus(onibus_path, tiles_existentes, z)
    print("      %d tiles com ponto de onibus" % len(agg_onibus))

    print("[3/4] calcada (recorte poligono x tile)...")
    calcada_path = os.path.join(dados, "VetoresFeatures", "area_passeio.geojson")
    agg_calcada, n_calcada_feats = processar_calcada(calcada_path, tiles_existentes, z)
    print("      %d poligonos processados -> %d tiles com calcada" % (n_calcada_feats, len(agg_calcada)))

    print("[4/4] juntando com %s..." % rotulos_path)
    with open(rotulos_path, encoding="utf-8-sig", newline="") as f:
        linhas = list(csv.DictReader(f, delimiter=";"))
    colunas_novas = ["tem_ponto_onibus", "n_pontos_onibus", "tipos_onibus",
                      "tem_calcada", "n_calcada_poligonos", "cobertura_calcada_frac"]
    for r in linhas:
        k = (int(r["x"]), int(r["y"]))
        ob = agg_onibus.get(k)
        cl = agg_calcada.get(k)
        r["tem_ponto_onibus"] = int(bool(ob))
        r["n_pontos_onibus"] = ob["n"] if ob else 0
        r["tipos_onibus"] = "|".join(sorted(ob["tipos"])) if ob else ""
        r["tem_calcada"] = int(bool(cl))
        r["n_calcada_poligonos"] = cl["n"] if cl else 0
        r["cobertura_calcada_frac"] = "%.4f" % cl["area_frac"] if cl else "0.0000"

    colunas = list(linhas[0].keys())
    with open(saida_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=colunas, delimiter=";")
        w.writeheader()
        w.writerows(linhas)

    n = len(linhas)
    n_ob = sum(1 for r in linhas if r["tem_ponto_onibus"] == 1)
    n_cl = sum(1 for r in linhas if r["tem_calcada"] == 1)
    print("\nOK: %d tiles em %.1fs" % (n, time.time() - t0))
    print("    com ponto de onibus: %d (%.2f%%)" % (n_ob, 100.0 * n_ob / n))
    print("    com calcada:         %d (%.2f%%)" % (n_cl, 100.0 * n_cl / n))
    print("    -> %s" % saida_path)


if __name__ == "__main__":
    main()
