#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classificar_regional.py
=========================
Classifica cada tile z20 e cada patch z19 pela REGIONAL OFICIAL do IPPUC
(poligono `regionais.geojson`, camada 3 do MapaCadastral - Decreto Municipal
844/2018), nao pelos retangulos aproximados usados so para o download
original (aqueles nao sao fronteira real, se sobrepoem entre regionais
vizinhas - ver baixar_ortofotos_ippuc.py).

So classifica e reporta contagem - NAO move nenhum arquivo. E o passo de
conferencia antes de reorganizar de fato (proximo script).

Requisitos: apenas a biblioteca padrao do Python (>=3.8).
"""

import argparse
import csv
import json
import os
import sys
import unicodedata
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rotular_tiles import deg2tilef
from rotular_quadras import ponto_em_anel


def slug(nome):
    n = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode("ascii")
    return n.strip().lower().replace(" ", "_")


def carregar_regionais(caminho, z):
    with open(caminho, encoding="utf-8") as f:
        fc = json.load(f)
    out = []
    for feat in fc["features"]:
        g = feat["geometry"]
        p = feat["properties"]
        polis = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        aneis = []
        for poli in polis:
            anel = [deg2tilef(c[1], c[0], z) for c in poli[0]]
            aneis.append(anel)
        xs = [pt[0] for anel in aneis for pt in anel]
        ys = [pt[1] for anel in aneis for pt in anel]
        out.append({"nome": p["nome"], "slug": slug(p["nome"]), "aneis": aneis,
                     "bbox": (min(xs), min(ys), max(xs), max(ys))})
    return out


def regional_do_ponto(px, py, regionais):
    for r in regionais:
        x0, y0, x1, y1 = r["bbox"]
        if not (x0 <= px <= x1 and y0 <= py <= y1):
            continue
        if any(ponto_em_anel(px, py, anel) for anel in r["aneis"]):
            return r["slug"], r["nome"]
    return None, None


def main():
    ap = argparse.ArgumentParser(description="Classifica tiles/patches pela regional oficial (sem mover nada).")
    ap.add_argument("--dados", default=r"D:\CityVision\TCC-data")
    ap.add_argument("--regionais", default=None, help="padrao: <dados>\\regionais.geojson")
    args = ap.parse_args()

    dados = args.dados
    regionais_path = args.regionais or os.path.join(dados, "regionais.geojson")

    print("[1/3] tiles z20 (tiles_rotulos_quadra.csv)...")
    regionais20 = carregar_regionais(regionais_path, 20)
    cont_tiles = defaultdict(int)
    sem_regional_tiles = 0
    with open(os.path.join(dados, "tiles_rotulos_quadra.csv"), encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            px, py = int(r["x"]) + 0.5, int(r["y"]) + 0.5
            slug_r, nome_r = regional_do_ponto(px, py, regionais20)
            if slug_r is None:
                sem_regional_tiles += 1
            else:
                cont_tiles[slug_r] += 1
    print("      tiles classificados:")
    for k, v in sorted(cont_tiles.items(), key=lambda kv: -kv[1]):
        print("        %-20s %6d" % (k, v))
    print("        %-20s %6d  <- fora de qualquer regional oficial (bug ou franja)" %
          ("(nenhuma)", sem_regional_tiles))

    print("\n[2/3] patches z19 (patches_quadra.csv)...")
    regionais19 = carregar_regionais(regionais_path, 19)
    cont_patches = defaultdict(int)
    sem_regional_patches = 0
    with open(os.path.join(dados, "patches_quadra.csv"), encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            px, py = int(r["px"]) + 0.5, int(r["py"]) + 0.5
            slug_r, nome_r = regional_do_ponto(px, py, regionais19)
            if slug_r is None:
                sem_regional_patches += 1
            else:
                cont_patches[slug_r] += 1
    print("      patches classificados:")
    for k, v in sorted(cont_patches.items(), key=lambda kv: -kv[1]):
        print("        %-20s %6d" % (k, v))
    print("        %-20s %6d  <- fora de qualquer regional oficial" %
          ("(nenhuma)", sem_regional_patches))

    print("\n[3/3] nomes/slugs das 10 regionais oficiais:")
    for r in regionais20:
        print("      %-20s <- %s" % (r["slug"], r["nome"]))


if __name__ == "__main__":
    main()
