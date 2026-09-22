#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reorganizar_por_regional.py
=============================
Move (nao copia - mesmo volume, so troca a entrada de diretorio) os tiles
z20 e os patches z19 para subpastas por regional OFICIAL (polígono do
Decreto Municipal 844/2018, camada 3 do MapaCadastral - ver
classificar_regional.py, que so fazia a contagem sem tocar em arquivo).

Estrutura nova:
  <dados>\\Ortofotos2019\\<regional>\\20\\<x>\\<y>.jpg
  <dados>\\Patches512\\<regional>\\19\\<px>\\<py>.jpg

<regional> e um dos 10 slugs oficiais (matriz, boa_vista, santa_felicidade,
portao, boqueirao, cajuru, cic, pinheirinho, bairro_novo, tatuquara) ou
`_sem_regional` para o que cai fora de qualquer poligono oficial (franja/
limite do municipio - decisao de 2026-09-13: manter tudo, nao descartar).

NAO atualiza os CSVs (tiles_rotulos_quadra*.csv, patches_quadra.csv) - a
coluna `arquivo`/os caminhos antigos ficam desatualizados ate um proximo
passo (decisao explicita para nao fazer tudo de uma vez).

Requisitos: apenas a biblioteca padrao do Python (>=3.8).
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classificar_regional import carregar_regionais, regional_do_ponto

SEM_REGIONAL = "_sem_regional"


def mover_tiles(dados, regionais20, dir_tiles_raiz):
    print("[1/2] tiles z20...")
    cont = defaultdict(int)
    n = 0
    with open(os.path.join(dados, "tiles_rotulos_quadra.csv"), encoding="utf-8-sig", newline="") as f:
        linhas = list(csv.DictReader(f, delimiter=";"))
    for r in linhas:
        x, y = int(r["x"]), int(r["y"])
        origem = os.path.join(dir_tiles_raiz, "20", str(x), "%d.jpg" % y)
        if not os.path.isfile(origem):
            continue
        slug_r, _ = regional_do_ponto(x + 0.5, y + 0.5, regionais20)
        slug_r = slug_r or SEM_REGIONAL
        destino_dir = os.path.join(dir_tiles_raiz, slug_r, "20", str(x))
        os.makedirs(destino_dir, exist_ok=True)
        destino = os.path.join(destino_dir, "%d.jpg" % y)
        os.replace(origem, destino)
        cont[slug_r] += 1
        n += 1
        if n % 20000 == 0:
            print("      %d movidos..." % n)
    return cont


def mover_patches(dados, regionais19, dir_patches_raiz):
    print("[2/2] patches z19...")
    cont = defaultdict(int)
    n = 0
    with open(os.path.join(dados, "patches_quadra.csv"), encoding="utf-8-sig", newline="") as f:
        linhas = list(csv.DictReader(f, delimiter=";"))
    for r in linhas:
        px, py = int(r["px"]), int(r["py"])
        origem = os.path.join(dir_patches_raiz, "19", str(px), "%d.jpg" % py)
        if not os.path.isfile(origem):
            continue
        slug_r, _ = regional_do_ponto(px + 0.5, py + 0.5, regionais19)
        slug_r = slug_r or SEM_REGIONAL
        destino_dir = os.path.join(dir_patches_raiz, slug_r, "19", str(px))
        os.makedirs(destino_dir, exist_ok=True)
        destino = os.path.join(destino_dir, "%d.jpg" % py)
        os.replace(origem, destino)
        cont[slug_r] += 1
        n += 1
        if n % 10000 == 0:
            print("      %d movidos..." % n)
    return cont


def limpar_dirs_vazios(raiz):
    """Remove os diretorios <x>/ e o '20'/'19' que ficaram vazios apos mover tudo."""
    for dirpath, dirnames, filenames in os.walk(raiz, topdown=False):
        if dirpath == raiz:
            continue
        try:
            os.rmdir(dirpath)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser(description="Reorganiza tiles/patches em pastas por regional oficial.")
    ap.add_argument("--dados", default=r"D:\CityVision\TCC-data")
    ap.add_argument("--regionais", default=None)
    args = ap.parse_args()

    dados = args.dados
    regionais_path = args.regionais or os.path.join(dados, "regionais.geojson")
    dir_tiles_raiz = os.path.join(dados, "Ortofotos2019")
    dir_patches_raiz = os.path.join(dados, "Patches512")

    t0 = time.time()
    regionais20 = carregar_regionais(regionais_path, 20)
    regionais19 = carregar_regionais(regionais_path, 19)

    cont_tiles = mover_tiles(dados, regionais20, dir_tiles_raiz)
    cont_patches = mover_patches(dados, regionais19, dir_patches_raiz)

    print("\n[limpeza] removendo diretorios antigos vazios (20/<x>, 19/<px>)...")
    velho_tiles = os.path.join(dir_tiles_raiz, "20")
    velho_patches = os.path.join(dir_patches_raiz, "19")
    if os.path.isdir(velho_tiles):
        limpar_dirs_vazios(velho_tiles)
        try:
            os.rmdir(velho_tiles)
        except OSError as e:
            print("  ! %s nao ficou vazio (%s) - sobrou algo nao classificado" % (velho_tiles, e))
    if os.path.isdir(velho_patches):
        limpar_dirs_vazios(velho_patches)
        try:
            os.rmdir(velho_patches)
        except OSError as e:
            print("  ! %s nao ficou vazio (%s) - sobrou algo nao classificado" % (velho_patches, e))

    print("\nOK em %.1fs" % (time.time() - t0))
    print("  tiles por regional:")
    for k, v in sorted(cont_tiles.items(), key=lambda kv: -kv[1]):
        print("    %-20s %6d" % (k, v))
    print("  patches por regional:")
    for k, v in sorted(cont_patches.items(), key=lambda kv: -kv[1]):
        print("    %-20s %6d" % (k, v))


if __name__ == "__main__":
    main()
