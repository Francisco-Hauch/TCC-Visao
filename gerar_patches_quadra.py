#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gerar_patches_quadra.py
=========================
Estagio 4 do pipeline "Quadra a Quadra": gera os patches 512x512 px que vao
alimentar o modelo de terreno baldio (SAM 3 zero-shot e o que vier depois).
Um tile z20 sozinho (34,5 m de lado) e pouco contexto para reconhecer lote
baldio; o atalho de graca ja identificado no CLAUDE.md (secao 5, item 3) e
que um bloco 2x2 de tiles z20 e MATEMATICAMENTE IDENTICO a um tile z19
(`foto_pai = (x // 2, y // 2)`), 512x512 px / ~69x69 m - so precisa colar
as 4 imagens, sem reprojetar nada.

Para cada foto_pai com pelo menos 1 dos 4 tiles-filho baixado:
  - cola os que existem no quadrante certo de um canvas 512x512 (RGB, preto
    onde falta filho - registra quantos dos 4 existem em `n_tiles`);
  - agrega metadados dos filhos vindos de `tiles_rotulos_quadra_vetor.csv`
    (Estagio 2+3+5.3): quadra(s), bairro, endereco, tem_ponto_onibus,
    tem_calcada, cobertura_calcada_frac media - o patch ja nasce com as
    features de vetor prontas, sem precisar reabrir os 4 tiles depois.

Saida:
  <saida>\\<px>\\<py>.jpg              - os patches
  <dados>\\patches_quadra.csv          - metadados por patch (1 linha cada)

Requisitos: Pillow (`pip install pillow`, ja presente no ambiente usado).

Exemplo
-------
python gerar_patches_quadra.py
python gerar_patches_quadra.py --amostra 5
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rotular_tiles import tile2deg

TAM_TILE = 256
TAM_PATCH = 512


def carregar_metadados(caminho):
    """(x,y) -> dict da linha do CSV (Estagio 2+3+5.3)."""
    meta = {}
    with open(caminho, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            meta[(int(r["x"]), int(r["y"]))] = r
    return meta


def listar_tiles_existentes(dir_tiles, z):
    """Escaneia o disco (nao o CSV) - pega qualquer tile realmente presente."""
    raiz = os.path.join(dir_tiles, str(z))
    existentes = set()
    for dx in os.listdir(raiz):
        pdx = os.path.join(raiz, dx)
        if not os.path.isdir(pdx):
            continue
        x = int(dx)
        for fn in os.listdir(pdx):
            if fn.endswith(".jpg"):
                existentes.add((x, int(fn[:-4])))
    return existentes


def agrega_meta(filhos_meta):
    """Junta os metadados dos ate 4 tiles-filho de um patch."""
    quadras = []
    bairros = set()
    enderecos = []
    tem_onibus = 0
    n_onibus = 0
    tem_calcada = 0
    n_calcada = 0
    cobertura = []
    for m in filhos_meta:
        if m is None:
            continue
        if m.get("quadra"):
            quadras.append(m["quadra"])
        if m.get("bairro"):
            bairros.add(m["bairro"])
        if m.get("endereco"):
            enderecos.append(m["endereco"])
        if m.get("tem_ponto_onibus") == "1":
            tem_onibus = 1
        n_onibus += int(m.get("n_pontos_onibus", 0) or 0)
        if m.get("tem_calcada") == "1":
            tem_calcada = 1
        n_calcada += int(m.get("n_calcada_poligonos", 0) or 0)
        try:
            cobertura.append(float(m.get("cobertura_calcada_frac", 0) or 0))
        except ValueError:
            pass
    quadras_unicas = sorted(set(quadras))
    return {
        "quadras": "|".join(quadras_unicas),
        "n_quadras": len(quadras_unicas),
        "bairro": "|".join(sorted(bairros)),
        "endereco": enderecos[0] if enderecos else "",
        "tem_ponto_onibus": tem_onibus,
        "n_pontos_onibus": n_onibus,
        "tem_calcada": tem_calcada,
        "n_calcada_poligonos": n_calcada,
        "cobertura_calcada_frac_media": "%.4f" % (sum(cobertura) / len(cobertura)) if cobertura else "0.0000",
    }


def main():
    ap = argparse.ArgumentParser(description="Gera patches 512x512 (2x2 tiles z20) por quadra.")
    ap.add_argument("--dados", default=r"D:\CityVision\TCC-data")
    ap.add_argument("--tiles", default=None, help="padrao: <dados>\\Ortofotos2019")
    ap.add_argument("--metadados", default=None,
                    help="padrao: <dados>\\tiles_rotulos_quadra_vetor.csv")
    ap.add_argument("--saida", default=None, help="padrao: <dados>\\Patches512\\19")
    ap.add_argument("--zoom", type=int, default=20)
    ap.add_argument("--qualidade", type=int, default=92)
    ap.add_argument("--amostra", type=int, default=0)
    args = ap.parse_args()

    dados = args.dados
    dir_tiles = args.tiles or os.path.join(dados, "Ortofotos2019")
    meta_path = args.metadados or os.path.join(dados, "tiles_rotulos_quadra_vetor.csv")
    dir_saida = args.saida or os.path.join(dados, "Patches512", "19")
    z = args.zoom
    z_patch = z - 1

    t0 = time.time()
    print("[1/4] lendo metadados (%s)..." % meta_path)
    meta = carregar_metadados(meta_path)
    print("      %d tiles com metadado" % len(meta))

    print("[2/4] listando tiles existentes em disco...")
    existentes = listar_tiles_existentes(dir_tiles, z)
    print("      %d tiles no disco" % len(existentes))

    print("[3/4] agrupando em blocos 2x2 (foto_pai)...")
    grupos = defaultdict(list)
    for (x, y) in existentes:
        grupos[(x // 2, y // 2)].append((x, y))
    print("      %d patches (blocos 2x2 com >=1 filho)" % len(grupos))

    print("[4/4] colando e gravando...")
    linhas = []
    n_completos = 0
    for i, ((px, py), filhos) in enumerate(sorted(grupos.items())):
        canvas = Image.new("RGB", (TAM_PATCH, TAM_PATCH))
        filhos_meta = []
        for (x, y) in filhos:
            caminho = os.path.join(dir_tiles, str(z), str(x), "%d.jpg" % y)
            try:
                im = Image.open(caminho).convert("RGB")
            except Exception as e:
                print("  ! falha lendo %s (%s), pulando filho" % (caminho, e), file=sys.stderr)
                continue
            ox = (x - 2 * px) * TAM_TILE
            oy = (y - 2 * py) * TAM_TILE
            canvas.paste(im, (ox, oy))
            filhos_meta.append(meta.get((x, y)))

        destino_dir = os.path.join(dir_saida, str(px))
        os.makedirs(destino_dir, exist_ok=True)
        destino = os.path.join(destino_dir, "%d.jpg" % py)
        canvas.save(destino, "JPEG", quality=args.qualidade)

        lat_c, lon_c = tile2deg(px + 0.5, py + 0.5, z_patch)
        agg = agrega_meta(filhos_meta)
        n_tiles = len(filhos)
        if n_tiles == 4:
            n_completos += 1
        linha = {"z_patch": z_patch, "px": px, "py": py, "arquivo": destino,
                 "n_tiles": n_tiles, "lat_centro": "%.7f" % lat_c, "lon_centro": "%.7f" % lon_c}
        linha.update(agg)
        linhas.append(linha)

        if (i + 1) % 5000 == 0:
            print("      %d / %d patches" % (i + 1, len(grupos)))

    destino_csv = os.path.join(dados, "patches_quadra.csv")
    colunas = list(linhas[0].keys())
    with open(destino_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=colunas, delimiter=";")
        w.writeheader()
        w.writerows(linhas)

    print("\nOK: %d patches em %.1fs" % (len(linhas), time.time() - t0))
    print("    completos (4/4 tiles): %d (%.1f%%)" % (n_completos, 100.0 * n_completos / len(linhas)))
    print("    -> %s" % dir_saida)
    print("    -> %s" % destino_csv)

    if args.amostra:
        print("\n  amostra:")
        for r in linhas[:args.amostra]:
            print("   ", r)


if __name__ == "__main__":
    main()
