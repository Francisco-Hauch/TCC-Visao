#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rodar_tile2net.py
===================
Roda o Tile2Net (github.com/VIDA-NYU/tile2net, BSD-3, sem gate - ver
conversa de 2026-09-13) nos nossos tiles locais, pra segmentar RUA,
CALCADA, FAIXA DE PEDESTRE e FOOTPATH numa passada so. Modelo treinado em
imagem aerea nadir (Cambridge/MA, DC, NYC, Boston, Oregon, Alameda County -
nao Curitiba, entao vale validar visualmente antes de confiar no resultado
em escala).

O formato de tile que o Tile2Net espera bate com o nosso: XYZ, 256x256,
zoom 19-20. `input_dir` usa "x" e "y" como PLACEHOLDERS LITERAIS no
caminho (nao substituicao de string) - e assim que a biblioteca deles
funciona, ver DATA_PREPARE.md do projeto.

USO (uma vez so, no ambiente):
  git clone https://github.com/VIDA-NYU/tile2net.git
  cd tile2net && pip install -e .
  (ja feito nesta maquina em 2026-09-13)

Rodar (voce controla o tamanho/GPU pelos flags - a inferencia de
segmentacao usa GPU, a geracao/stitch dos tiles nao):
  python rodar_tile2net.py --regional matriz --teste
  python rodar_tile2net.py --regional matriz

--teste usa uma janela pequena (~12x12 tiles) no centro da regional, pra
validar rapido antes de rodar a regional inteira.
"""

import argparse
import math
import os

from tile2net import Raster


def tile2deg(x, y, z):
    n = 2.0 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


def calcular_bbox(dir_regional_zoom, janela=None):
    """dir_regional_zoom: .../Ortofotos2019/<regional>/<zoom>. Devolve string
    'lat,lon,lat,lon' cobrindo todos os tiles no disco (ou uma janela
    quadrada de `janela` tiles de lado, centrada, se `janela` for passado)."""
    xs = [int(d) for d in os.listdir(dir_regional_zoom)
          if os.path.isdir(os.path.join(dir_regional_zoom, d))]
    xmin, xmax = min(xs), max(xs)
    ys = []
    for d in os.listdir(dir_regional_zoom):
        pd = os.path.join(dir_regional_zoom, d)
        if not os.path.isdir(pd):
            continue
        for fn in os.listdir(pd):
            if fn.endswith(".jpg"):
                ys.append(int(fn[:-4]))
    ymin, ymax = min(ys), max(ys)

    if janela:
        cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
        h = janela // 2
        xmin, xmax = cx - h, cx + h
        ymin, ymax = cy - h, cy + h

    z = int(os.path.basename(dir_regional_zoom))
    lat0, lon0 = tile2deg(xmin, ymin, z)
    lat1, lon1 = tile2deg(xmax + 1, ymax + 1, z)
    return "%s,%s,%s,%s" % (lat0, lon0, lat1, lon1)


def main():
    ap = argparse.ArgumentParser(description="Roda Tile2Net (rua/calcada/faixa de pedestre) nos tiles locais.")
    ap.add_argument("--dados", default=r"D:\CityVision\TCC-data")
    ap.add_argument("--regional", required=True, help="ex.: matriz, boa_vista, cic...")
    ap.add_argument("--zoom", type=int, default=20)
    ap.add_argument("--stitch", type=int, default=4,
                    help="tiles agrupados por lado ao quadrado, tile2net usa 'stitch every n' "
                         "- 4 == equivalente ao bloco 2x2 do nosso Estagio 4")
    ap.add_argument("--saida", default=None, help="padrao: <dados>\\Tile2Net\\<regional>")
    ap.add_argument("--teste", action="store_true",
                    help="usa so uma janela pequena (~12x12 tiles) no centro da regional, pra validar rapido")
    ap.add_argument("--dump-percent", type=int, default=100,
                    help="%% de imagens de segmentacao a salvar (100 = todas, uteis pra conferir visualmente)")
    args = ap.parse_args()

    dir_regional_zoom = os.path.join(args.dados, "Ortofotos2019", args.regional, str(args.zoom))
    if not os.path.isdir(dir_regional_zoom):
        raise SystemExit("nao achei %s - regional errado ou zoom errado?" % dir_regional_zoom)

    input_dir = os.path.join(args.dados, "Ortofotos2019", args.regional, str(args.zoom), "x", "y.jpg")
    saida = args.saida or os.path.join(args.dados, "Tile2Net", args.regional)
    nome = args.regional + ("_teste" if args.teste else "")

    print("[1/3] calculando bbox da regional (janela=%s)..." % (12 if args.teste else "toda"))
    bbox = calcular_bbox(dir_regional_zoom, janela=12 if args.teste else None)
    print("      bbox:", bbox)

    print("[2/3] montando Raster + stitch (sem GPU ainda)...")
    raster = Raster(
        location=bbox,
        name=nome,
        input_dir=input_dir,
        output_dir=saida,
        zoom=args.zoom,
    )
    raster.generate(args.stitch)

    print("[3/3] rodando inferencia de segmentacao (usa GPU agora)...")
    raster.inference()

    print("\nOK -> %s" % saida)


if __name__ == "__main__":
    main()
