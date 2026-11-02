#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rotular_tiles.py
================
Estagio 2b do pipeline "Quadra a Quadra": da NOME DE RUA (e bairro/regional/
CEP) a cada tile de ortofoto ja baixado.

Como funciona
-------------
Cada tile z/x/y do cache XYZ cobre um quadrado conhecido em WGS84. No zoom 20
esse quadrado tem ~34,5 m de lado em Curitiba - ou seja, o tile normalmente
pega um pedaco de uma ou duas ruas. Entao:

  1. os eixos de logradouro (eixos_logradouro.geojson, vindo do
     baixar_eixos_logradouro.py) sao convertidos para coordenadas de tile
     fracionarias no mesmo zoom;
  2. cada segmento de eixo eh recortado (Liang-Barsky) contra o quadrado de
     cada tile que ele cruza, e o comprimento recortado eh acumulado;
  3. a "rua principal" do tile eh o logradouro com maior comprimento de eixo
     dentro dele - criterio de dominancia, nao de mera presenca;
  4. tiles sem nenhum eixo dentro (miolo de quadra, parque, patio) recebem,
     opcionalmente, a rua mais proxima do centro do tile dentro de um raio.

Tudo offline depois do download dos eixos: nenhuma chamada de rede por tile
(seriam 180 mil chamadas a um geocoder).

Requisitos: apenas a biblioteca padrao do Python (>=3.8).

Exemplos
--------
# rotula tudo que esta no disco e escreve o CSV:
python rotular_tiles.py

# so uma faixa de tiles, com JSONL tambem e sem o fallback de proximidade:
python rotular_tiles.py --limite-x 380691 380822 --limite-y 600836 600997 \
    --formato ambos --raio-vizinho 0

# conferir alguns tiles rotulados:
python rotular_tiles.py --amostra 20
"""

import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict

# --------------------------------------------------------------------------
# Matematica de tiles (identica a do baixar_ortofotos_ippuc.py)
# --------------------------------------------------------------------------
CIRC = 40075016.685578488          # circunferencia equatorial (m), EPSG:3857


def deg2tilef(lat, lon, z):
    """lat/lon -> coordenada de tile FRACIONARIA (float) no zoom z."""
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 2.0 ** z
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def tile2deg(x, y, z):
    """Canto superior-esquerdo do tile (aceita x/y fracionarios)."""
    n = 2.0 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


# --------------------------------------------------------------------------
# Recorte de segmento contra o quadrado do tile (Liang-Barsky)
# --------------------------------------------------------------------------
def fracao_dentro(x0, y0, x1, y1, xmin, ymin, xmax, ymax):
    """Fracao [0..1] do segmento (x0,y0)-(x1,y1) que cai dentro do retangulo.

    Devolve 0.0 se nao ha interseccao. Liang-Barsky resolve em O(1) e trata
    segmentos paralelos aos lados sem caso especial.
    """
    dx = x1 - x0
    dy = y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - xmin), (dx, xmax - x0),
                 (-dy, y0 - ymin), (dy, ymax - y0)):
        if p == 0.0:
            if q < 0.0:
                return 0.0          # paralelo ao lado e fora dele
            continue
        r = q / p
        if p < 0.0:
            if r > t1:
                return 0.0
            if r > t0:
                t0 = r
        else:
            if r < t0:
                return 0.0
            if r < t1:
                t1 = r
    return t1 - t0 if t1 > t0 else 0.0


def dist_ponto_segmento(px, py, x0, y0, x1, y1):
    """Distancia ponto-segmento no plano, em unidades de tile."""
    dx, dy = x1 - x0, y1 - y0
    den = dx * dx + dy * dy
    if den == 0.0:
        return math.hypot(px - x0, py - y0)
    t = ((px - x0) * dx + (py - y0) * dy) / den
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return math.hypot(px - (x0 + t * dx), py - (y0 + t * dy))


# --------------------------------------------------------------------------
# Carga dos eixos
# --------------------------------------------------------------------------
def carregar_eixos(caminho, z):
    """Le o GeoJSON e devolve (segmentos, atributos_por_logradouro).

    segmentos: lista de (x0, y0, x1, y1, chave) ja em coordenada de tile
               fracionaria - a projecao acontece uma vez so, aqui.
    """
    with open(caminho, encoding="utf-8") as f:
        fc = json.load(f)

    segs = []
    attrs = {}
    for feat in fc.get("features", []):
        g = feat.get("geometry") or {}
        p = feat.get("properties") or {}
        nome = (p.get("gtm_nm_logradouro") or "").strip()
        if not nome:
            continue                                  # trecho sem denominacao
        cod = p.get("gtm_cod_logradouro")
        # Chave de agrupamento: o codigo do logradouro quando existe. Dois
        # trechos da mesma rua devem somar comprimento, nao competir entre si.
        chave = ("c%s" % cod) if cod is not None else ("n%s" % nome)
        if chave not in attrs:
            attrs[chave] = {
                "nome": nome,
                "abrev": (p.get("nm_log_abrev") or "").strip(),
                "bairro": (p.get("geo_nm_bairro_e") or p.get("geo_nm_bairro_d") or "").strip(),
                "regional": (p.get("geo_nm_regional_e") or p.get("geo_nm_regional_d") or "").strip(),
                "cep": (p.get("cep_e") or p.get("cep_d") or "").strip(),
                "hierarquia": (p.get("hierarquia_viaria") or "").strip(),
            }

        t = g.get("type")
        if t == "LineString":
            partes = [g.get("coordinates") or []]
        elif t == "MultiLineString":
            partes = g.get("coordinates") or []
        else:
            continue

        for linha in partes:
            ant = None
            for c in linha:
                atual = deg2tilef(c[1], c[0], z)
                if ant is not None and atual != ant:
                    segs.append((ant[0], ant[1], atual[0], atual[1], chave))
                ant = atual
    return segs, attrs


# --------------------------------------------------------------------------
# Varredura dos tiles em disco
# --------------------------------------------------------------------------
def listar_tiles(raiz, z, lim_x, lim_y, exts=(".jpg", ".jpeg", ".png")):
    """Devolve {(x, y): caminho} dos tiles presentes em raiz/<z>/<x>/<y>.ext."""
    base = os.path.join(raiz, str(z))
    if not os.path.isdir(base):
        raise SystemExit("nao encontrei o diretorio de tiles: %s" % base)
    achados = {}
    for de in os.scandir(base):
        if not de.is_dir():
            continue
        try:
            x = int(de.name)
        except ValueError:
            continue
        if lim_x and not (lim_x[0] <= x <= lim_x[1]):
            continue
        for fe in os.scandir(de.path):
            if not fe.is_file():
                continue
            nome, ext = os.path.splitext(fe.name)
            if ext.lower() not in exts:
                continue
            try:
                y = int(nome)
            except ValueError:
                continue
            if lim_y and not (lim_y[0] <= y <= lim_y[1]):
                continue
            achados[(x, y)] = fe.path
    return achados


# --------------------------------------------------------------------------
# Nucleo: acumular comprimento de eixo por tile
# --------------------------------------------------------------------------
def cruzar(segs, tiles, z, grid_shift):
    """cobertura[(x,y)][chave] = comprimento de eixo dentro do tile, em metros.

    Devolve tambem um indice grosseiro grade->segmentos, usado pelo fallback
    de "rua mais proxima" nos tiles sem nenhum eixo dentro.
    """
    # Metros por unidade de tile, com a correcao de escala do Mercator: em
    # Curitiba (lat ~ -25.45) o fator cos(lat) vale ~0.903.
    y_medio = sum(t[1] for t in tiles) / float(len(tiles))
    lat_ref, _ = tile2deg(0.0, y_medio, z)
    escala = CIRC / (2.0 ** z) * math.cos(math.radians(lat_ref))

    cobertura = defaultdict(lambda: defaultdict(float))
    grade = defaultdict(list)
    for i, (x0, y0, x1, y1, chave) in enumerate(segs):
        comp = math.hypot(x1 - x0, y1 - y0) * escala
        if comp <= 0.0:
            continue
        tx0, tx1 = int(math.floor(min(x0, x1))), int(math.floor(max(x0, x1)))
        ty0, ty1 = int(math.floor(min(y0, y1))), int(math.floor(max(y0, y1)))
        for gx in range(tx0 >> grid_shift, (tx1 >> grid_shift) + 1):
            for gy in range(ty0 >> grid_shift, (ty1 >> grid_shift) + 1):
                grade[(gx, gy)].append(i)
        # A bbox de um trecho raramente passa de alguns tiles no z20.
        for tx in range(tx0, tx1 + 1):
            for ty in range(ty0, ty1 + 1):
                if (tx, ty) not in tiles:
                    continue
                fr = fracao_dentro(x0, y0, x1, y1, tx, ty, tx + 1.0, ty + 1.0)
                if fr > 0.0:
                    cobertura[(tx, ty)][chave] += fr * comp
    return cobertura, grade, escala


def vizinho_mais_proximo(tx, ty, segs, grade, grid_shift, escala, raio_m):
    """Rua mais proxima do centro do tile, dentro de raio_m. (None, None) se nao houver."""
    px, py = tx + 0.5, ty + 0.5
    raio_t = raio_m / escala
    gx, gy = tx >> grid_shift, ty >> grid_shift
    vistos = set()
    melhor_d, melhor_k = float("inf"), None
    for ax in (gx - 1, gx, gx + 1):
        for ay in (gy - 1, gy, gy + 1):
            for i in grade.get((ax, ay), ()):
                if i in vistos:
                    continue
                vistos.add(i)
                x0, y0, x1, y1, chave = segs[i]
                d = dist_ponto_segmento(px, py, x0, y0, x1, y1)
                if d < melhor_d:
                    melhor_d, melhor_k = d, chave
    if melhor_k is None or melhor_d > raio_t:
        return None, None
    return melhor_k, melhor_d * escala


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Atribui nome de rua (e bairro/regional/CEP) a cada tile de ortofoto.")
    ap.add_argument("--tiles", default=r"D:\TCC-data\Ortofotos2019",
                    help="raiz do cache de tiles (padrao: D:\\TCC-data\\Ortofotos2019)")
    ap.add_argument("--eixos", default=r"D:\TCC-data\eixos_logradouro.geojson",
                    help="GeoJSON dos eixos (gerado por baixar_eixos_logradouro.py)")
    ap.add_argument("--zoom", type=int, default=20)
    ap.add_argument("--saida", default=None,
                    help="caminho base da saida, sem extensao (padrao: <tiles>/../tiles_rotulos)")
    ap.add_argument("--formato", choices=("csv", "jsonl", "ambos"), default="csv")
    ap.add_argument("--limite-x", nargs=2, type=int, default=None, metavar=("X0", "X1"))
    ap.add_argument("--limite-y", nargs=2, type=int, default=None, metavar=("Y0", "Y1"))
    ap.add_argument("--raio-vizinho", type=float, default=60.0,
                    help="raio (m) para achar a rua mais proxima em tiles sem eixo; 0 desliga")
    ap.add_argument("--min-cobertura", type=float, default=1.0,
                    help="ignora ruas com menos de N metros de eixo dentro do tile")
    ap.add_argument("--amostra", type=int, default=0,
                    help="imprime N linhas de exemplo no final")
    args = ap.parse_args()

    t0 = time.time()
    z = args.zoom
    grid_shift = 3                       # celula da grade grossa = 8x8 tiles (~276 m)

    print("[1/4] varrendo tiles em disco...")
    tiles = listar_tiles(args.tiles, z, args.limite_x, args.limite_y)
    if not tiles:
        raise SystemExit("nenhum tile encontrado")
    print("      %d tiles" % len(tiles))

    print("[2/4] carregando eixos de logradouro...")
    segs, attrs = carregar_eixos(args.eixos, z)
    print("      %d segmentos, %d logradouros distintos" % (len(segs), len(attrs)))

    print("[3/4] cruzando eixos x tiles...")
    cobertura, grade, escala = cruzar(segs, tiles, z, grid_shift)
    print("      %d tiles com rua por interseccao (tile = %.1f m)"
          % (len(cobertura), escala))

    print("[4/4] escrevendo saida...")
    base = args.saida or os.path.join(
        os.path.dirname(os.path.abspath(args.tiles)), "tiles_rotulos")
    colunas = ["z", "x", "y", "arquivo", "lat_centro", "lon_centro",
               "rua_principal", "rua_abrev", "cobertura_m", "n_ruas", "ruas",
               "bairro", "regional", "cep", "hierarquia_viaria", "origem"]

    f_csv = f_jsonl = w = None
    if args.formato in ("csv", "ambos"):
        f_csv = open(base + ".csv", "w", newline="", encoding="utf-8-sig")
        w = csv.writer(f_csv, delimiter=";")
        w.writerow(colunas)
    if args.formato in ("jsonl", "ambos"):
        f_jsonl = open(base + ".jsonl", "w", encoding="utf-8")

    n_int = n_viz = n_sem = 0
    exemplos = []
    for (x, y), caminho in sorted(tiles.items()):
        lat_c, lon_c = tile2deg(x + 0.5, y + 0.5, z)   # centro do tile
        ruas = cobertura.get((x, y))
        if ruas:
            ordenadas = sorted(((m, k) for k, m in ruas.items()
                                if m >= args.min_cobertura), reverse=True)
            if not ordenadas:                          # so restos abaixo do corte
                ordenadas = sorted(((m, k) for k, m in ruas.items()), reverse=True)
            origem = "interseccao"
            n_int += 1
        else:
            ordenadas = []
            origem = "sem_rua"
            if args.raio_vizinho > 0:
                chave, dist = vizinho_mais_proximo(x, y, segs, grade, grid_shift,
                                                   escala, args.raio_vizinho)
                if chave:
                    ordenadas = [(0.0, chave)]
                    origem = "vizinho_%.0fm" % dist
                    n_viz += 1
            if not ordenadas:
                n_sem += 1

        if ordenadas:
            m_prin, k_prin = ordenadas[0]
            a = attrs[k_prin]
            linha = [z, x, y, caminho, "%.7f" % lat_c, "%.7f" % lon_c,
                     a["nome"], a["abrev"], "%.1f" % m_prin, len(ordenadas),
                     "|".join("%s:%.1f" % (attrs[k]["nome"], m) for m, k in ordenadas),
                     a["bairro"], a["regional"], a["cep"], a["hierarquia"], origem]
        else:
            linha = [z, x, y, caminho, "%.7f" % lat_c, "%.7f" % lon_c,
                     "", "", "0.0", 0, "", "", "", "", "", origem]

        if w:
            w.writerow(linha)
        if f_jsonl:
            f_jsonl.write(json.dumps(dict(zip(colunas, linha)), ensure_ascii=False) + "\n")
        if args.amostra and len(exemplos) < args.amostra and linha[6]:
            exemplos.append(linha)

    for f in (f_csv, f_jsonl):
        if f:
            f.close()

    n = len(tiles)
    print("OK: %d tiles rotulados em %.1fs" % (n, time.time() - t0))
    print("    por interseccao: %6d (%.1f%%)" % (n_int, 100.0 * n_int / n))
    print("    por proximidade: %6d (%.1f%%)" % (n_viz, 100.0 * n_viz / n))
    print("    sem rua:         %6d (%.1f%%)" % (n_sem, 100.0 * n_sem / n))
    if f_csv is not None:
        print("    -> %s.csv" % base)
    if f_jsonl is not None:
        print("    -> %s.jsonl" % base)
    for e in exemplos:
        print("      z%s/%s/%s  %-40s %6sm  %s / %s" % (e[0], e[1], e[2], e[6], e[8], e[11], e[12]))


if __name__ == "__main__":
    main()
