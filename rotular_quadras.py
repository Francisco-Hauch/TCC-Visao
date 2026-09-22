#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rotular_quadras.py
==================
Estagio 3 do pipeline "Quadra a Quadra": resolve o buraco deixado pelo
estagio 2b (rotular_tiles.py) nos tiles marcados `sem_rua`.

O problema
----------
O criterio do estagio 2b eh estrito e correto: a rua do tile eh o eixo de
logradouro que passa DENTRO dele (ou, no fallback, a ate 60 m do centro).
So que um tile z20 tem ~34,5 m de lado, menor que a maioria das quadras de
Curitiba. Entao o tile que cai no miolo de uma quadra - fundo de lote, patio
de industria, telhado de galpao - nao tem eixo nenhum dentro e sai `sem_rua`,
mesmo estando numa quadra perfeitamente enderecada e mesmo quando a imagem
mostra via asfaltada (via interna de lote privado nao eh logradouro publico e
nao existe no cadastro do IPPUC).

A solucao
---------
Trocar a pergunta. Em vez de "que rua passa dentro deste tile?", perguntar
"a que QUADRA este tile pertence, e quais ruas confrontam essa quadra?".

  1. carrega as quadras cadastrais (quadras_cadastrais.geojson, via
     `baixar_eixos_logradouro.py --camada 17`);
  2. localiza o centro de cada tile dentro de uma quadra (ray casting);
  3. para cada quadra, pre-computa as ruas confrontantes - eixos a ate
     `--confrontante` metros da borda do poligono;
  4. `rua_quadra` do tile eh a confrontante mais proxima do centro dele.

O estagio 2b nao eh alterado: suas colunas continuam iguais e com o mesmo
significado. Este script apenas ACRESCENTA colunas ao CSV.

Requisitos: apenas a biblioteca padrao do Python (>=3.8).

Exemplos
--------
python baixar_eixos_logradouro.py --camada 17     # se ainda nao baixou
python rotular_quadras.py
python rotular_quadras.py --confrontante 40 --amostra 15
"""

import argparse
import csv
import math
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rotular_tiles import (CIRC, carregar_eixos, deg2tilef, tile2deg,
                           dist_ponto_segmento)

SHIFT = 3                      # celula da grade grossa = 8x8 tiles (~276 m)


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
def dist_seg_seg(ax, ay, bx, by, cx, cy, dx, dy):
    """Menor distancia entre os segmentos AB e CD (0.0 se eles se cruzam)."""
    d1x, d1y = bx - ax, by - ay
    d2x, d2y = dx - cx, dy - cy
    den = d1x * d2y - d1y * d2x
    if den != 0.0:                       # nao paralelos: testa interseccao
        t = ((cx - ax) * d2y - (cy - ay) * d2x) / den
        u = ((cx - ax) * d1y - (cy - ay) * d1x) / den
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            return 0.0
    return min(dist_ponto_segmento(ax, ay, cx, cy, dx, dy),
               dist_ponto_segmento(bx, by, cx, cy, dx, dy),
               dist_ponto_segmento(cx, cy, ax, ay, bx, by),
               dist_ponto_segmento(dx, dy, ax, ay, bx, by))


def ponto_em_anel(px, py, anel):
    """Ray casting: o ponto esta dentro do anel (lista de (x, y))?"""
    dentro = False
    n = len(anel)
    j = n - 1
    for i in range(n):
        xi, yi = anel[i]
        xj, yj = anel[j]
        if (yi > py) != (yj > py):
            if px < (xj - xi) * (py - yi) / (yj - yi) + xi:
                dentro = not dentro
        j = i
    return dentro


# --------------------------------------------------------------------------
# Carga das quadras
# --------------------------------------------------------------------------
def carregar_quadras(caminho, z):
    """Le o GeoJSON de quadras e devolve lista de dicts ja em coord. de tile.

    Cada quadra: {cod, aneis (externo + buracos), bbox}. Os aneis internos
    (buracos) sao mantidos porque uma quadra cadastral pode conter um vazio.
    """
    import json
    with open(caminho, encoding="utf-8") as f:
        fc = json.load(f)

    quadras = []
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
        for poli in poligonos:
            aneis = []
            for anel in poli:
                pts = [deg2tilef(c[1], c[0], z) for c in anel]
                if len(pts) >= 3:
                    aneis.append(pts)
            if not aneis:
                continue
            xs = [q[0] for q in aneis[0]]
            ys = [q[1] for q in aneis[0]]
            quadras.append({
                "cod": p.get("gtm_cod_quadrafiscal"),
                "aneis": aneis,
                "bbox": (min(xs), min(ys), max(xs), max(ys)),
            })
    return quadras


def indexar(itens_bbox):
    """bbox -> celulas da grade grossa. Recebe [(i, (x0,y0,x1,y1))]."""
    grade = defaultdict(list)
    for i, (x0, y0, x1, y1) in itens_bbox:
        for gx in range(int(x0) >> SHIFT, (int(x1) >> SHIFT) + 1):
            for gy in range(int(y0) >> SHIFT, (int(y1) >> SHIFT) + 1):
                grade[(gx, gy)].append(i)
    return grade


def quadra_do_ponto(px, py, quadras, grade_q):
    """Indice da quadra que contem o ponto, ou None."""
    for i in grade_q.get((int(px) >> SHIFT, int(py) >> SHIFT), ()):
        q = quadras[i]
        x0, y0, x1, y1 = q["bbox"]
        if not (x0 <= px <= x1 and y0 <= py <= y1):
            continue
        aneis = q["aneis"]
        if ponto_em_anel(px, py, aneis[0]):
            if not any(ponto_em_anel(px, py, a) for a in aneis[1:]):
                return i
    return None


def confrontantes(iq, quadras, segs, grade_s, escala, raio_m):
    """Eixos que confrontam a quadra: a ate raio_m da borda do poligono.

    Devolve [(dist_m, indice_do_segmento)] ordenado. Cache por quadra, porque
    varios tiles caem na mesma quadra.
    """
    q = quadras[iq]
    raio_t = raio_m / escala
    x0, y0, x1, y1 = q["bbox"]
    cand = set()
    for gx in range((int(x0 - raio_t)) >> SHIFT, ((int(x1 + raio_t)) >> SHIFT) + 1):
        for gy in range((int(y0 - raio_t)) >> SHIFT, ((int(y1 + raio_t)) >> SHIFT) + 1):
            cand.update(grade_s.get((gx, gy), ()))

    achados = []
    for i in cand:
        sx0, sy0, sx1, sy1, chave = segs[i]
        # descarte barato pela bbox antes da conta cara
        if (max(sx0, sx1) < x0 - raio_t or min(sx0, sx1) > x1 + raio_t or
                max(sy0, sy1) < y0 - raio_t or min(sy0, sy1) > y1 + raio_t):
            continue
        melhor = float("inf")
        for anel in q["aneis"]:
            for j in range(len(anel)):
                ax, ay = anel[j]
                bx, by = anel[(j + 1) % len(anel)]
                d = dist_seg_seg(ax, ay, bx, by, sx0, sy0, sx1, sy1)
                if d < melhor:
                    melhor = d
                    if melhor == 0.0:
                        break
            if melhor == 0.0:
                break
        if melhor <= raio_t:
            achados.append((melhor, i))
    achados.sort()
    return achados


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Acrescenta quadra cadastral e rua confrontante ao CSV de tiles.")
    ap.add_argument("--rotulos", default=r"D:\TCC-data\tiles_rotulos.csv",
                    help="CSV gerado por rotular_tiles.py")
    ap.add_argument("--quadras", default=r"D:\TCC-data\quadras_cadastrais.geojson")
    ap.add_argument("--eixos", default=r"D:\TCC-data\eixos_logradouro.geojson")
    ap.add_argument("--zoom", type=int, default=20)
    ap.add_argument("--saida", default=None,
                    help="CSV de saida (padrao: <rotulos> com sufixo _quadra)")
    ap.add_argument("--confrontante", type=float, default=30.0,
                    help="distancia (m) da borda da quadra para um eixo contar como confrontante")
    ap.add_argument("--amostra", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    z = args.zoom

    print("[1/5] carregando eixos...")
    segs, attrs = carregar_eixos(args.eixos, z)
    grade_s = indexar([(i, (min(s[0], s[2]), min(s[1], s[3]),
                            max(s[0], s[2]), max(s[1], s[3])))
                       for i, s in enumerate(segs)])
    print("      %d segmentos" % len(segs))

    print("[2/5] carregando quadras cadastrais...")
    quadras = carregar_quadras(args.quadras, z)
    grade_q = indexar([(i, q["bbox"]) for i, q in enumerate(quadras)])
    print("      %d quadras" % len(quadras))

    print("[3/5] lendo rotulos do estagio 2b...")
    with open(args.rotulos, encoding="utf-8-sig", newline="") as f:
        linhas = list(csv.DictReader(f, delimiter=";"))
    print("      %d tiles" % len(linhas))

    y_medio = sum(int(r["y"]) for r in linhas) / float(len(linhas))
    lat_ref, _ = tile2deg(0.0, y_medio, z)
    escala = CIRC / (2.0 ** z) * math.cos(math.radians(lat_ref))

    print("[4/5] localizando tiles nas quadras e achando confrontantes...")
    cache = {}
    n_q = n_recup = 0
    saida = []
    for r in linhas:
        tx, ty = int(r["x"]), int(r["y"])
        px, py = tx + 0.5, ty + 0.5
        iq = quadra_do_ponto(px, py, quadras, grade_q)
        cod = rua_q = ""
        dist_q = ""
        lista_q = ""
        n_conf = 0
        if iq is not None:
            n_q += 1
            cod = quadras[iq]["cod"] or ""
            if iq not in cache:
                cache[iq] = confrontantes(iq, quadras, segs, grade_s, escala,
                                          args.confrontante)
            conf = cache[iq]
            n_conf = len({segs[i][4] for _, i in conf})
            if conf:
                # a confrontante mais proxima do CENTRO DO TILE, nao da quadra:
                # num quarteirao comprido, o fundo do lote pertence a rua que
                # esta do seu lado, nao a primeira da lista da quadra.
                melhor_d, melhor_k = float("inf"), None
                for _, i in conf:
                    sx0, sy0, sx1, sy1, chave = segs[i]
                    d = dist_ponto_segmento(px, py, sx0, sy0, sx1, sy1)
                    if d < melhor_d:
                        melhor_d, melhor_k = d, chave
                if melhor_k is not None:
                    rua_q = attrs[melhor_k]["nome"]
                    dist_q = "%.1f" % (melhor_d * escala)
                vistos = []
                for _, i in conf:
                    nome = attrs[segs[i][4]]["nome"]
                    if nome not in vistos:
                        vistos.append(nome)
                lista_q = "|".join(vistos)
                if r["origem"] == "sem_rua" and rua_q:
                    n_recup += 1

        # coluna de conveniencia: o melhor endereco disponivel para o tile,
        # na ordem eixo-dentro > vizinho-60m > confrontante-da-quadra.
        endereco = r["rua_principal"] or rua_q
        base = "eixo" if r["origem"] == "interseccao" else \
               "vizinho" if r["origem"].startswith("vizinho") else \
               "quadra" if rua_q else "indefinido"

        novo = dict(r)
        novo.update({"quadra": cod, "rua_quadra": rua_q,
                     "dist_rua_quadra_m": dist_q, "n_ruas_quadra": n_conf,
                     "ruas_quadra": lista_q, "endereco": endereco,
                     "endereco_origem": base})
        saida.append(novo)

    print("[5/5] escrevendo...")
    destino = args.saida or (os.path.splitext(args.rotulos)[0] + "_quadra.csv")
    colunas = list(linhas[0].keys()) + ["quadra", "rua_quadra", "dist_rua_quadra_m",
                                        "n_ruas_quadra", "ruas_quadra",
                                        "endereco", "endereco_origem"]
    with open(destino, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=colunas, delimiter=";")
        w.writeheader()
        w.writerows(saida)

    n = len(saida)
    sem_antes = sum(1 for r in linhas if r["origem"] == "sem_rua")
    indef = sum(1 for r in saida if r["endereco_origem"] == "indefinido")
    print("OK: %d tiles em %.1fs" % (n, time.time() - t0))
    print("    dentro de quadra cadastral: %d (%.1f%%)" % (n_q, 100.0 * n_q / n))
    print("    sem_rua antes:              %d (%.1f%%)" % (sem_antes, 100.0 * sem_antes / n))
    print("    recuperados pela quadra:    %d" % n_recup)
    print("    ainda sem endereco:         %d (%.1f%%)" % (indef, 100.0 * indef / n))
    print("    -> %s" % destino)

    if args.amostra:
        print("\n  exemplos recuperados (eram sem_rua):")
        m = 0
        for r in saida:
            if r["origem"] == "sem_rua" and r["rua_quadra"]:
                print("    z%s/%s/%s  quadra %-10s %-38s %sm" %
                      (r["z"], r["x"], r["y"], r["quadra"], r["rua_quadra"],
                       r["dist_rua_quadra_m"]))
                m += 1
                if m >= args.amostra:
                    break


if __name__ == "__main__":
    main()
