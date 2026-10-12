#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Monta a pasta categorize/: 150 jpgs planos + classes.txt + gabarito_vetor.csv
com a posicao (em pixel do patch 512) de cada objeto esperado pelo cadastro."""
import csv, json, math, os, shutil, sys
from shapely.geometry import Polygon, box
from shapely.validation import make_valid

D = sys.argv[1]; OUT = sys.argv[2]; Z = 19
os.makedirs(OUT, exist_ok=True)
def t(lat, lon):
    n = 2.0 ** Z
    return (lon + 180) / 360 * n, (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
sel = json.load(open("selecao.json"))
S = {(int(r["px"]), int(r["py"])): r for r in sel}
obj = {k: [] for k in S}
def px_of(k, x, y): return int(round((x - k[0]) * 512)), int(round((y - k[1]) * 512))

ONI = {"Estação tubo", "Chapéu chinês", "Novo mobiliário", "Domus"}
for f in json.load(open(os.path.join(D, "VetoresFeatures/pontos_onibus.geojson")))["features"]:
    tp = f["properties"].get("tipo")
    if tp not in ONI: continue
    lon, lat = f["geometry"]["coordinates"][:2]; x, y = t(lat, lon); k = (int(x), int(y))
    if k in S: obj[k].append(("onibus", tp, px_of(k, x, y)))
for f in json.load(open(os.path.join(D, "VetoresFeatures/faixa_pedestre_osm.geojson")))["features"]:
    p = f["properties"]
    if not (p.get("crossing") in ("marked", "zebra", "traffic_signals", "uncontrolled") or p.get("crossing:markings") in ("yes", "zebra")): continue
    g = f["geometry"]
    cs = [g["coordinates"]] if g["type"] == "Point" else g["coordinates"]
    lon = sum(c[0] for c in cs) / len(cs); lat = sum(c[1] for c in cs) / len(cs)
    x, y = t(lat, lon); k = (int(x), int(y))
    if k in S: obj[k].append(("faixa", p.get("crossing") or "?", px_of(k, x, y), [px_of(k, *t(c[1], c[0])) for c in cs]))
for f in json.load(open(os.path.join(D, "VetoresFeatures/lotes_cadastrais.geojson")))["features"]:
    if (f["properties"] or {}).get("gtm_desc_natureza") != "Territorial" or not f["geometry"]: continue
    g = f["geometry"]; polis = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
    for poli in polis:
        pts = [t(c[1], c[0]) for c in poli[0]]
        if len(pts) < 3 or make_valid(Polygon(pts)).area > 3.15: continue
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        for kx in range(int(min(xs)), int(max(xs)) + 1):
            for ky in range(int(min(ys)), int(max(ys)) + 1):
                k = (kx, ky)
                if k not in S: continue
                pg = make_valid(Polygon(pts)); inter = pg.intersection(box(kx, ky, kx + 1, ky + 1))
                if inter.area < 0.02: continue   # pedaco < ~95 m2 no patch: ignora
                c = inter.centroid; b = inter.bounds
                obj[k].append(("baldio", "%d m2" % round(inter.area * 4761), px_of(k, c.x, c.y),
                               [px_of(k, b[0], b[1]), px_of(k, b[2], b[3])], [px_of(k, *p) for p in pts]))
json.dump({"%d_%d" % k: v for k, v in obj.items()}, open("objetos.json", "w"))

linhas = []
for i, r in enumerate(sel, 1):
    k = (int(r["px"]), int(r["py"]))
    nome = "%s_%d_%d.jpg" % (r["regional"], k[0], k[1])
    src = os.path.join(D, "Patches512", r["regional"], "19", str(k[0]), "%d.jpg" % k[1])
    shutil.copy2(src, os.path.join(OUT, nome))
    o = obj[k]
    def dicas(tipo):
        return " ; ".join("%s @(%d,%d)" % (a[1], a[2][0], a[2][1]) for a in o if a[0] == tipo)
    linhas.append({"n": i, "arquivo": nome, "regional": r["regional"], "bairro": r["bairro"],
                   "lat": r["lat"], "lon": r["lon"], "cruzamento": r["cruz_ruas"],
                   "esperado": " + ".join(c for c in ("onibus", "faixa", "baldio") if c in r["classes"]) + (" (tubo)" if "tubo" in r["classes"] else ""),
                   "onibus_px": dicas("onibus"), "faixa_px": dicas("faixa"), "baldio_px": dicas("baldio"),
                   "onibus_nomes": r["onibus_nomes"], "baldio_area_frac": r["baldio_frac"],
                   "origem": "Patches512\\%s\\19\\%d\\%d.jpg" % (r["regional"], k[0], k[1])})
with open(os.path.join(OUT, "gabarito_vetor.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()), delimiter=";"); w.writeheader(); w.writerows(linhas)
open(os.path.join(OUT, "classes.txt"), "w").write("terreno_baldio\nponto_onibus\nfaixa_pedestre\n")
print(len(linhas), "imagens em", OUT)
