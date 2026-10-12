#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pontuar_patches.py
==================
Para cada patch 512x512 (z19, Patches512) calcula, em coordenada de tile z19:
  - cruzamentos de rua (nó dos eixos_logradouro com >=3 trechos e >=2 ruas distintas)
  - pontos de onibus com cobertura (tubo ou telhado; placa fica de fora)
  - faixas de pedestre OSM (sem as marcadas como 'unmarked' / sem pintura)
  - area de lote fiscal "Territorial" (pseudo-rotulo de terreno baldio)
e a regional oficial (mesma regra de reorganizar_por_regional.py: centro do patch).
Saida: patches_features.csv
"""
import csv, json, math, os, sys, unicodedata
from collections import defaultdict
from shapely.geometry import Polygon, box
from shapely.validation import make_valid

D = sys.argv[1] if len(sys.argv) > 1 else r"D:\CityVision\TCC-data"
Z = 19

def deg2tilef(lat, lon, z=Z):
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 2.0 ** z
    return (lon + 180.0) / 360.0 * n, (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n

def slug(nome):
    n = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode("ascii")
    return n.strip().lower().replace(" ", "_")

ONIBUS_OK = {"Estação tubo": "tubo", "Chapéu chinês": "telhado", "Novo mobiliário": "telhado", "Domus": "telhado"}

# ---------------- patches ----------------
patches = {}
with open(os.path.join(D, "patches_quadra.csv"), encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f, delimiter=";"):
        k = (int(r["px"]), int(r["py"]))
        patches[k] = {"px": k[0], "py": k[1], "n_tiles": int(r["n_tiles"]), "lat": r["lat_centro"], "lon": r["lon_centro"],
                      "bairro": r["bairro"], "endereco": r["endereco"],
                      "n_cruz": 0, "cruz_dist": 9.9, "cruz_ruas": "",
                      "n_tubo": 0, "n_telhado": 0, "onibus_nomes": "",
                      "n_faixa": 0, "n_faixa_forte": 0,
                      "baldio_frac": 0.0, "baldio_maior_frac": 0.0, "n_lotes_territ": 0}
print("patches:", len(patches))

# ---------------- regional ----------------
reg = []
for feat in json.load(open(os.path.join(D, "regionais.geojson"), encoding="utf-8"))["features"]:
    g = feat["geometry"]; polis = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
    pg = [make_valid(Polygon([deg2tilef(c[1], c[0]) for c in p[0]])) for p in polis]
    reg.append((slug(feat["properties"]["nome"]), pg))
from shapely.geometry import Point
from shapely import STRtree
for k, p in patches.items():
    pt = Point(k[0] + 0.5, k[1] + 0.5); p["regional"] = "_sem_regional"
    for s, pg in reg:
        if any(q.contains(pt) for q in pg):
            p["regional"] = s; break

# ---------------- cruzamentos ----------------
nos = defaultdict(lambda: {"deg": 0, "ruas": set(), "nomes": set()})
for feat in json.load(open(os.path.join(D, "eixos_logradouro.geojson"), encoding="utf-8"))["features"]:
    g = feat["geometry"]; pr = feat["properties"]
    linhas = [g["coordinates"]] if g["type"] == "LineString" else g["coordinates"]
    for ln in linhas:
        for c in (ln[0], ln[-1]):
            key = (round(c[0], 5), round(c[1], 5))   # ~1 m de tolerancia
            n = nos[key]; n["deg"] += 1; n["ruas"].add(pr.get("gtm_cod_logradouro")); n["nomes"].add(pr.get("gtm_nm_logradouro") or "")
cruz = [(k, v) for k, v in nos.items() if v["deg"] >= 3 and len(v["ruas"]) >= 2]
print("cruzamentos:", len(cruz))
for (lon, lat), v in cruz:
    x, y = deg2tilef(lat, lon); k = (int(x), int(y))
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            kk = (k[0] + dx, k[1] + dy); p = patches.get(kk)
            if not p: continue
            d = math.hypot(x - (kk[0] + 0.5), y - (kk[1] + 0.5))
            if d < p["cruz_dist"]:
                p["cruz_dist"] = d; p["cruz_ruas"] = " x ".join(sorted(v["nomes"]))[:160]
            fx, fy = x - kk[0], y - kk[1]
            if 0.15 <= fx <= 0.85 and 0.15 <= fy <= 0.85:   # cruzamento bem visivel no patch
                p["n_cruz"] += 1

# ---------------- onibus ----------------
M = 0.05  # margem ~25 px: objeto nao cortado na borda
nomes_on = defaultdict(list)
for feat in json.load(open(os.path.join(D, "VetoresFeatures", "pontos_onibus.geojson"), encoding="utf-8"))["features"]:
    t = ONIBUS_OK.get((feat["properties"] or {}).get("tipo"))
    if not t or feat["geometry"]["type"] != "Point": continue
    lon, lat = feat["geometry"]["coordinates"][:2]; x, y = deg2tilef(lat, lon); k = (int(x), int(y))
    p = patches.get(k)
    if not p or not (M <= x - k[0] <= 1 - M and M <= y - k[1] <= 1 - M): continue
    p["n_" + t] += 1; nomes_on[k].append("%s (%s)" % (feat["properties"].get("nome_ponto") or "", feat["properties"]["tipo"]))
for k, l in nomes_on.items(): patches[k]["onibus_nomes"] = " | ".join(l)[:200]

# ---------------- faixa de pedestre ----------------
for feat in json.load(open(os.path.join(D, "VetoresFeatures", "faixa_pedestre_osm.geojson"), encoding="utf-8"))["features"]:
    pr = feat["properties"]; g = feat["geometry"]
    cr, mk = pr.get("crossing"), pr.get("crossing:markings")
    if cr == "unmarked" or mk == "no": continue
    forte = cr in ("marked", "zebra", "traffic_signals", "uncontrolled") or mk in ("yes", "zebra")
    if g["type"] == "Point":
        lon, lat = g["coordinates"][:2]
    else:
        cs = g["coordinates"]; lon = sum(c[0] for c in cs) / len(cs); lat = sum(c[1] for c in cs) / len(cs)
    x, y = deg2tilef(lat, lon); k = (int(x), int(y)); p = patches.get(k)
    if not p or not (M <= x - k[0] <= 1 - M and M <= y - k[1] <= 1 - M): continue
    p["n_faixa"] += 1; p["n_faixa_forte"] += int(forte)

# ---------------- baldio (lote Territorial) ----------------
fc = json.load(open(os.path.join(D, "VetoresFeatures", "lotes_cadastrais.geojson"), encoding="utf-8"))
nt = 0
for feat in fc["features"]:
    if (feat["properties"] or {}).get("gtm_desc_natureza") != "Territorial": continue
    g = feat["geometry"]
    if not g: continue
    polis = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
    for poli in polis:
        try:
            pg = make_valid(Polygon([deg2tilef(c[1], c[0]) for c in poli[0]], [[deg2tilef(c[1], c[0]) for c in h] for h in poli[1:]]))
        except Exception:
            continue
        if pg.area > 3.15: continue   # > ~1,5 ha: praca/terminal/faixa de dominio, nao baldio visual
        x0, y0, x1, y1 = pg.bounds
        for px in range(int(x0), int(x1) + 1):
            for py in range(int(y0), int(y1) + 1):
                p = patches.get((px, py))
                if not p: continue
                a = pg.intersection(box(px, py, px + 1, py + 1)).area
                if a > 1e-4:
                    p["baldio_frac"] += a; p["n_lotes_territ"] += 1
                    p["baldio_maior_frac"] = max(p["baldio_maior_frac"], a)
    nt += 1
del fc
print("lotes territoriais:", nt)

cols = list(next(iter(patches.values())).keys())
with open("patches_features.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=cols, delimiter=";"); w.writeheader()
    for p in patches.values():
        q = dict(p); q["baldio_frac"] = "%.4f" % min(1, p["baldio_frac"]); q["baldio_maior_frac"] = "%.4f" % p["baldio_maior_frac"]; q["cruz_dist"] = "%.3f" % p["cruz_dist"]
        w.writerow(q)
print("ok")
