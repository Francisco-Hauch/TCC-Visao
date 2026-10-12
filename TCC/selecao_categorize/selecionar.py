#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Seleciona 150 patches ancorados em cruzamento de ruas, balanceando
onibus (tubo/telhado), faixa de pedestre e terreno baldio (lote Territorial)."""
import csv, random, collections, json

N = 150
META = {"onibus": 75, "tubo": 30, "faixa": 75, "baldio": 75}
CAP_REGIONAL = 32
DIST_MIN = 3          # em patches (~200 m): evita escolher vizinhos quase iguais
random.seed(19)

rows = list(csv.DictReader(open("patches_features.csv", encoding="utf-8-sig"), delimiter=";"))
def classes(r):
    c = set()
    if int(r["n_tubo"]): c |= {"tubo", "onibus"}
    if int(r["n_telhado"]): c.add("onibus")
    if int(r["n_faixa_forte"]): c.add("faixa")   # so faixa com pintura/semaforo confirmada no OSM
    if float(r["baldio_maior_frac"]) >= 0.10: c.add("baldio")   # lote territorial >= ~475 m2 no patch
    return c
cand = []
for r in rows:
    if r["n_tiles"] != "4" or r["regional"] == "_sem_regional": continue
    c = classes(r)
    if not c: continue
    no_cruz = int(r["n_cruz"]) > 0
    perto = float(r["cruz_dist"]) <= 1.0            # cruzamento a ate ~70 m do centro
    if not (no_cruz or ("tubo" in c and perto)): continue
    r["_c"] = c; r["_k"] = (int(r["px"]), int(r["py"])); r["_rand"] = random.random()
    cand.append(r)
print("candidatos:", len(cand), collections.Counter(x for r in cand for x in r["_c"]))

sel, cnt, creg = [], collections.Counter(), collections.Counter()
def ok(r):
    if creg[r["regional"]] >= CAP_REGIONAL: return False
    return all(max(abs(r["_k"][0] - s["_k"][0]), abs(r["_k"][1] - s["_k"][1])) >= DIST_MIN for s in sel)
def ganho(r):
    g = 0.0
    for c in r["_c"]:
        falta = META[c] - cnt[c]
        g += 2.0 if falta > 0 else -0.25
    g += 0.3 * (len(r["_c"]) - 1)                  # co-ocorrencia (imagem com varias classes rende mais)
    g += 0.2 * min(int(r["n_faixa_forte"]), 2)
    g -= 0.02 * creg[r["regional"]]                 # espalha pelas regionais
    return g + 0.05 * r["_rand"]
while len(sel) < N:
    best = max((r for r in cand if r not in sel and ok(r)), key=ganho, default=None)
    if best is None: break
    sel.append(best); creg[best["regional"]] += 1
    for c in best["_c"]: cnt[c] += 1
print("selecionados:", len(sel), dict(cnt)); print(dict(creg))
json.dump([{k: v for k, v in r.items() if not k.startswith("_")} | {"classes": sorted(r["_c"])} for r in sel],
          open("selecao.json", "w"), ensure_ascii=False, indent=0)
