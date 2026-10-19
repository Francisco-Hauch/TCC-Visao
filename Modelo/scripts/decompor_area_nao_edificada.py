"""
Decomposição exploratória das caixas `area_NaoEdificada` do v4 XL usando só
o que já está em disco (sem camada 64 "maciço arbóreo", que não foi baixada):

  1. Cadastro fiscal (VetoresFeatures/lotes_cadastrais.geojson): que fração
     da caixa cai em lote "Territorial" (só terreno — candidato a baldio),
     em lote "Predial" (tem construção tributada), em lote sem natureza, e
     fora de qualquer lote (rua, praça, parque, APP — candidato a mata/área
     verde pública).
  2. Imagem (o próprio patch RGB): fração de pixels de vegetação dentro da
     caixa pelo índice ExG em coordenadas cromáticas (ExG = 2g - r - b,
     Woebbecke et al. 1995), brilho médio e textura (desvio-padrão do cinza
     nos pixels vegetados — copa de mata é escura e rugosa, gramado/capim é
     claro e liso). RGB sozinho NÃO separa mata de capim alto com certeza;
     é indício para cruzar com o item 1, não classificador.

Não é rótulo novo nem decisão de classe (CLAUDE.md: não reabrir a
rotulagem). É diagnóstico para entender os 41,7% de patches com
area_NaoEdificada e se dá para separar mata/baldio como VARIÁVEL analítica.

Entrada:  TCC-data/deteccoes_objetos_rua_nms.csv  (NMS IoU 0,5)
Saída:    TCC-data/area_nao_edificada_decomposicao.csv  (`;`, UTF-8 BOM)
          + resumo por faixa de confiança no terminal.

Uso:  python decompor_area_nao_edificada.py [--limite N] [--exg 0.05]
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely import STRtree
from shapely.geometry import Polygon, shape
from shapely.ops import transform

DADOS = Path("D:/CityVision/TCC-data")
DETECCOES = DADOS / "deteccoes_objetos_rua_nms.csv"
LOTES = DADOS / "VetoresFeatures" / "lotes_cadastrais.geojson"
SAIDA = DADOS / "area_nao_edificada_decomposicao.csv"

CLASSE = "area_NaoEdificada"
LAT0 = -25.45  # Curitiba — projeção equiretangular local, erro < 1% em área
KX = 111320.0 * math.cos(math.radians(LAT0))
KY = 110540.0
FAIXAS = [0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]


def para_metros(lon, lat, z=None):
    return np.asarray(lon) * KX, np.asarray(lat) * KY


def carregar_lotes():
    with open(LOTES, encoding="utf-8") as f:
        fc = json.load(f)
    geoms, nat = [], []
    for ft in fc["features"]:
        if not ft.get("geometry"):
            continue
        g = transform(para_metros, shape(ft["geometry"]))
        if not g.is_valid:
            g = g.buffer(0)
        if g.is_empty:
            continue
        geoms.append(g)
        nat.append((ft["properties"].get("gtm_desc_natureza") or "").strip() or "sem_natureza")
    return geoms, nat


def mascara_caixa(d):
    cx, cy, w, h, a = (float(d[k]) for k in ("px_cx", "px_cy", "px_w", "px_h", "angulo_rad"))
    c, s = math.cos(a), math.sin(a)
    v1 = (w / 2 * c, w / 2 * s)
    v2 = (-h / 2 * s, h / 2 * c)
    cantos = [(cx + v1[0] + v2[0], cy + v1[1] + v2[1]), (cx + v1[0] - v2[0], cy + v1[1] - v2[1]),
              (cx - v1[0] - v2[0], cy - v1[1] - v2[1]), (cx - v1[0] + v2[0], cy - v1[1] + v2[1])]
    m = Image.new("L", (512, 512), 0)
    ImageDraw.Draw(m).polygon(cantos, fill=1)
    return np.asarray(m, dtype=bool)


def medidas_imagem(img, mask, limiar_exg):
    px = img[mask]
    if len(px) == 0:
        return None
    soma = px.sum(axis=1, keepdims=True) + 1e-6
    r, g, b = (px / soma).T
    exg = 2 * g - r - b
    veg = exg > limiar_exg
    cinza = px.mean(axis=1)
    return {
        "veg_frac": float(veg.mean()),
        "exg_medio": float(exg.mean()),
        "brilho_medio": float(cinza.mean()),
        "textura_veg": float(cinza[veg].std()) if veg.sum() > 30 else "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int, default=None)
    ap.add_argument("--exg", type=float, default=0.05, help="limiar ExG cromático p/ vegetação")
    args = ap.parse_args()

    with open(DETECCOES, encoding="utf-8-sig", newline="") as f:
        dets = [d for d in csv.DictReader(f, delimiter=";") if d["classe"] == CLASSE]
    if args.limite:
        dets = dets[: args.limite]
    print(f"{len(dets)} caixas {CLASSE}")

    print("carregando lotes...")
    geoms, nat = carregar_lotes()
    arvore = STRtree(geoms)
    print(f"{len(geoms)} lotes; naturezas: "
          + ", ".join(f"{k}={v}" for k, v in sorted(_contar(nat).items(), key=lambda kv: -kv[1])))

    por_arquivo = defaultdict(list)
    for d in dets:
        por_arquivo[d["arquivo"]].append(d)

    linhas = []
    for i, (arq, lst) in enumerate(por_arquivo.items()):
        if i % 2000 == 0:
            print(f"  {i}/{len(por_arquivo)} patches")
        try:
            img = np.asarray(Image.open(arq).convert("RGB"), dtype=np.float32)
        except OSError:
            img = None
        for d in lst:
            cantos = [(float(d[f"canto{k}_lon"]), float(d[f"canto{k}_lat"])) for k in range(1, 5)]
            caixa = transform(para_metros, Polygon(cantos))
            area = caixa.area
            frac = defaultdict(float)
            for j in arvore.query(caixa, predicate="intersects"):
                frac[nat[j]] += caixa.intersection(geoms[j]).area / area
            em_lote = sum(frac.values())
            out = {
                "id_objeto": d["id_objeto"], "regional_pasta": d["regional_pasta"],
                "bairro": d["bairro"], "confianca": d["confianca"], "area_m2": round(area),
                "frac_territorial": round(frac.get("Territorial", 0.0), 3),
                "frac_predial": round(frac.get("Predial", 0.0), 3),
                "frac_lote_outro": round(em_lote - frac.get("Territorial", 0.0)
                                         - frac.get("Predial", 0.0), 3),
                "frac_fora_lote": round(max(0.0, 1 - em_lote), 3),
            }
            m = medidas_imagem(img, mascara_caixa(d), args.exg) if img is not None else None
            out.update({k: (round(v, 3) if isinstance(v, float) else v)
                        for k, v in (m or {"veg_frac": "", "exg_medio": "",
                                           "brilho_medio": "", "textura_veg": ""}).items()})
            linhas.append(out)

    with open(SAIDA, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]), delimiter=";")
        w.writeheader()
        w.writerows(linhas)
    print(f"salvo: {SAIDA} ({len(linhas)} linhas)\n")
    resumo(linhas)


def _contar(xs):
    c = defaultdict(int)
    for x in xs:
        c[x] += 1
    return c


def resumo(linhas):
    cols = ["frac_territorial", "frac_predial", "frac_lote_outro", "frac_fora_lote", "veg_frac"]
    print("Médias por faixa de confiança (fração da área da caixa):")
    print(f"{'faixa':>11} {'n':>6} " + " ".join(f"{c.replace('frac_', ''):>12}" for c in cols))
    for lo, hi in zip(FAIXAS, FAIXAS[1:]):
        sel = [l for l in linhas if lo <= float(l["confianca"]) < hi]
        if not sel:
            continue
        med = [np.mean([float(l[c]) for l in sel if l[c] != ""]) for c in cols]
        print(f"{lo:.2f}-{min(hi, 1):.2f} {len(sel):>6} " + " ".join(f"{m:>12.3f}" for m in med))

    print("\nTipologia indicativa (conf >= 0,5; maioria da área da caixa):")
    sel = [l for l in linhas if float(l["confianca"]) >= 0.5 and l["veg_frac"] != ""]
    tipos = defaultdict(int)
    for l in sel:
        veg = float(l["veg_frac"]) >= 0.5
        if float(l["frac_fora_lote"]) >= 0.5:
            t = "fora de lote (+veg: área verde pública/mata)" if veg else "fora de lote (sem veg: via/pátio?)"
        elif float(l["frac_territorial"]) >= 0.5:
            t = "lote Territorial (+veg: baldio c/ mato)" if veg else "lote Territorial (sem veg: solo exposto)"
        elif float(l["frac_predial"]) >= 0.5:
            t = "lote Predial (+veg: quintal/mata em lote)" if veg else "lote Predial (sem veg: suspeita FP)"
        else:
            t = "misto"
        tipos[t] += 1
    for t, n in sorted(tipos.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>6} ({n / len(sel):5.1%})  {t}")


if __name__ == "__main__":
    main()
