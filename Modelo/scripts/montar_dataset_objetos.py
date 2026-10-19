"""
Dataset objeto x patch x rua: uma linha por detecção do modelo (cada
caixa OBB), exatamente como o modelo entregou (deteccoes_lote.csv, conf
>= 0,25 — sem filtro extra; a coluna `confianca` fica para filtrar depois),
com as informações do patch onde ela caiu (rua, quadra, bairro, regional,
coordenada do patch, de patches512_rotulos_quadra.csv) e as coordenadas
da própria caixa: centro e os 4 cantos em lat/lon.

Saídas (TCC-data/):
  deteccoes_objetos_rua.csv      — tabela, `;`, UTF-8 BOM
  deteccoes_objetos_rua.geojson  — mesmas linhas, caixa como polígono
                                   (sobrepor na ortofoto no QGIS)

ATENÇÃO (CLAUDE.md §7-b): rua/quadra aqui são as do PATCH (~69 m), não as
do objeto. O objeto pode estar em outra rua (esquina, miolo de quadra).

NMS opcional (--nms-iou): o modelo já sai com NMS interno em IoU 0,7 (nenhum
par acima disso sobra), então aqui é um NMS extra, mais estrito, com caixa
girada (IoU dos polígonos via shapely), por patch e por classe — a de maior
confiança fica. --nms-contencao também remove a caixa cuja área está
majoritariamente DENTRO de outra maior/mais confiante (interseção / área da
menor), caso que o IoU não pega. Com NMS, a saída ganha sufixo _nms.

Uso:  python montar_dataset_objetos.py                  # bruto
      python montar_dataset_objetos.py --nms-iou 0.5     # NMS extra
      python montar_dataset_objetos.py --nms-iou 0.5 --nms-contencao 0.7
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from shapely.geometry import Polygon

RAIZ = Path(__file__).resolve().parent.parent  # D:\CityVision\Modelo
DETECCOES = RAIZ / "runs" / "yolo26x_obb_v4_xl" / "deteccoes_lote.csv"
PATCHES = Path("D:/CityVision/TCC-data/patches512_rotulos_quadra.csv")
SAIDA_CSV = Path("D:/CityVision/TCC-data/deteccoes_objetos_rua.csv")
SAIDA_GEOJSON = SAIDA_CSV.with_suffix(".geojson")

Z = 19
PX = 512  # lado do patch em pixel

COLS_PATCH = ["regional_pasta", "bairro", "rua_principal", "origem", "quadra",
              "rua_quadra", "dist_rua_quadra_m", "endereco", "endereco_origem"]


# mesma tile2deg() de TCC/rotular_tiles.py
def tile2deg(x: float, y: float, z: int) -> tuple[float, float]:
    n = 2.0 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


def cantos_px(cx, cy, w, h, ang):
    """4 cantos da caixa OBB em pixel (mesma conta do xywhr->xyxyxyxy do Ultralytics)."""
    c, s = math.cos(ang), math.sin(ang)
    v1 = (w / 2 * c, w / 2 * s)
    v2 = (-h / 2 * s, h / 2 * c)
    return [(cx + v1[0] + v2[0], cy + v1[1] + v2[1]),
            (cx + v1[0] - v2[0], cy + v1[1] - v2[1]),
            (cx - v1[0] - v2[0], cy - v1[1] - v2[1]),
            (cx - v1[0] + v2[0], cy - v1[1] + v2[1])]


def nms(dets, iou_max, cont_max):
    """NMS guloso por (patch, classe), maior confiança primeiro."""
    grupos = defaultdict(list)
    for d in dets:
        grupos[(d["tile_x"], d["tile_y"], d["classe"])].append(d)
    mantidas = []
    for lst in grupos.values():
        lst.sort(key=lambda d: -float(d["confianca"]))
        keep = []
        for d in lst:
            pol = Polygon(cantos_px(*(float(d[k]) for k in
                                      ("px_cx", "px_cy", "px_w", "px_h", "angulo_rad"))))
            ok = True
            for q in keep:
                inter = pol.intersection(q).area
                if inter / (pol.area + q.area - inter) > iou_max:
                    ok = False
                    break
                if cont_max is not None and inter / min(pol.area, q.area) > cont_max:
                    ok = False
                    break
            if ok:
                keep.append(pol)
                mantidas.append(d)
    return mantidas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nms-iou", type=float, default=None,
                    help="remove caixa da mesma classe/patch com IoU > N com outra mais confiante")
    ap.add_argument("--nms-contencao", type=float, default=None,
                    help="(com --nms-iou) remove também se interseção/área da menor > N")
    args = ap.parse_args()
    saida_csv = SAIDA_CSV
    if args.nms_iou is not None:
        saida_csv = SAIDA_CSV.with_name(SAIDA_CSV.stem + "_nms.csv")
    saida_geojson = saida_csv.with_suffix(".geojson")

    with open(PATCHES, encoding="utf-8-sig", newline="") as f:
        patches = {(p["x"], p["y"]): p for p in csv.DictReader(f, delimiter=";")}

    with open(DETECCOES, encoding="utf-8-sig", newline="") as f:
        dets = list(csv.DictReader(f, delimiter=";"))
    n_bruto = len(dets)
    if args.nms_iou is not None:
        dets = nms(dets, args.nms_iou, args.nms_contencao)
        print(f"NMS (IoU>{args.nms_iou}, contenção>{args.nms_contencao}): "
              f"{n_bruto} -> {len(dets)} ({n_bruto - len(dets)} removidas)")

    # ordem estável: por patch, e dentro do patch por classe/confiança
    dets.sort(key=lambda d: (d["regional"], int(d["tile_x"]), int(d["tile_y"]),
                             d["classe"], -float(d["confianca"])))
    seq = defaultdict(int)       # (x, y, classe) -> nº do objeto dentro do patch
    total_cls = defaultdict(int)  # (x, y, classe) -> total no patch
    for d in dets:
        total_cls[(d["tile_x"], d["tile_y"], d["classe"])] += 1

    linhas, feats = [], []
    sem_patch = 0
    for d in dets:
        tx, ty = int(d["tile_x"]), int(d["tile_y"])
        p = patches.get((d["tile_x"], d["tile_y"]))
        if p is None:
            sem_patch += 1
            p = {}
        chave = (d["tile_x"], d["tile_y"], d["classe"])
        seq[chave] += 1

        lat_p, lon_p = tile2deg(tx + 0.5, ty + 0.5, Z)
        cx, cy = float(d["px_cx"]), float(d["px_cy"])
        w, h, ang = float(d["px_w"]), float(d["px_h"]), float(d["angulo_rad"])
        lat_c, lon_c = tile2deg(tx + cx / PX, ty + cy / PX, Z)
        cantos = [tile2deg(tx + x / PX, ty + y / PX, Z) for x, y in cantos_px(cx, cy, w, h, ang)]
        m_px = 40075016.685578488 / (2 ** Z) / PX * math.cos(math.radians(lat_c))

        id_obj = f"{tx}_{ty}_{d['classe']}_{seq[chave]}"
        linha = {
            "id_objeto": id_obj,
            "patch_x": tx, "patch_y": ty,
            "patch_lat": f"{lat_p:.7f}", "patch_lon": f"{lon_p:.7f}",
            **{c: p.get(c, "") for c in COLS_PATCH},
            "classe": d["classe"],
            "n_obj_classe_no_patch": seq[chave],
            "total_classe_no_patch": total_cls[chave],
            "confianca": d["confianca"],
            "obj_lat": f"{lat_c:.7f}", "obj_lon": f"{lon_c:.7f}",
        }
        for i, (la, lo) in enumerate(cantos, 1):
            linha[f"canto{i}_lat"] = f"{la:.7f}"
            linha[f"canto{i}_lon"] = f"{lo:.7f}"
        linha.update({
            "largura_m": f"{w * m_px:.1f}", "altura_m": f"{h * m_px:.1f}",
            "area_m2": f"{w * h * m_px * m_px:.0f}",
            "px_cx": d["px_cx"], "px_cy": d["px_cy"], "px_w": d["px_w"],
            "px_h": d["px_h"], "angulo_rad": d["angulo_rad"],
            "arquivo": d["arquivo"],
        })
        linhas.append(linha)

        anel = [[round(lo, 7), round(la, 7)] for la, lo in cantos]
        anel.append(anel[0])
        feats.append({"type": "Feature",
                      "geometry": {"type": "Polygon", "coordinates": [anel]},
                      "properties": {k: v for k, v in linha.items()
                                     if not k.startswith("canto")}})

    with open(saida_csv, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(linhas[0].keys()), delimiter=";")
        wr.writeheader()
        wr.writerows(linhas)
    with open(saida_geojson, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False)

    print(f"{len(linhas)} objetos; sem patch correspondente: {sem_patch}")
    print(f"-> {saida_csv}")
    print(f"-> {saida_geojson}")


if __name__ == "__main__":
    main()
