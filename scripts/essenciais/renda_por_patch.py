"""
Renda por patch: localiza o centro de cada patch z19 dentro de um setor
censitário de 2022 (setor_censitario_2022.geojson, campo renda_media) e
grava a renda daquele setor. Ponto-em-polígono à mão (ray casting com
filtro de bbox), sem shapely.

Entrada: TCC-data/patches512_rotulos_quadra.csv (lat_centro/lon_centro)
Saída:   TCC-data/patches512_renda.csv  (x; y; cd_setor; renda_media)
Patch fora de qualquer setor ou em setor sem renda fica com renda vazia.

Uso: python renda_por_patch.py
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

DADOS = Path("D:/CityVision/TCC-data")
PATCHES = DADOS / "patches512_rotulos_quadra.csv"
SETORES = DADOS / "VetoresFeatures" / "setor_censitario_2022.geojson"
SAIDA = DADOS / "patches512_renda.csv"
CELULA = 0.01  # grade de ~1 km para achar os setores candidatos


def aneis(geom):
    """Lista de polígonos, cada um uma lista de anéis [(lon, lat), ...]."""
    if geom["type"] == "Polygon":
        return [geom["coordinates"]]
    return geom["coordinates"]  # MultiPolygon


def dentro_anel(lon, lat, anel):
    dentro = False
    j = len(anel) - 1
    for i in range(len(anel)):
        xi, yi = anel[i][0], anel[i][1]
        xj, yj = anel[j][0], anel[j][1]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi) + xi:
            dentro = not dentro
        j = i
    return dentro


def dentro(lon, lat, poligonos):
    for anel_ext, *buracos in poligonos:
        if dentro_anel(lon, lat, anel_ext) and not any(dentro_anel(lon, lat, b) for b in buracos):
            return True
    return False


def main():
    setores = []
    grade = defaultdict(list)
    for ft in json.load(open(SETORES, encoding="utf-8"))["features"]:
        if not ft["geometry"]:
            continue
        pols = aneis(ft["geometry"])
        xs = [p[0] for pol in pols for p in pol[0]]
        ys = [p[1] for pol in pols for p in pol[0]]
        idx = len(setores)
        setores.append((ft["properties"]["cd_setor"], ft["properties"]["renda_media"], pols))
        for gx in range(int(min(xs) // CELULA), int(max(xs) // CELULA) + 1):
            for gy in range(int(min(ys) // CELULA), int(max(ys) // CELULA) + 1):
                grade[(gx, gy)].append(idx)

    n = achou = com_renda = 0
    with open(PATCHES, encoding="utf-8-sig", newline="") as f, \
            open(SAIDA, "w", encoding="utf-8-sig", newline="") as g:
        w = csv.writer(g, delimiter=";")
        w.writerow(["x", "y", "cd_setor", "renda_media"])
        for row in csv.DictReader(f, delimiter=";"):
            n += 1
            lat, lon = float(row["lat_centro"]), float(row["lon_centro"])
            cd, renda = "", ""
            for idx in grade.get((int(lon // CELULA), int(lat // CELULA)), ()):
                if dentro(lon, lat, setores[idx][2]):
                    cd, r = setores[idx][0], setores[idx][1]
                    renda = "" if r is None else r
                    achou += 1
                    com_renda += r is not None
                    break
            w.writerow([row["x"], row["y"], cd, renda])
    print(f"{n} patches: {achou} dentro de setor ({achou / n:.1%}), {com_renda} com renda")
    print(f"-> {SAIDA}")


if __name__ == "__main__":
    main()
