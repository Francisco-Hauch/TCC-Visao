"""
Revisão cega (3ª rodada, teste 7): N patches sorteados ao acaso de TODOS os
45.523 (semente fixa), sem condicionar em detecção, excluindo os do dataset
de treino/val e os 400 já revisados. Vão para a pasta SEM pré-rótulo (imagem +
.txt vazio + classes.txt), no formato do rotulador-yolo.html.

As predições do v4 XL (lote filtrado por tamanho, conf >= 0,25, cantos
reconstruídos de px_cx/px_cy/px_w/px_h/angulo_rad) ficam FORA da pasta, em
<dest>_predicoes/, com um manifesto por patch, para comparar depois da
rotulação manual (precisão e recall de verdade; ancoragem vs. revisão de 400).

Uso: python preparar_revisao_cega.py [--n 100] [--semente 20261018]
"""
import argparse
import csv
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path

from filtro_tamanho import CLASSES, LOTE_FILTRADO

RAIZ = Path(__file__).resolve().parent.parent
PATCHES = Path("D:/CityVision/TCC-data/patches512_rotulos_quadra.csv")
DATASET_IMGS = RAIZ / "dataset" / "images"
REVISAO_400 = RAIZ / "revisao_400_xl"
DEST = RAIZ / "revisao_cega_100"
CONF = 0.25


def xy_de_nome(stem):
    return tuple(int(v) for v in stem.rsplit("_", 2)[1:])


def cantos(cx, cy, w, h, ang):
    c, s = math.cos(ang), math.sin(ang)
    return [((cx + dx * c - dy * s) / 512, (cy + dx * s + dy * c) / 512)
            for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--semente", type=int, default=20261018)
    args = ap.parse_args()

    excluir = {xy_de_nome(p.stem) for p in DATASET_IMGS.rglob("*.jpg")}
    n_dataset = len(excluir)
    excluir |= {xy_de_nome(p.stem) for p in REVISAO_400.glob("*.jpg")}
    with open(PATCHES, encoding="utf-8-sig", newline="") as f:
        todos = [r for r in csv.DictReader(f, delimiter=";")]
    candidatos = sorted((r for r in todos if (int(r["x"]), int(r["y"])) not in excluir),
                        key=lambda r: (int(r["x"]), int(r["y"])))
    sorteio = random.Random(args.semente).sample(candidatos, args.n)
    print(f"{len(todos)} patches; excluídos {n_dataset} do dataset e "
          f"{len(excluir) - n_dataset} da revisão de 400; {len(candidatos)} candidatos; "
          f"sorteados {len(sorteio)}")

    escolhidos = {(r["x"], r["y"]): r for r in sorteio}
    pred = defaultdict(list)
    with open(LOTE_FILTRADO, encoding="utf-8-sig", newline="") as f:
        for d in csv.DictReader(f, delimiter=";"):
            k = (d["tile_x"], d["tile_y"])
            if k in escolhidos and float(d["confianca"]) >= CONF:
                pts = cantos(*(float(d[c]) for c in ("px_cx", "px_cy", "px_w", "px_h", "angulo_rad")))
                pred[k].append((int(d["classe_id"]), float(d["confianca"]), pts))

    DEST.mkdir(exist_ok=True)
    dest_pred = DEST.with_name(DEST.name + "_predicoes")
    dest_pred.mkdir(exist_ok=True)
    shutil.copy(RAIZ / "dataset" / "classes.txt", DEST / "classes.txt")
    manifesto = []
    for (x, y), r in sorted(escolhidos.items()):
        nome = f"{r['regional_pasta']}_{x}_{y}"
        shutil.copy(r["arquivo"], DEST / f"{nome}.jpg")
        (DEST / f"{nome}.txt").write_text("", encoding="utf-8")
        linhas = [f"{c} " + " ".join(f"{v:.6f}" for p in pts for v in p) for c, _, pts in pred[(x, y)]]
        (dest_pred / f"{nome}.txt").write_text("\n".join(linhas) + ("\n" if linhas else ""),
                                               encoding="utf-8")
        lin = {"patch": nome, "regional": r["regional_pasta"], "tile_x": x, "tile_y": y,
               "bairro": r["bairro"], "rua_principal": r["rua_principal"] or r["rua_quadra"]}
        for j, c in enumerate(CLASSES):
            confs = [cf for cc, cf, _ in pred[(x, y)] if cc == j]
            lin[f"{c}_n_modelo"] = len(confs)
            lin[f"{c}_conf_max"] = f"{max(confs):.4f}" if confs else ""
        manifesto.append(lin)
    with open(dest_pred / "manifesto.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifesto[0]), delimiter=";")
        w.writeheader()
        w.writerows(manifesto)
    for j, c in enumerate(CLASSES):
        print(f"  modelo: {c:18s} em {sum(m[f'{c}_n_modelo'] > 0 for m in manifesto):3d} dos {args.n}")
    print(f"-> {DEST} (sem pré-rótulo)\n-> {dest_pred} (predições + manifesto.csv)")


if __name__ == "__main__":
    main()
