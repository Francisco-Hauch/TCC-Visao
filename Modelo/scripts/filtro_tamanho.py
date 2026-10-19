"""
Filtro de tamanho mínimo por classe (decisão de 2026-10-18).

Uma detecção do modelo é descartada se o lado menor OU o lado maior da caixa
girada for menor que o menor lado correspondente entre as caixas rotuladas à
mão no dataset de treino (348 imagens, train+val, Modelo/dataset/labels).
Medidas em px do patch 512 (~0,135 m/px).

Os mínimos estão FIXOS aqui (medidos em 2026-10-18, arredondados para baixo),
de propósito: se o dataset crescer (ex. entrar a revisão de 400), o filtro
não muda sozinho. Para recalcular: python filtro_tamanho.py

Uso como módulo:
  from filtro_tamanho import passa
  passa("ponto_onibus", largura_px, altura_px)  -> True se fica

Uso como script:
  python filtro_tamanho.py            # mostra os mínimos atuais do dataset
  python filtro_tamanho.py --lote     # grava deteccoes_lote_filtrado.csv
"""
import argparse
import csv
import math
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
LABELS = RAIZ / "dataset" / "labels"
LOTE = RAIZ / "runs" / "yolo26x_obb_v4_xl" / "deteccoes_lote.csv"
LOTE_FILTRADO = LOTE.with_name("deteccoes_lote_filtrado.csv")

CLASSES = ["area_NaoEdificada", "ponto_onibus", "faixa_pedestre"]  # id 0, 1, 2
PATCH_PX = 512

# (lado menor mínimo, lado maior mínimo) em px — mínimos do treino, 2026-10-18
MINIMO = {
    "area_NaoEdificada": (21.2, 25.6),
    "ponto_onibus": (6.5, 15.0),
    "faixa_pedestre": (5.0, 18.6),
}


def passa(classe, w, h):
    """True se a caixa (w, h em px do patch 512) tem pelo menos o mínimo da classe."""
    menor_min, maior_min = MINIMO[classe]
    return min(w, h) >= menor_min and max(w, h) >= maior_min


def lados_obb(valores):
    """8 coordenadas normalizadas (4 cantos, formato do .txt) -> (lado1, lado2) em px."""
    p = [(valores[i] * PATCH_PX, valores[i + 1] * PATCH_PX) for i in range(0, 8, 2)]
    return math.dist(p[0], p[1]), math.dist(p[1], p[2])


def minimos_dataset():
    m = {c: [math.inf, math.inf] for c in CLASSES}
    for f in LABELS.rglob("*.txt"):
        for linha in f.read_text(encoding="utf-8").split("\n"):
            v = linha.split()
            if len(v) != 9:
                continue
            a, b = lados_obb(list(map(float, v[1:])))
            c = CLASSES[int(v[0])]
            m[c][0] = min(m[c][0], min(a, b))
            m[c][1] = min(m[c][1], max(a, b))
    return m


def filtrar_lote():
    removidas, total = Counter(), Counter()
    with open(LOTE, encoding="utf-8-sig", newline="") as f, \
            open(LOTE_FILTRADO, "w", encoding="utf-8-sig", newline="") as g:
        r = csv.DictReader(f, delimiter=";")
        w = csv.DictWriter(g, fieldnames=r.fieldnames, delimiter=";")
        w.writeheader()
        for row in r:
            c = row["classe"]
            total[c] += 1
            if passa(c, float(row["px_w"]), float(row["px_h"])):
                w.writerow(row)
            else:
                removidas[c] += 1
    for c in CLASSES:
        print(f"  {c:18s} {removidas[c]:5d} de {total[c]:6d} removidas ({removidas[c] / total[c]:.2%})")
    print(f"-> {LOTE_FILTRADO}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lote", action="store_true")
    args = ap.parse_args()
    print("mínimos fixos no filtro (lado menor x lado maior, px):")
    for c, (a, b) in MINIMO.items():
        print(f"  {c:18s} {a:5.1f} x {b:5.1f}")
    print("mínimos medidos agora no dataset:")
    for c, (a, b) in minimos_dataset().items():
        print(f"  {c:18s} {a:5.1f} x {b:5.1f}")
    if args.lote:
        filtrar_lote()


if __name__ == "__main__":
    main()
