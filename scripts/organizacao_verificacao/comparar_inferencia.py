"""
Compara uma inferência nova (inferir_lote.py rodado de novo, em parte dos patches)
com o deteccoes_lote.csv original: para cada patch e classe, casa as caixas pelo
centro mais próximo (≤ --tol px) e mede a diferença de confiança e de tamanho.

Uso: python comparar_inferencia.py --original A.csv --nova B.csv [--tol 3]
"""
import argparse
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment

ap = argparse.ArgumentParser()
ap.add_argument("--original", required=True); ap.add_argument("--nova", required=True)
ap.add_argument("--tol", type=float, default=3.0)
a = ap.parse_args()
o = pd.read_csv(a.original, sep=";", encoding="utf-8-sig")
n = pd.read_csv(a.nova, sep=";", encoding="utf-8-sig")
pat = set(zip(n.tile_x, n.tile_y)) | set()
alvo = n[["tile_x", "tile_y"]].drop_duplicates()
print(f"patches na inferência nova com ≥1 detecção: {len(alvo)}")
chaves = set(map(tuple, alvo.values))
o = o[[ (x, y) in chaves for x, y in zip(o.tile_x, o.tile_y)]]
print(f"detecções — original nesses patches: {len(o)}   nova: {len(n)}")
cas, dconf, dlado, so_o, so_n = 0, [], [], [], []
for (x, y, c), g_n in n.groupby(["tile_x", "tile_y", "classe"]):
    g_o = o[(o.tile_x == x) & (o.tile_y == y) & (o.classe == c)]
    if len(g_o) == 0:
        so_n += g_n.confianca.tolist(); continue
    D = np.hypot(g_o.px_cx.values[:, None] - g_n.px_cx.values[None], g_o.px_cy.values[:, None] - g_n.px_cy.values[None])
    i, j = linear_sum_assignment(D)
    ok = D[i, j] <= a.tol
    cas += ok.sum()
    dconf += list(np.abs(g_o.confianca.values[i[ok]] - g_n.confianca.values[j[ok]]))
    dlado += list(np.abs(g_o.px_w.values[i[ok]] - g_n.px_w.values[j[ok]]))
    so_o += [v for k, v in enumerate(g_o.confianca.values) if k not in set(i[ok])]
    so_n += [v for k, v in enumerate(g_n.confianca.values) if k not in set(j[ok])]
# original em patches/classes sem nada na nova
for (x, y, c), g_o in o.groupby(["tile_x", "tile_y", "classe"]):
    if not ((n.tile_x == x) & (n.tile_y == y) & (n.classe == c)).any():
        so_o += g_o.confianca.tolist()
print(f"casadas (centro ≤ {a.tol} px): {cas}")
print(f"só no original: {len(so_o)} (confianças: {sorted(round(v,3) for v in so_o)})")
print(f"só na nova:     {len(so_n)} (confianças: {sorted(round(v,3) for v in so_n)})")
if dconf:
    print(f"|Δ confiança| nas casadas: mediana {np.median(dconf):.4f}, máx {max(dconf):.4f}")
    print(f"|Δ largura| (px) nas casadas: mediana {np.median(dlado):.2f}, máx {max(dlado):.2f}")
