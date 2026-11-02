"""Compara arquivos regerados (espelho) com os originais do computador do autor.
Idêntico byte a byte -> OK. Senão, compara célula a célula (numérico com tolerância)."""
import hashlib, sys
import numpy as np, pandas as pd
from pathlib import Path
ORIG = Path("/mnt/user-data/uploads/CityVision")
NOVO = Path(sys.argv[1])
arqs = sys.argv[2:]
def md5(p): return hashlib.md5(p.read_bytes()).hexdigest()
for a in arqs:
    o, n = ORIG / a, NOVO / a
    if not o.exists(): print(f"{a}: original não existe"); continue
    if not n.exists(): print(f"{a}: regerado não existe"); continue
    if md5(o) == md5(n):
        print(f"[IDÊNTICO byte a byte] {a}"); continue
    do = pd.read_csv(o, sep=";", encoding="utf-8-sig", low_memory=False)
    dn = pd.read_csv(n, sep=";", encoding="utf-8-sig", low_memory=False)
    print(f"[DIFERE nos bytes] {a}: linhas {len(do)} vs {len(dn)}, colunas {do.shape[1]} vs {dn.shape[1]}")
    if list(do.columns) != list(dn.columns):
        print("   colunas só no original:", [c for c in do.columns if c not in dn.columns])
        print("   colunas só no regerado:", [c for c in dn.columns if c not in do.columns])
    if len(do) != len(dn): continue
    comuns = [c for c in do.columns if c in dn.columns]
    ruins = 0
    for c in comuns:
        x, y = do[c], dn[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            dif = pd.Series(~np.isclose(x.astype(float), y.astype(float), rtol=1e-6, atol=1e-6, equal_nan=True))
        else:
            dif = ~((x.astype(str) == y.astype(str)) | (x.isna() & y.isna()))
        if dif.any():
            ruins += 1
            i = int(np.flatnonzero(dif.values)[0])
            print(f"   coluna {c}: {int(dif.sum())} células diferentes; ex. linha {i}: {x.iloc[i]!r} vs {y.iloc[i]!r}")
    if not ruins: print("   conteúdo igual (diferença só de formatação/bytes)")
