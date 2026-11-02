"""
Confere se cada imagem do dataset (Modelo/dataset/images/<split>/<regional>_<x>_<y>.jpg)
é o mesmo recorte que o patch TCC-data/Patches512/<regional>/19/<x>/<y>.jpg usado na
inferência em lote: MD5 dos bytes, MD5 dos pixels e diferença média absoluta por pixel
(0-255). Patch ausente na pasta é reportado.

Uso: python conferir_dataset_patches.py [--split val] [--patches CAMINHO]
"""
import argparse, hashlib, re
from pathlib import Path
import numpy as np
from PIL import Image

RAIZ = Path(__file__).resolve().parents[2] / "Modelo"
ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val")
ap.add_argument("--patches", default=str(RAIZ.parent / "TCC-data" / "Patches512"))
a = ap.parse_args()
iguais_bytes = iguais_pix = 0; difs = []; faltam = []
imgs = sorted((RAIZ / "dataset" / "images" / a.split).glob("*.jpg"))
for f in imgs:
    reg, x, y = re.match(r"(.+)_(\d+)_(\d+)$", f.stem).groups()
    p = Path(a.patches) / reg / "19" / x / f"{y}.jpg"
    if not p.exists():
        faltam.append(f.stem); continue
    ba, bb = f.read_bytes(), p.read_bytes()
    A = np.asarray(Image.open(f).convert("RGB"), dtype=np.int16)
    B = np.asarray(Image.open(p).convert("RGB"), dtype=np.int16)
    iguais_bytes += hashlib.md5(ba).digest() == hashlib.md5(bb).digest()
    iguais_pix += A.shape == B.shape and np.array_equal(A, B)
    difs.append((float(np.abs(A - B).mean()) if A.shape == B.shape else 999.0, f.stem))
print(f"{a.split}: {len(imgs)} imagens; patch ausente em Patches512: {len(faltam)} {faltam[:5]}")
print(f"  bytes idênticos: {iguais_bytes}   pixels idênticos: {iguais_pix}")
d = np.array([v for v, _ in difs])
print(f"  diferença média absoluta por pixel (0-255): mediana {np.median(d):.2f}, máx {d.max():.2f}")
for v, n in sorted(difs, reverse=True)[:8]:
    print(f"    {n}: {v:.2f}")
