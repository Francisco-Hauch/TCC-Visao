"""
Checagem de vazamento entre treino e validação do dataset YOLO
(Modelo/dataset/images/{train,val}, labels/{train,val}).

Testes, do mais forte ao mais fraco:
  1. mesmo patch      — a chave (x, y) do tile z19 (tirada do nome
                        <regional>_<x>_<y>.jpg) aparece nos dois conjuntos;
  2. arquivo idêntico — MD5 dos bytes do .jpg igual entre train e val;
  3. pixels idênticos — MD5 dos pixels decodificados (pega a mesma imagem
                        regravada com outra compressão);
  5. quase-duplicata  — distância de Hamming entre hashes perceptuais
                        (pHash 64 bits, DCT 32x32) ≤ --limiar;
  4. rótulo idêntico  — mesmo conteúdo de .txt não vazio nos dois conjuntos;
  6. vizinhança       — distância (em tiles z19, ~69 m) do patch de val ao
                        patch de treino mais próximo. Patches não se
                        sobrepõem, então vizinho não é cópia; mas um objeto
                        na borda pode aparecer metade em cada um. Informativo.

Os testes 1-5 são os que definem vazamento; qualquer ocorrência => FALHA.
Também roda 1-5 dentro de cada conjunto (duplicata interna).

Uso:  python checar_vazamento.py [--dataset CAMINHO] [--limiar 4] [--csv saida.csv]
Só usa a biblioteca padrão, numpy, scipy e Pillow.
"""
import argparse
import csv
import hashlib
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn

RAIZ = Path(__file__).resolve().parents[2] / "Modelo"  # Modelo/
PADRAO = re.compile(r"^(?P<reg>.+)_(?P<x>\d+)_(?P<y>\d+)$")


def phash(img, tam=32, bloco=8):
    """pHash clássico: cinza 32x32, DCT 2D, bloco 8x8 de baixa frequência,
    bit = coeficiente > mediana (sem o DC). Devolve int de 64 bits."""
    g = np.asarray(img.convert("L").resize((tam, tam), Image.LANCZOS), dtype=np.float64)
    d = dctn(g, norm="ortho")[:bloco, :bloco].flatten()
    med = np.median(d[1:])
    bits = d > med
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


def carregar(dataset, split):
    itens = []
    for img in sorted((dataset / "images" / split).glob("*.jpg")):
        m = PADRAO.match(img.stem)
        lbl = dataset / "labels" / split / (img.stem + ".txt")
        bruto = img.read_bytes()
        im = Image.open(img)
        im.load()
        pix = np.asarray(im.convert("RGB"))
        rot = lbl.read_text(encoding="utf-8").strip() if lbl.exists() else None
        itens.append({
            "split": split, "nome": img.stem,
            "regional": m["reg"] if m else "?",
            "x": int(m["x"]) if m else None, "y": int(m["y"]) if m else None,
            "md5_arquivo": hashlib.md5(bruto).hexdigest(),
            "md5_pixels": hashlib.md5(pix.tobytes()).hexdigest(),
            "tamanho": im.size, "phash": phash(im),
            "rotulo": rot, "tem_rotulo": lbl.exists(),
            "n_caixas": 0 if not rot else len(rot.splitlines()),
        })
    return itens


def duplicatas(itens, chave):
    grupos = defaultdict(list)
    for it in itens:
        k = it[chave]
        if k is not None and k != "":
            grupos[k].append(it)
    return [g for g in grupos.values() if len(g) > 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(RAIZ / "dataset"))
    ap.add_argument("--limiar", type=int, default=4,
                    help="Hamming máximo (de 64) para chamar de quase-duplicata")
    ap.add_argument("--csv", default=None, help="grava uma linha por imagem de val")
    a = ap.parse_args()
    ds = Path(a.dataset)

    tr, va = carregar(ds, "train"), carregar(ds, "val")
    print(f"dataset: {ds}")
    print(f"train: {len(tr)} imagens, {sum(i['n_caixas'] for i in tr)} caixas, "
          f"{sum(1 for i in tr if not i['tem_rotulo'])} sem .txt")
    print(f"val:   {len(va)} imagens, {sum(i['n_caixas'] for i in va)} caixas, "
          f"{sum(1 for i in va if not i['tem_rotulo'])} sem .txt")
    sem_padrao = [i["nome"] for i in tr + va if i["x"] is None]
    print(f"nomes fora do padrão <regional>_<x>_<y>: {len(sem_padrao)} {sem_padrao[:5]}")
    tamanhos = sorted({i["tamanho"] for i in tr + va})
    print(f"tamanhos de imagem: {tamanhos}")
    print()

    falhas = 0
    todos = tr + va
    testes = [("1 mesmo patch (x,y)", "xy"), ("2 arquivo idêntico (MD5 bytes)", "md5_arquivo"),
              ("3 pixels idênticos (MD5 pixels)", "md5_pixels"),
              ("4 rótulo idêntico (.txt não vazio)", "rotulo")]
    for it in todos:
        it["xy"] = (it["x"], it["y"]) if it["x"] is not None else None
    print("=== testes de vazamento ===")
    for titulo, chave in testes:
        grupos = duplicatas(todos, chave)
        cruz = [g for g in grupos if {i["split"] for i in g} == {"train", "val"}]
        intra = [g for g in grupos if len({i["split"] for i in g}) == 1]
        falhas += len(cruz)
        st = "OK" if not cruz else "FALHA"
        print(f"{titulo:38s} train×val: {len(cruz):3d}  dentro do mesmo conjunto: {len(intra):3d}  [{st}]")
        for g in (cruz + intra)[:10]:
            print("      ", " | ".join(f"{i['split']}/{i['nome']}" for i in g))

    # 5. quase-duplicatas perceptuais
    dist_min = []
    pares_cruz = []
    for v in va:
        d, melhor = min((hamming(v["phash"], t["phash"]), t["nome"]) for t in tr)
        dist_min.append((d, v["nome"], melhor))
        if d <= a.limiar:
            pares_cruz.append((d, v["nome"], melhor))
    pares_intra = [(hamming(p["phash"], q["phash"]), p["split"], p["nome"], q["nome"])
                   for s in (tr, va) for p, q in combinations(s, 2)
                   if hamming(p["phash"], q["phash"]) <= a.limiar]
    falhas += len(pares_cruz)
    st = "OK" if not pares_cruz else "FALHA"
    print(f"{'5 quase-duplicata (pHash ≤ ' + str(a.limiar) + '/64)':38s} train×val: "
          f"{len(pares_cruz):3d}  dentro do mesmo conjunto: {len(pares_intra):3d}  [{st}]")
    for d, vn, tn in pares_cruz[:10]:
        print(f"       d={d}  val/{vn} ~ train/{tn}")
    for d, s, p, q in pares_intra[:10]:
        print(f"       d={d}  {s}/{p} ~ {s}/{q}")
    ds_ = np.array([d for d, _, _ in dist_min])
    print(f"   distância pHash val→train mais próxima: mín {ds_.min()}, "
          f"p5 {np.percentile(ds_, 5):.0f}, mediana {np.median(ds_):.0f}, máx {ds_.max()} (de 64)")
    print("   5 pares mais parecidos:", ", ".join(f"{vn}~{tn} (d={d})"
                                                  for d, vn, tn in sorted(dist_min)[:5]))
    print()

    # 6. vizinhança espacial (informativo)
    print("=== 6 vizinhança espacial val → train (informativo, não é vazamento) ===")
    txy = np.array([(i["x"], i["y"]) for i in tr if i["x"] is not None])
    linhas = []
    for v in va:
        if v["x"] is None:
            continue
        cheb = np.max(np.abs(txy - np.array([v["x"], v["y"]])), axis=1)
        j = int(np.argmin(cheb))
        linhas.append({"val": v["nome"], "train_mais_proximo": tr[j]["nome"],
                       "dist_tiles": int(cheb[j]), "dist_m_aprox": int(cheb[j]) * 69,
                       "phash_dist_min": next(d for d, n, _ in dist_min if n == v["nome"])})
    dist = np.array([l["dist_tiles"] for l in linhas])
    for k, rot in [(0, "mesmo tile (0)"), (1, "encostado (1 tile, divide borda/canto)"),
                   (2, "a 2 tiles (~138 m)")]:
        print(f"   val com train mais próximo a {rot}: {int((dist == k).sum())}")
    print(f"   val com train mais próximo a ≥3 tiles (≥~200 m): {int((dist >= 3).sum())}")
    print(f"   distância (tiles): mediana {np.median(dist):.0f}, mín {dist.min()}, máx {dist.max()}")
    enc = [l for l in linhas if l["dist_tiles"] == 1]
    for l in enc:
        print(f"      encostado: val/{l['val']}  ~  train/{l['train_mais_proximo']}")
    print()

    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()), delimiter=";")
            w.writeheader()
            w.writerows(linhas)
        print(f"-> {a.csv}")

    print("RESULTADO:", "SEM VAZAMENTO (testes 1-5)" if falhas == 0 else f"{falhas} OCORRÊNCIA(S) DE VAZAMENTO")
    sys.exit(1 if falhas else 0)


if __name__ == "__main__":
    main()
