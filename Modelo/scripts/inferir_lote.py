"""
Inferência em lote do best.pt sobre um diretório de patches (por padrão,
TCC-data/Patches512 — os 45.523 recortes 512x512/z19 de toda a cobertura,
dos quais 348 entraram no dataset rotulado de treino/val).

Georreferencia cada detecção: o nome do arquivo <regional>/19/<x>/<y>.jpg
dá o índice do tile z19 (ver CLAUDE.md §5.1 — bloco 2x2 de tiles z20,
foto_pai = (x//2, y//2), 512x512 px = mesmo footprint de 1 tile z19).
O centro de cada caixa detectada (pixel 0..511) é convertido em
lat/lon com a mesma tile2deg() de TCC/rotular_tiles.py.

NÃO georreferencia rua/quadra (isso é um passo depois, reusando
rotular_quadras.py) — só lat/lon do centro de cada detecção.

Uso:
  python inferir_lote.py                       # roda em Patches512 inteiro
  python inferir_lote.py --limite 200           # só os 200 primeiros (teste rápido)
  python inferir_lote.py --pasta caminho/outra  # roda em outra pasta de patches
"""
import argparse
import csv
import math
import re
import time
from pathlib import Path

from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parent.parent  # D:\CityVision\Modelo
# 2026-10-12: aponta para o v4 XL (yolo26x-obb, 348 imgs dedup, mAP50 0,735).
# Antes: runs/yolo26_obb_baldio_onibus_faixa (v1 nano, 150 imgs).
RUN = RAIZ / "runs" / "yolo26x_obb_v4_xl"
PT = RUN / "weights" / "best.pt"
PASTA_PADRAO = Path("D:/CityVision/TCC-data/Patches512")
CSV_SAIDA = RUN / "deteccoes_lote.csv"

NOMES_CLASSE = {0: "area_NaoEdificada", 1: "ponto_onibus", 2: "faixa_pedestre"}
Z = 19  # zoom dos patches (ver docstring)

# mesma tile2deg() de TCC/rotular_tiles.py
def tile2deg(x: float, y: float, z: int) -> tuple[float, float]:
    """Canto superior-esquerdo do tile (aceita x/y fracionários)."""
    n = 2.0 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


PADRAO_CAMINHO = re.compile(r"([^/\\]+)[/\\](\d+)[/\\](\d+)[/\\](\d+)\.jpg$", re.IGNORECASE)


def parse_caminho(caminho: Path):
    """Extrai (regional, x_tile, y_tile) do caminho <regional>/<z>/<x>/<y>.jpg."""
    m = PADRAO_CAMINHO.search(str(caminho))
    if not m:
        return None
    regional, _z, x, y = m.groups()
    return regional, int(x), int(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default=str(PASTA_PADRAO))
    ap.add_argument("--limite", type=int, default=None, help="processa só os N primeiros arquivos (teste)")
    ap.add_argument("--conf", type=float, default=0.25, help="limiar de confiança")
    ap.add_argument("--batch", type=int, default=8, help="XL: 16 deu OOM no treino; 8 é folgado só pra inferência")
    args = ap.parse_args()

    pasta = Path(args.pasta)
    imagens = sorted(pasta.rglob("*.jpg"))
    if args.limite:
        imagens = imagens[: args.limite]
    print(f"{len(imagens)} imagens em {pasta}")

    modelo = YOLO(str(PT))
    CSV_SAIDA.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_deteccoes = 0
    n_sem_parse = 0
    with open(CSV_SAIDA, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow([
            "arquivo", "regional", "tile_x", "tile_y", "classe", "classe_id",
            "confianca", "lat_centro", "lon_centro", "px_cx", "px_cy",
            "px_w", "px_h", "angulo_rad",
        ])
        for i in range(0, len(imagens), args.batch):
            lote = imagens[i : i + args.batch]
            resultados = modelo.predict(
                [str(p) for p in lote], imgsz=1024, conf=args.conf, verbose=False
            )
            for caminho, r in zip(lote, resultados):
                info = parse_caminho(caminho)
                if info is None:
                    n_sem_parse += 1
                    regional, tx, ty = "?", None, None
                else:
                    regional, tx, ty = info
                if r.obb is None or len(r.obb) == 0:
                    continue
                for j in range(len(r.obb)):
                    cx, cy, bw, bh, ang = r.obb.xywhr[j].tolist()
                    cls_id = int(r.obb.cls[j].item())
                    conf = float(r.obb.conf[j].item())
                    lat = lon = ""
                    if tx is not None:
                        frac_x = tx + cx / 512.0
                        frac_y = ty + cy / 512.0
                        lat, lon = tile2deg(frac_x, frac_y, Z)
                    w.writerow([
                        str(caminho), regional, tx, ty,
                        NOMES_CLASSE.get(cls_id, cls_id), cls_id,
                        round(conf, 4), lat, lon,
                        round(cx, 1), round(cy, 1), round(bw, 1), round(bh, 1),
                        round(ang, 4),
                    ])
                    n_deteccoes += 1
            if (i // args.batch) % 50 == 0:
                dt = time.time() - t0
                print(f"  {i + len(lote)}/{len(imagens)} imagens, {n_deteccoes} detecções, {dt:.0f}s")

    dt = time.time() - t0
    print(f"Concluído: {len(imagens)} imagens, {n_deteccoes} detecções em {dt:.0f}s ({dt/max(len(imagens),1)*1000:.1f} ms/img)")
    if n_sem_parse:
        print(f"AVISO: {n_sem_parse} imagens fora do padrão <regional>/<z>/<x>/<y>.jpg (sem georreferência)")
    print("CSV salvo em:", CSV_SAIDA)


if __name__ == "__main__":
    main()
