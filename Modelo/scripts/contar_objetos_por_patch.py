"""
Junta a tabela patch -> rua (TCC-data/patches512_rotulos_quadra.csv, gerada
por rotular_tiles.py + rotular_quadras.py com --zoom 19) com a saída da
inferência em lote (deteccoes_lote.csv) e acrescenta 3 colunas de contagem
por patch: ponto_onibus, faixa_pedestre, area_NaoEdificada.

Só conta detecções com confiança >= --conf. Patch sem detecção fica com 0.
2026-10-18: padrão passou a ser conf 0,25 sobre deteccoes_lote_filtrado.csv
(filtro de tamanho mínimo, filtro_tamanho.py) — a revisão de 400 deu
precisão 0,98 (área), 0,79 (ônibus) e 0,86 (faixa) já na faixa 0,25-0,5, e
o corte 0,5 jogava fora ~28% dos pontos de ônibus reais. Antes: conf 0,5
sobre deteccoes_lote.csv (repetir com --conf 0.5 --deteccoes <bruto>).

ATENÇÃO (CLAUDE.md §7-b): a rua aqui é a do PATCH (~69 m de lado), não a do
objeto. Serve para agregar/explorar; para atribuir rua a um objeto
individual, usar o lat/lon da própria detecção.

Uso:
  python contar_objetos_por_patch.py
  python contar_objetos_por_patch.py --conf 0.5 --deteccoes ../runs/yolo26x_obb_v4_xl/deteccoes_lote.csv
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent  # D:\CityVision\Modelo
DETECCOES = RAIZ / "runs" / "yolo26x_obb_v4_xl" / "deteccoes_lote_filtrado.csv"
PATCHES = Path("D:/CityVision/TCC-data/patches512_rotulos_quadra.csv")
SAIDA = Path("D:/CityVision/TCC-data/patches512_rua_objetos.csv")

CLASSES = ["ponto_onibus", "faixa_pedestre", "area_NaoEdificada"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.25, help="confiança mínima para contar")
    ap.add_argument("--deteccoes", default=str(DETECCOES))
    ap.add_argument("--patches", default=str(PATCHES))
    ap.add_argument("--saida", default=str(SAIDA))
    args = ap.parse_args()

    contagem = Counter()  # (x, y, classe) -> n
    n_det = n_usadas = 0
    with open(args.deteccoes, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            n_det += 1
            if float(r["confianca"]) < args.conf:
                continue
            n_usadas += 1
            contagem[(r["tile_x"], r["tile_y"], r["classe"])] += 1

    with open(args.patches, encoding="utf-8-sig", newline="") as f:
        patches = list(csv.DictReader(f, delimiter=";"))

    chaves_patch = {(p["x"], p["y"]) for p in patches}
    orfas = sum(n for (x, y, _), n in contagem.items() if (x, y) not in chaves_patch)

    totais = Counter()
    for p in patches:
        for c in CLASSES:
            n = contagem.get((p["x"], p["y"], c), 0)
            p[c] = n
            totais[c] += n

    with open(args.saida, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(patches[0].keys()), delimiter=";")
        w.writeheader()
        w.writerows(patches)

    print(f"{n_det} detecções no CSV, {n_usadas} com confiança >= {args.conf}")
    print(f"{len(patches)} patches; detecções sem patch correspondente: {orfas}")
    for c in CLASSES:
        com = sum(1 for p in patches if p[c] > 0)
        print(f"  {c:18s} {totais[c]:6d} detecções em {com:6d} patches ({100 * com / len(patches):.1f}%)")
    print(f"-> {args.saida}")


if __name__ == "__main__":
    main()
