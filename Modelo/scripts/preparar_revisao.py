"""
Monta uma pasta com N patches onde o modelo detectou algo, roda a
inferência de novo em cada um (pra pegar os 4 cantos exatos via
obb.xyxyxyxyn) e grava imagem + rótulo .txt (mesmo formato do dataset de
treino) numa pasta "solta" (sem subpastas images/labels) — formato que o
`rotulador-yolo.html` já sabe abrir e desenhar por cima.

Seleção: estratificada por classe a partir de deteccoes_lote.csv (já
rodado em toda a Patches512), tentando cobrir uma faixa de confiança
ampla por classe (não só os mais confiantes), não só os "slam dunks".

Exclui patches que já estão no dataset de treino/val (dataset/images/*),
pra não revisar de novo o que já foi rotulado. 2026-10-18: a exclusão é
pelo número do patch (x,y), não pelo nome, e cobre também os vizinhos num
raio de --raio-exclusao patches (padrão 2, ~140 m) — na revisão de 400
anterior, 29 patches encostavam num patch do dataset (mesma mata/rua
continuando de uma imagem para a outra), o que contamina a revisão como
teste independente. Guarda uma cópia das
predições originais em <dest>/predicoes_modelo/ (o rotulador sobrescreve
os .txt ao lado das imagens), pra comparar modelo x correção depois.

Uso:  python preparar_revisao.py [--n 200] [--conf 0.25]
      python preparar_revisao.py --n 400 --dest ../revisao_400_xl   # 2026-10-13, v4 XL (refeita 2026-10-18 com raio de exclusão 2)
"""
import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parent.parent
# 2026-10-13: aponta para o v4 XL. revisao_200 (2026-10-04) foi feita com o v1
# nano (runs/yolo26_obb_baldio_onibus_faixa, hoje fora do repositório).
RUN = RAIZ / "runs" / "yolo26x_obb_v4_xl"
PT = RUN / "weights" / "best.pt"
DETECCOES_CSV = RUN / "deteccoes_lote.csv"
DATASET_IMGS = RAIZ / "dataset" / "images"
CLASSES_TXT = RAIZ / "dataset" / "classes.txt"
DEST_PADRAO = RAIZ / "revisao_200"

random.seed(0)


def ja_rotuladas(raio: int) -> set:
    """(x, y) dos patches de train/val e de todos os vizinhos até `raio`
    patches de distância (Chebyshev), independente da regional no nome."""
    excluir = set()
    for p in DATASET_IMGS.rglob("*.jpg"):
        x, y = (int(v) for v in p.stem.rsplit("_", 2)[1:])
        for dx in range(-raio, raio + 1):
            for dy in range(-raio, raio + 1):
                excluir.add((x + dx, y + dy))
    return excluir


def xy_patch(caminho: str) -> tuple:
    """'.../<regional>/19/<x>/<y>.jpg' -> (x, y)."""
    p = Path(caminho)
    return int(p.parent.name), int(p.stem)


def selecionar_imagens(n_total: int, conf_min: float, excluir: set):
    """Agrupa por imagem, pega a melhor confiança por classe, e sorteia
    uma faixa espalhada (não só os mais confiantes) por classe."""
    por_classe = defaultdict(list)  # classe_id -> [(conf, caminho)]
    vistos_por_imagem = defaultdict(dict)  # caminho -> {classe_id: conf_max}
    with open(DETECCOES_CSV, encoding="utf-8-sig") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            conf = float(row["confianca"])
            if conf < conf_min:
                continue
            cls = int(row["classe_id"])
            caminho = row["arquivo"]
            if xy_patch(caminho) in excluir:
                continue
            atual = vistos_por_imagem[caminho].get(cls, 0.0)
            if conf > atual:
                vistos_por_imagem[caminho][cls] = conf

    for caminho, classes in vistos_por_imagem.items():
        for cls, conf in classes.items():
            por_classe[cls].append((conf, caminho))

    n_classes = len(por_classe)
    alvo_por_classe = n_total // n_classes
    escolhidos = set()
    for cls, lista in por_classe.items():
        lista.sort(key=lambda t: t[0])  # menor -> maior confiança
        k = min(alvo_por_classe, len(lista))
        # amostra espalhada ao longo da lista ordenada por confiança,
        # não só o topo — cobre caso fácil e caso ambíguo
        idxs = [round(i * (len(lista) - 1) / max(k - 1, 1)) for i in range(k)]
        for i in sorted(set(idxs)):
            escolhidos.add(lista[i][1])

    escolhidos = list(escolhidos)
    if len(escolhidos) > n_total:
        random.shuffle(escolhidos)
        escolhidos = escolhidos[:n_total]
    elif len(escolhidos) < n_total:
        resto = [c for lst in por_classe.values() for _, c in lst if c not in escolhidos]
        random.shuffle(resto)
        for c in resto:
            if len(escolhidos) >= n_total:
                break
            if c not in escolhidos:
                escolhidos.append(c)
    return escolhidos


def nome_flat(caminho: str) -> str:
    """'.../<regional>/19/<x>/<y>.jpg' -> '<regional>_<x>_<y>.jpg' (mesmo
    padrão de nome usado no dataset_yolo rotulado manualmente)."""
    p = Path(caminho)
    y = p.stem
    x = p.parent.name
    regional = p.parent.parent.parent.name
    return f"{regional}_{x}_{y}.jpg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--dest", default=str(DEST_PADRAO))
    ap.add_argument("--raio-exclusao", type=int, default=2)
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    print("Selecionando imagens (estratificado por classe/confiança)...")
    excluir = ja_rotuladas(args.raio_exclusao)
    print(f"  {len(excluir)} posições (x,y) excluídas: dataset de treino/val + raio de {args.raio_exclusao} patches")
    caminhos = selecionar_imagens(args.n, args.conf, excluir)
    print(f"  {len(caminhos)} imagens selecionadas")

    shutil.copy(CLASSES_TXT, dest / "classes.txt")
    orig = dest / "predicoes_modelo"
    orig.mkdir(exist_ok=True)

    modelo = YOLO(str(PT))
    n_ok = n_vazio = 0
    for i in range(0, len(caminhos), 8):  # XL: batch 8
        lote = caminhos[i : i + 8]
        resultados = modelo.predict(lote, imgsz=1024, conf=args.conf, verbose=False)
        for caminho, r in zip(lote, resultados):
            nome = nome_flat(caminho)
            shutil.copy(caminho, dest / nome)
            linhas = []
            if r.obb is not None and len(r.obb) > 0:
                corners = r.obb.xyxyxyxyn.reshape(len(r.obb), 8).tolist()
                for j, pts in enumerate(corners):
                    cls = int(r.obb.cls[j].item())
                    linhas.append(str(cls) + " " + " ".join(f"{v:.6f}" for v in pts))
                n_ok += 1
            else:
                n_vazio += 1
            texto = "\n".join(linhas) + ("\n" if linhas else "")
            (dest / (Path(nome).stem + ".txt")).write_text(texto, encoding="utf-8")
            (orig / (Path(nome).stem + ".txt")).write_text(texto, encoding="utf-8")
        print(f"  {min(i + 8, len(caminhos))}/{len(caminhos)}")

    print(f"Concluído: {n_ok} imagens com caixa, {n_vazio} sem detecção na re-inferência (raro, confiança no limite)")
    print(f"Pasta pronta em: {dest}")
    print('Abrir no rotulador-yolo.html com "Abrir pasta do projeto" (ou arrastar a pasta).')


if __name__ == "__main__":
    main()
