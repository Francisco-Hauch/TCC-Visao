"""
Revalida o best.pt do v4 XL no dataset rotulado, com as mesmas opções do treino
(imgsz 1024, iou 0,7, conf padrão do val), e imprime as métricas por classe.

Serve para conferir os números do treino (results.csv / BoxPR_curve.png) sem
depender da memória: rodar de novo deve dar o mesmo resultado, a menos de
diferenças numéricas de GPU/CPU e fp16/fp32.

Uso:
  python validar_modelo.py                    # split val (padrão)
  python validar_modelo.py --split train      # mesmo modelo no conjunto de treino
  python validar_modelo.py --device cpu       # sem GPU
Saída: tabela no terminal e Modelo/runs/val_<split>/ (curvas, matriz de confusão).
"""
import argparse
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parents[2] / "Modelo"  # Modelo/
PT = RAIZ / "runs" / "yolo26x_obb_v4_xl" / "weights" / "best.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--device", default=None, help="'0' (GPU) ou 'cpu'; padrão: automático")
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()

    # data.yaml tem path absoluto do Windows; uma cópia com o caminho desta máquina
    cfg = yaml.safe_load((RAIZ / "dataset" / "data.yaml").read_text(encoding="utf-8"))
    cfg["path"] = str((RAIZ / "dataset").resolve())
    tmp = Path(tempfile.mkdtemp()) / "data.yaml"
    tmp.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    m = YOLO(str(PT))
    r = m.val(data=str(tmp), split=a.split, imgsz=1024, batch=a.batch, iou=0.7,
              device=a.device, plots=True, project=str(RAIZ / "runs"),
              name=f"val_{a.split}", exist_ok=True, verbose=False)

    nomes = r.names
    print(f"\nbest.pt em {a.split}: {len(r.box.ap_class_index)} classes avaliadas")
    print(f"{'classe':20s} {'P':>6s} {'R':>6s} {'mAP50':>7s} {'mAP50-95':>9s}")
    print(f"{'todas':20s} {r.box.mp:6.3f} {r.box.mr:6.3f} {r.box.map50:7.3f} {r.box.map:9.3f}")
    for i, c in enumerate(r.box.ap_class_index):
        p, rr, ap50, ap = r.box.class_result(i)
        print(f"{nomes[int(c)]:20s} {p:6.3f} {rr:6.3f} {ap50:7.3f} {ap:9.3f}")


if __name__ == "__main__":
    main()
