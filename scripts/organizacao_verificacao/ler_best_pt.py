"""
Lê os metadados gravados no próprio best.pt (versão do Ultralytics do treino,
métricas de validação do checkpoint escolhido e o critério 'fitness') e diz a
que época do results.csv o checkpoint corresponde.

Uso: python ler_best_pt.py
"""
from pathlib import Path
import torch

RUN = Path(__file__).resolve().parents[2] / "Modelo" / "runs" / "yolo26x_obb_v4_xl"
ck = torch.load(RUN / "weights" / "best.pt", map_location="cpu", weights_only=False)
tm = ck["train_metrics"]
print(f"Ultralytics usado no treino: {ck['version']}   data: {ck['date']}")
for k in ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)", "fitness"):
    print(f"  {k:24s} {tm[k]:.5f}")
tr = ck["train_results"]
ep = [e for e, a, b in zip(tr["epoch"], tr["metrics/mAP50(B)"], tr["metrics/mAP50-95(B)"])
      if abs(a - tm["metrics/mAP50(B)"]) < 1e-6 and abs(b - tm["metrics/mAP50-95(B)"]) < 1e-6]
print(f"época do results.csv com essas métricas: {ep}")
i = max(range(len(tr["epoch"])), key=lambda j: tr["metrics/mAP50-95(B)"][j])
print(f"época de maior mAP50-95 (= fitness): {tr['epoch'][i]}")
j = max(range(len(tr["epoch"])), key=lambda j: tr["metrics/mAP50(B)"][j])
print(f"época de maior mAP50: {tr['epoch'][j]}  (mAP50 {tr['metrics/mAP50(B)'][j]:.5f}, "
      f"P {tr['metrics/precision(B)'][j]:.5f}, R {tr['metrics/recall(B)'][j]:.5f}) — NÃO é o best.pt")
