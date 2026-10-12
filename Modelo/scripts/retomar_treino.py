"""
Retoma um treino interrompido a partir do último checkpoint (last.pt),
preservando época, otimizador e configuração original (lidos do
args.yaml salvo na pasta da run pelo Ultralytics).

Usado em 2026-09-27 para retomar do epoch 34/100 depois de um crash por
falta de RAM (outro programa consumindo memória na máquina, não bug do
treino — ver README.md). Reusado em 2026-10-04 pra retomar o fine-tuning
v2 (348 imgs) depois do usuário pausar de propósito pra usar a GPU.

Uso:  python retomar_treino.py [nome_da_run]
      (default: yolo26_obb_v2_348img — a run ativa em 2026-10-04)
"""
import sys
from pathlib import Path
from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parent.parent  # D:\CityVision\Modelo
NOME_RUN = sys.argv[1] if len(sys.argv) > 1 else "yolo26_obb_v2_348img"
ULTIMO_CHECKPOINT = RAIZ / "runs" / NOME_RUN / "weights" / "last.pt"


def main():
    if not ULTIMO_CHECKPOINT.exists():
        raise SystemExit(f"Checkpoint não encontrado: {ULTIMO_CHECKPOINT}")
    modelo = YOLO(str(ULTIMO_CHECKPOINT))
    modelo.train(resume=True)


if __name__ == "__main__":
    main()
