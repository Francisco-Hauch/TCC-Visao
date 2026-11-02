"""
Retoma um treino interrompido a partir do último checkpoint (last.pt),
preservando época, otimizador e configuração original (lidos do
args.yaml salvo na pasta da run pelo Ultralytics).

Uso:  python retomar_treino.py [nome_da_run]
      (sem argumento, usa yolo26_obb_v2_348img; informar o nome da run)
"""
import sys
from pathlib import Path
from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parents[2] / "Modelo"  # D:\CityVision\Modelo
NOME_RUN = sys.argv[1] if len(sys.argv) > 1 else "yolo26_obb_v2_348img"
ULTIMO_CHECKPOINT = RAIZ / "runs" / NOME_RUN / "weights" / "last.pt"


def main():
    if not ULTIMO_CHECKPOINT.exists():
        raise SystemExit(f"Checkpoint não encontrado: {ULTIMO_CHECKPOINT}")
    modelo = YOLO(str(ULTIMO_CHECKPOINT))
    modelo.train(resume=True)


if __name__ == "__main__":
    main()
