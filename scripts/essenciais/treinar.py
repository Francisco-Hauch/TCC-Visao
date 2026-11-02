"""
Treino do YOLO26-OBB para as 3 classes do dataset_yolo (area_NaoEdificada /
ponto_onibus / faixa_pedestre) — ver D:\\CityVision\\Modelo\\dataset\\data.yaml.

Os hiperparâmetros estão em CONFIG, abaixo. Resultado e métricas em Modelo/README.md.

Uso:  python treinar.py
"""
from pathlib import Path
from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parents[2] / "Modelo"          # D:\CityVision\Modelo
DATA_YAML = RAIZ / "dataset" / "data.yaml"
PASTA_PESOS = RAIZ / "weights"
PASTA_RUNS = RAIZ / "runs"

# ---------------------------------------------------------------------------
# CONFIG DE TREINO
# ---------------------------------------------------------------------------
CONFIG = {
    "modelo_base": "yolo26x-obb.pt",  # XL, 348 imagens; Ultralytics baixa o peso se não estiver em weights/
    "imgsz": 1024,           # upscale 2x da imagem nativa (512px) — grid de features maior p/ objetos pequenos
    "epochs": 100,
    "batch": 4,              # XL com batch 16 ou 8 deu CUDA OOM na RTX 5060 Ti de 16 GB
    "patience": 30,          # tolerância maior que o usual (val set pequeno = métrica mais ruidosa)
    "optimizer": "auto",     # Ultralytics escolhe (tipicamente AdamW) e ajusta lr0/momentum internamente
    "lr0": "default",        # não sobrescrever: o optimizer=auto ajusta sozinho
    "device": 0,             # RTX 5060 Ti única GPU da máquina
    "workers": 0,            # workers>0 falha no Windows (erro 1455, "shared file mapping",
                              # bug conhecido do multiprocessing do PyTorch). Com um dataset
                              # pequeno, carregar no processo principal não pesa.
    "seed": 0,               # reprodutibilidade — default
    "amp": True,             # precisão mista (fp16) no treino — default recomendado
    "project": str(PASTA_RUNS),
    "name": "yolo26x_obb_v4_xl",  # v4 XL
}


def validar_config():
    faltando = [k for k, v in CONFIG.items() if v is None]
    if faltando:
        raise SystemExit(
            "Faltam decidir estas métricas antes de treinar: "
            + ", ".join(faltando)
            + "  (ver CONFIG no topo deste arquivo)."
        )


def main():
    validar_config()
    peso_base = PASTA_PESOS / CONFIG["modelo_base"]
    modelo = YOLO(str(peso_base) if peso_base.exists() else CONFIG["modelo_base"])
    kwargs = dict(
        data=str(DATA_YAML),
        imgsz=CONFIG["imgsz"],
        epochs=CONFIG["epochs"],
        batch=CONFIG["batch"],
        patience=CONFIG["patience"],
        optimizer=CONFIG["optimizer"],
        device=CONFIG["device"],
        workers=CONFIG["workers"],
        seed=CONFIG["seed"],
        amp=CONFIG["amp"],
        project=CONFIG["project"],
        name=CONFIG["name"],
    )
    if CONFIG["lr0"] != "default":
        kwargs["lr0"] = CONFIG["lr0"]
    modelo.train(**kwargs)


if __name__ == "__main__":
    main()
