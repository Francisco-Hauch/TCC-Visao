"""
Treino do YOLO26-OBB para as 3 classes do dataset_yolo (area_NaoEdificada /
ponto_onibus / faixa_pedestre) — ver D:\\CityVision\\Modelo\\dataset\\data.yaml.

Ambiente preparado em 2026-09-27. Hiperparâmetros abaixo foram decididos com
o usuário (perguntados 1 a 1) em 2026-09-27 — ver histórico da conversa.
Script ainda NÃO FOI EXECUTADO — rodar só quando o usuário pedir.

Uso:  python treinar.py
"""
from pathlib import Path
from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parent.parent          # D:\CityVision\Modelo
DATA_YAML = RAIZ / "dataset" / "data.yaml"
PASTA_PESOS = RAIZ / "weights"
PASTA_RUNS = RAIZ / "runs"

# ---------------------------------------------------------------------------
# CONFIG DE TREINO — decidido com o usuário 1 a 1 em 2026-09-27.
# ---------------------------------------------------------------------------
CONFIG = {
    "modelo_base": "yolo26x-obb.pt",  # 2026-10-06: XL (antes nano) — mesmo dataset v3 dedup, 348 imgs; Ultralytics baixa o peso se não estiver em weights/
    "imgsz": 1024,           # upscale 2x da imagem nativa (512px) — grid de features maior p/ objetos pequenos
    "epochs": 100,
    "batch": 4,              # 2026-10-06: XL com batch 16 (e 8 no auto-retry) deu CUDA OOM na 5060 Ti 16 GB — usuário escolheu 4
    "patience": 30,          # tolerância maior que o usual (val set pequeno = métrica mais ruidosa)
    "optimizer": "auto",     # Ultralytics escolhe (tipicamente AdamW) e ajusta lr0/momentum internamente
    "lr0": "default",        # decidido: não sobrescrever — deixar o optimizer=auto ajustar sozinho
    "device": 0,             # RTX 5060 Ti única GPU da máquina
    "workers": 0,            # 2026-09-27: workers>0 crashou no Windows (erro 1455,
                              # "shared file mapping" — bug conhecido do multiprocessing
                              # do PyTorch nesta plataforma). Dataset é pequeno (117 imgs),
                              # workers=0 (dataloading no processo principal) não pesa.
    "seed": 0,               # reprodutibilidade — default
    "amp": True,             # precisão mista (fp16) no treino — default recomendado
    "project": str(PASTA_RUNS),
    "name": "yolo26x_obb_v4_xl",  # 2026-10-06: v4 = v3 com modelo XL. Histórico v3 abaixo:
    # "name": "yolo26_obb_v3_dedup",  # 2026-10-04: v3 — mesmo dataset de 348 imgs do v2, mas com as
                                     # caixas duplicadas/sobrepostas de area_NaoEdificada removidas
                                     # (212 instâncias em vez de 240 — ver dataset/LEIA-ME.txt).
                                     # v2 teve precisão de baldio degradada (1,00->0,45) por causa
                                     # exatamente dessas duplicatas. Hiperparâmetros idênticos.
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
