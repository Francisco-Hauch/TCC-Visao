#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deteccao_yolo_obb.py
======================
Detector de OBJETO RIGIDO em imagem aerea, pre-treinado em DOTAv1 (YOLO11
via Ultralytics). Diferente do Grounding DINO/SAM2 (descartado em
2026-09-13, ver inferencia_grounded_sam2_OLD_ERR.py) e do SAM 3 (treinados
majoritariamente em foto de nivel de rua/obliqua), este modelo ja NASCEU
treinado em vista de cima - DOTAv1 e o benchmark padrao de deteccao em
imagem aerea/satelite (avioes, navios, veiculo, quadra esportiva etc.).

Foco desta rodada: **carro** (mapeia para as classes DOTA "small-vehicle" /
"large-vehicle" - a saida do modelo vem em ingles/hifen, o script so marca
"e_veiculo" pra facilitar filtro, sem forcar renomeacao de todas as 15
classes do DOTA). As outras 14 classes do DOTA saem de graca no CSV (ex.:
"swimming-pool" e "roundabout" ja estavam no radar do CLAUDE.md - pode
interessar mais pra frente mesmo fora do foco de hoje).

Nao serve pra: lixo, entulho, faixa de pedestre, painel solar, ponto de
onibus, rua, calcada - nenhuma dessas 8 classes do foco atual esta no
vocabulario fechado do DOTA. Isso e so a peca "carro" do quebra-cabeca.

USO (uma vez so, no ambiente - torch/transformers ja devem estar
instalados de antes):
  pip install ultralytics

Rodar (baixa o checkpoint sozinho na 1a vez, ~5-100 MB dependendo do
tamanho escolhido):
  python deteccao_yolo_obb.py --regional matriz --amostra 20 --salvar-mascaras
  python deteccao_yolo_obb.py --modelo yolo11s-obb.pt --unidade patch --amostra 50

Modelos (do menor/mais rapido ao maior/mais preciso):
  yolo11n-obb.pt (padrao) | yolo11s-obb.pt | yolo11m-obb.pt | yolo11l-obb.pt | yolo11x-obb.pt
"""

import argparse
import csv
import glob
import os
import random
import sys
import time

_PALETA = None  # inicializada em main() apos importar ultralytics


def desenhar_overlay_obb(orig_img_bgr, deteccoes):
    """orig_img_bgr: numpy BGR (Results.orig_img). deteccoes: lista de
    (cls_id, nome, conf, poly) - poly = 4 pontos (x,y). So caixa colorida
    por classe, sem texto em cima de cada uma; legenda unica no canto."""
    import cv2
    import numpy as np

    img = orig_img_bgr.copy()
    for cls_id, nome, conf, poly in deteccoes:
        cor = _PALETA(cls_id, True)
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=cor, thickness=2)

    vistos = {}
    for cls_id, nome, conf, poly in deteccoes:
        vistos.setdefault(cls_id, nome)
    if vistos:
        linha_h = 18
        largura_legenda = 22 + max(len(n) for n in vistos.values()) * 8
        altura_legenda = 6 + linha_h * len(vistos)
        overlay = img.copy()
        cv2.rectangle(overlay, (4, 4), (4 + largura_legenda, 4 + altura_legenda), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
        y = 4 + linha_h
        for cls_id, nome in sorted(vistos.items(), key=lambda kv: kv[1]):
            cor = _PALETA(cls_id, True)
            cv2.rectangle(img, (10, y - 12), (26, y + 2), cor, -1)
            cv2.putText(img, nome, (32, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            y += linha_h
    return img


UNIDADES = {
    "tile":  ("Ortofotos2019", "20"),
    "patch": ("Patches512", "19"),
}


def listar_imagens(dados, unidade, regional, amostra, seed):
    raiz_nome, sub = UNIDADES[unidade]
    raiz = os.path.join(dados, raiz_nome)
    padrao = os.path.join(raiz, regional or "*", sub, "*", "*.jpg")
    arquivos = glob.glob(padrao)
    arquivos.sort()
    if amostra and amostra < len(arquivos):
        random.Random(seed).shuffle(arquivos)
        arquivos = arquivos[:amostra]
        arquivos.sort()
    return arquivos


def parse_patch_path(caminho):
    partes = caminho.replace("\\", "/").split("/")
    py = int(os.path.splitext(partes[-1])[0])
    px = int(partes[-2])
    regional = partes[-4]
    return regional, px, py


def main():
    ap = argparse.ArgumentParser(description="Deteccao YOLO11-obb (DOTAv1) em tile/patch de ortofoto.")
    ap.add_argument("--dados", default=r"D:\CityVision\TCC-data")
    ap.add_argument("--unidade", choices=sorted(UNIDADES), default="tile")
    ap.add_argument("--regional", default=None)
    ap.add_argument("--amostra", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conf", type=float, default=0.25, help="confianca minima (padrao ultralytics)")
    ap.add_argument("--saida", default=None, help="padrao: <dados>\\YoloObb_<unidade>")
    ap.add_argument("--salvar-mascaras", action="store_true",
                    help="grava 1 imagem por tile/patch com as caixas orientadas coloridas por classe "
                         "e uma legenda unica no canto (sem texto em cima de cada caixa)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--modelo", default="yolo11n-obb.pt")
    args = ap.parse_args()

    saida = args.saida or os.path.join(args.dados, "YoloObb_%s" % args.unidade)
    dir_overlays = os.path.join(saida, "overlays")
    os.makedirs(saida, exist_ok=True)
    if args.salvar_mascaras:
        os.makedirs(dir_overlays, exist_ok=True)

    print("[setup] carregando %s..." % args.modelo)
    try:
        from ultralytics import YOLO
        from ultralytics.utils.plotting import Colors
        import cv2
    except ImportError as e:
        sys.exit("Faltou instalar o ambiente (pip install ultralytics). Erro: %s" % e)

    global _PALETA
    _PALETA = Colors()  # mesma paleta que o Ultralytics usa internamente, por indice de classe

    model = YOLO(args.modelo)

    print("[setup] listando imagens (unidade=%s, regional=%s, amostra=%s)..."
          % (args.unidade, args.regional, args.amostra or "todas"))
    arquivos = listar_imagens(args.dados, args.unidade, args.regional, args.amostra, args.seed)
    print("      %d imagens" % len(arquivos))

    destino_csv = os.path.join(saida, "deteccoes_yolo_obb.csv")
    campos = ["arquivo", "regional", "px", "py", "classe_dota", "e_veiculo", "conf",
              "cx", "cy", "largura", "altura", "angulo_rad", "mascara"]
    f_csv = open(destino_csv, "w", newline="", encoding="utf-8-sig")
    w = csv.DictWriter(f_csv, fieldnames=campos, delimiter=";")
    w.writeheader()

    t0 = time.time()
    n_deteccoes = 0
    for i, caminho in enumerate(arquivos):
        regional, px, py = parse_patch_path(caminho)
        results = model.predict(source=caminho, conf=args.conf, device=args.device, verbose=False)
        r = results[0]
        obb = r.obb

        if obb is not None and len(obb) > 0:
            overlay_path = ""
            deteccoes_img = []
            for j in range(len(obb)):
                cls_id = int(obb.cls[j])
                nome = r.names[cls_id]
                conf = float(obb.conf[j])
                poly = obb.xyxyxyxy[j].tolist()
                deteccoes_img.append((cls_id, nome, conf, poly))

            if args.salvar_mascaras:
                overlay_path = os.path.join(dir_overlays, "%s_%d_%d.jpg" % (regional, px, py))
                anotada = desenhar_overlay_obb(r.orig_img, deteccoes_img)
                cv2.imwrite(overlay_path, anotada)

            for j in range(len(obb)):
                cls_id = int(obb.cls[j])
                nome = r.names[cls_id]
                conf = float(obb.conf[j])
                cx, cy, largura, altura, ang = obb.xywhr[j].tolist()
                w.writerow({
                    "arquivo": caminho, "regional": regional, "px": px, "py": py,
                    "classe_dota": nome, "e_veiculo": int("vehicle" in nome.lower()),
                    "conf": "%.4f" % conf,
                    "cx": cx, "cy": cy, "largura": largura, "altura": altura,
                    "angulo_rad": ang, "mascara": overlay_path,
                })
                n_deteccoes += 1

        if (i + 1) % 20 == 0 or (i + 1) == len(arquivos):
            dt = time.time() - t0
            print("  %d/%d imagens | %d deteccoes | %.1fs (%.2fs/imagem)"
                  % (i + 1, len(arquivos), n_deteccoes, dt, dt / (i + 1)))

    f_csv.close()
    print("\nOK: %d deteccoes -> %s" % (n_deteccoes, destino_csv))


if __name__ == "__main__":
    main()
