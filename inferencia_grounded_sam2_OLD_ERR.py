#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
############################################################################
# DESATIVADO em 2026-09-13 - NAO USAR. Mantido so como registro historico.
#
# Motivo: Grounding DINO + SAM 2 sao treinados majoritariamente em foto de
# nivel de rua/obliqua. Testado numa amostra de 20 tiles/patches da Matriz:
# errou praticamente tudo (marcou fileira de carro estacionado como "tubo",
# telhado de galpao como "calcada", copa de arvore como "tubo", detectou
# "vacant lot" num tile com buraco de dado/preto). Literatura confirma que
# nao e so prompt mal escrito - e um problema arquitetural (o encoder
# aprende feicao de perfil - roda de carro, fachada - que nao existe em
# vista nadir): "Do Open-Vocabulary Detectors Transfer to Aerial Imagery?"
# (arXiv 2601.22164) mede ate 69% de falso positivo no melhor modelo
# testado (OWLv2, 27,6% F1) em imagem aerea zero-shot.
#
# Ver conversa de 2026-09-13 para o raciocinio completo. Proximo modelo a
# escolher por classe, nao um so pra tudo - ver arquivo ativo mais recente.
############################################################################

inferencia_grounded_sam2.py
=============================
Mesma ideia do inferencia_sam3.py (mesmo CSV de saida, mesmo conjunto de
classes), mas com um combo 100% aberto - sem pedido de acesso, sem espera -
enquanto o acesso ao checkpoint do SAM 3 nao chega: **Grounding DINO**
(deteccao por texto, caixa) + **SAM 2** (caixa -> mascara). Os dois vem
prontos pela biblioteca `transformers`, download direto do Hugging Face,
sem gate.

Troca para SAM 3 depois: o CSV de saida (`deteccoes_sam3.csv` /
`deteccoes_grounded_sam2.csv`) tem o mesmo formato, entao qualquer analise
feita em cima de um serve pro outro sem mudar nada rio abaixo.

Historico de ajuste de classes (2026-09-13, depois da 1a rodada de teste
numa amostra de 20 patches da Matriz): "tubo" (estacao-tubo do BRT) saiu -
0/N acertos, marcou fileira de carro estacionado, copa de arvore e caixa de
saida de telhado; e um conceito especifico demais de Curitiba pra um
modelo generico reconhecer zero-shot. Entraram classes "distratoras" (rua,
telhado, carro, caminhao, moto, faixa de pedestre) - a ideia e dar ao
detector opcao concorrente, pra ele parar de forcar objeto que nao entende
dentro de "baldio"/"tubo" por falta de alternativa melhor.

USO (uma vez so, no ambiente):
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
  pip install transformers pillow accelerate

Rodar (voce controla o tamanho/GPU pelos flags):
  python inferencia_grounded_sam2.py --regional matriz --amostra 20 --salvar-mascaras
  python inferencia_grounded_sam2.py --unidade patch --regional matriz --amostra 20 --salvar-mascaras

--unidade tile (padrao) usa o tile z20 cru (256x256, ~34,5 m) em vez do
patch 512x512 do Estagio 4 - mesma resolucao por pixel (~13,5 cm/px), mas
como o modelo redimensiona a imagem de entrada pra um tamanho fixo, uma
imagem de origem menor da mais "zoom" relativo aos objetos pequenos
(carro, lixeira, faixa de pedestre) depois desse redimensionamento.

Modelos (trocaveis por flag, do menor/mais rapido ao maior/mais preciso):
  --grounding-model IDEA-Research/grounding-dino-tiny (padrao) | grounding-dino-base
  --sam-model        facebook/sam2.1-hiera-tiny | -small | -base-plus | -large (padrao)
"""

import argparse
import csv
import glob
import os
import random
import sys
import time

CLASSES = {
    # revisado em 2026-09-13 (3a rodada): "tubo" ja tinha saido (0/N acertos,
    # marcava carro/arvore/telhado). "baldio" fica de fora POR AGORA - vai
    # ganhar via regra geometrica/fiscal (lote sem edificacao + altura + NDVI
    # + campo fiscal Territorial), nao por detector generico - ver CLAUDE.md.
    # "lixeira", "telhado" e "caminhao" tambem saem do foco desta rodada.
    "ponto_onibus":   "bus stop shelter",
    "lixo":           "garbage litter on the ground",
    "entulho":        "construction debris",
    "carro":          "car",
    "moto":           "motorcycle",
    "rua":            "paved street",
    "calcada":        "sidewalk",
    "faixa_pedestre": "pedestrian crosswalk",
    "painel_solar":   "solar panel",
}

# cor fixa por classe, pra ficar consistente entre imagens (R,G,B)
CORES = {
    "ponto_onibus":   (60, 100, 255),
    "lixo":           (230, 230, 0),
    "entulho":        (255, 150, 0),
    "carro":          (255, 255, 255),
    "moto":           (0, 255, 150),
    "rua":            (150, 150, 150),
    "calcada":        (0, 220, 220),
    "faixa_pedestre": (255, 215, 0),
    "painel_solar":   (0, 120, 220),
}


def desenhar_overlay(image, deteccoes):
    """image: PIL RGB. deteccoes: lista de (chave, score, box, mascara_bin).
    Devolve uma copia RGB com mascara semi-transparente + caixa + rotulo."""
    import numpy as np
    from PIL import Image as PILImage, ImageDraw, ImageFont

    base = image.convert("RGBA")
    for chave, score, box, mascara_bin in deteccoes:
        cor = CORES.get(chave, (255, 255, 255))
        camada = np.zeros((*mascara_bin.shape, 4), dtype=np.uint8)
        camada[mascara_bin] = (*cor, 110)
        base = PILImage.alpha_composite(base, PILImage.fromarray(camada, mode="RGBA"))

    draw = ImageDraw.Draw(base)
    try:
        fonte = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        fonte = ImageFont.load_default()
    for chave, score, box, mascara_bin in deteccoes:
        cor = CORES.get(chave, (255, 255, 255))
        draw.rectangle(box, outline=cor, width=2)
        texto = "%s %.2f" % (chave, score)
        ty = max(0, box[1] - 18)
        draw.rectangle([box[0], ty, box[0] + 8 * len(texto), ty + 16], fill=(0, 0, 0, 180))
        draw.text((box[0] + 2, ty), texto, fill=cor, font=fonte)

    return base.convert("RGB")


UNIDADES = {
    # unidade -> (pasta raiz, subpasta de zoom) - mesma profundidade de
    # diretorio nos dois casos (<raiz>\<regional>\<zoom>\<x>\<y>.jpg), entao
    # parse_patch_path() funciona igual pros dois.
    "tile":  ("Ortofotos2019", "20"),   # 256x256, ~34,5 m, tile z20 cru
    "patch": ("Patches512", "19"),      # 512x512, ~69 m, bloco 2x2 (Estagio 4)
}


def listar_patches(dados, unidade, regional, amostra, seed):
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
    ap = argparse.ArgumentParser(description="Inferencia Grounding DINO + SAM 2 (zero-shot) nos patches 512x512.")
    ap.add_argument("--dados", default=r"D:\CityVision\TCC-data")
    ap.add_argument("--unidade", choices=sorted(UNIDADES), default="tile",
                    help="tile = z20 cru 256x256/34,5m (mais zoom relativo, padrao); "
                         "patch = z19 512x512/69m (Estagio 4, mais contexto)")
    ap.add_argument("--regional", default=None)
    ap.add_argument("--classes", default=None, help="lista separada por virgula (padrao: todas)")
    ap.add_argument("--amostra", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--box-threshold", type=float, default=0.30)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--saida", default=None, help="padrao: <dados>\\GroundedSAM2")
    ap.add_argument("--salvar-mascaras", action="store_true",
                    help="grava 1 imagem por patch com todas as deteccoes desenhadas por cima (mascara + caixa + rotulo)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--grounding-model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--sam-model", default="facebook/sam2.1-hiera-large")
    args = ap.parse_args()

    saida = args.saida or os.path.join(args.dados, "GroundedSAM2_%s" % args.unidade)
    dir_overlays = os.path.join(saida, "overlays")
    os.makedirs(saida, exist_ok=True)
    if args.salvar_mascaras:
        os.makedirs(dir_overlays, exist_ok=True)

    classes = CLASSES
    if args.classes:
        chaves = [c.strip() for c in args.classes.split(",")]
        classes = {k: CLASSES[k] for k in chaves}
    prompts = list(classes.values())
    prompt_para_chave = {v: k for k, v in classes.items()}

    print("[setup] carregando Grounding DINO (%s) e SAM 2 (%s)..." % (args.grounding_model, args.sam_model))
    try:
        import numpy as np
        import torch
        from PIL import Image
        from transformers import (AutoProcessor, AutoModelForZeroShotObjectDetection,
                                   Sam2Processor, Sam2Model)
    except ImportError as e:
        sys.exit("Faltou instalar o ambiente (ver docstring deste arquivo, secao USO). Erro: %s" % e)

    dino_processor = AutoProcessor.from_pretrained(args.grounding_model)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(args.grounding_model).to(args.device)
    dino_model.eval()

    sam_processor = Sam2Processor.from_pretrained(args.sam_model)
    sam_model = Sam2Model.from_pretrained(args.sam_model).to(args.device)
    sam_model.eval()

    print("[setup] listando imagens (unidade=%s, regional=%s, amostra=%s)..."
          % (args.unidade, args.regional, args.amostra or "todas"))
    arquivos = listar_patches(args.dados, args.unidade, args.regional, args.amostra, args.seed)
    print("      %d imagens, %d classes num prompt so (1 chamada DINO/imagem)" % (len(arquivos), len(classes)))

    destino_csv = os.path.join(saida, "deteccoes_grounded_sam2.csv")
    campos = ["patch", "regional", "px", "py", "classe", "prompt", "score",
              "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "area_px", "mascara"]
    f_csv = open(destino_csv, "w", newline="", encoding="utf-8-sig")
    w = csv.DictWriter(f_csv, fieldnames=campos, delimiter=";")
    w.writeheader()

    t0 = time.time()
    n_deteccoes = 0
    text_labels = [prompts]  # uma imagem por vez, N frases nessa imagem

    for i, caminho in enumerate(arquivos):
        regional, px, py = parse_patch_path(caminho)
        image = Image.open(caminho).convert("RGB")

        dino_inputs = dino_processor(images=image, text=text_labels, return_tensors="pt").to(args.device)
        with torch.no_grad():
            dino_outputs = dino_model(**dino_inputs)
        resultado = dino_processor.post_process_grounded_object_detection(
            dino_outputs, dino_inputs.input_ids,
            threshold=args.box_threshold, text_threshold=args.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        boxes = resultado["boxes"]
        scores = resultado["scores"]
        rotulos = resultado["text_labels"]

        if len(boxes) > 0:
            input_boxes = [[b.tolist() for b in boxes]]
            sam_inputs = sam_processor(images=image, input_boxes=input_boxes, return_tensors="pt").to(args.device)
            with torch.no_grad():
                sam_outputs = sam_model(**sam_inputs, multimask_output=False)
            mascaras = sam_processor.post_process_masks(
                sam_outputs.pred_masks.cpu(), sam_inputs["original_sizes"]
            )[0]  # (n_boxes, 1, H, W)

            overlay_path = ""
            if args.salvar_mascaras:
                overlay_path = os.path.join(dir_overlays, "%s_%d_%d.jpg" % (regional, px, py))

            deteccoes_patch = []
            for j in range(len(boxes)):
                score = float(scores[j])
                texto = rotulos[j]
                chave = prompt_para_chave.get(texto)
                if chave is None:
                    # correspondencia aproximada, caso o texto volte com pontuacao/espaco diferente
                    for p, k in prompt_para_chave.items():
                        if p in texto or texto in p:
                            chave = k
                            break
                box = [float(v) for v in boxes[j].tolist()]
                mascara_bin = mascaras[j, 0].numpy().astype(bool)
                area = int(mascara_bin.sum())
                deteccoes_patch.append((chave or "desconhecido", score, box, mascara_bin))
                w.writerow({
                    "patch": caminho, "regional": regional, "px": px, "py": py,
                    "classe": chave or "", "prompt": texto, "score": "%.4f" % score,
                    "bbox_x0": box[0], "bbox_y0": box[1], "bbox_x1": box[2], "bbox_y1": box[3],
                    "area_px": area, "mascara": overlay_path,
                })
                n_deteccoes += 1

            if args.salvar_mascaras and deteccoes_patch:
                desenhar_overlay(image, deteccoes_patch).save(overlay_path, quality=92)

        if (i + 1) % 20 == 0 or (i + 1) == len(arquivos):
            dt = time.time() - t0
            print("  %d/%d patches | %d deteccoes | %.1fs (%.2fs/patch)"
                  % (i + 1, len(arquivos), n_deteccoes, dt, dt / (i + 1)))

    f_csv.close()
    print("\nOK: %d deteccoes -> %s" % (n_deteccoes, destino_csv))


if __name__ == "__main__":
    main()
