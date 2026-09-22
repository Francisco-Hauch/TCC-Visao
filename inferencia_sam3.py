#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inferencia_sam3.py
====================
Estagio 5 (modelo): roda o SAM 3 (facebookresearch/sam3) zero-shot, por
prompt de texto, nos patches 512x512 do Estagio 4 (`Patches512\\<regional>\\
19\\<px>\\<py>.jpg`). Sem treino - so mede o que sai de graca, como
combinado no CLAUDE.md (secao 3.8/5).

Classes-alvo (2026-09-13): terreno baldio, calcada, entulho, lixo, lixeira,
ponto de onibus, tubo (estacao-tubo do BRT de Curitiba). Os prompts em
ingles abaixo sao um PONTO DE PARTIDA, nao verdade definitiva - SAM 3 e
open-vocabulary mas a qualidade do zero-shot muda muito com a redacao do
prompt. Espera-se iterar isso olhando os resultados (--salvar-mascaras
numa amostra pequena primeiro).

NAO instala nada e NAO baixa checkpoint - isso e feito uma vez, manualmente
(ver instrucoes no topo da secao "USO" abaixo). Este script so consome o
que ja estiver instalado/autenticado no ambiente de quem rodar.

USO
---
1) Ambiente (uma vez so, fora deste script):
     git clone https://github.com/facebookresearch/sam3.git
     cd sam3 && pip install -e .
   (isso traz torch/etc. como dependencia - confirme que veio a build com
   CUDA 12.6+; se nao, instale o torch certo primeiro a partir de
   https://pytorch.org/get-started/locally/)

2) Acesso ao checkpoint (uma vez so): pedir acesso em
   https://huggingface.co/facebook/sam3 , esperar aprovacao, gerar um
   access token e rodar `hf auth login` no terminal.

3) Rodar uma amostra pequena primeiro (controle de GPU nas suas maos -
   ajuste --amostra e --regional para o tamanho que quiser testar):

     python inferencia_sam3.py --regional matriz --amostra 20 --salvar-mascaras

   Depois, sem --amostra, roda a regional inteira; sem --regional, roda
   tudo (45.523 patches - caro, so depois de validar prompts na amostra).

Saida
-----
  <saida>\\deteccoes_sam3.csv   - 1 linha por deteccao (patch, classe, score, bbox)
  <saida>\\mascaras\\...        - PNG da mascara, so se --salvar-mascaras
"""

import argparse
import csv
import glob
import os
import random
import sys
import time

# Prompts em ingles - ponto de partida, ajustar depois de olhar resultado.
CLASSES = {
    # revisado em 2026-09-13 (3a rodada, ver inferencia_grounded_sam2.py):
    # "tubo" ja tinha saido (0/N acertos). "baldio" fica de fora POR AGORA -
    # vai ganhar via regra geometrica/fiscal, nao por detector generico.
    # "lixeira", "telhado" e "caminhao" tambem saem do foco desta rodada.
    "ponto_onibus":   "bus stop shelter",
    "lixo":           "garbage litter on the ground",
    "entulho":        "construction debris",
    "carro":          "car",
    # "moto" removida em 2026-09-13: de cima e um borrao pequeno demais,
    # nem DOTA/DIOR distinguem moto de carro pequeno.
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
    "rua":            (150, 150, 150),
    "calcada":        (0, 220, 220),
    "faixa_pedestre": (255, 215, 0),
    "painel_solar":   (0, 120, 220),
}


def desenhar_overlay(image, deteccoes):
    """image: PIL RGB. deteccoes: lista de (chave, score, box, mascara_bin).
    Devolve uma copia RGB com mascara semi-transparente + caixa colorida por
    classe (sem texto em cima de cada caixa) + legenda unica no canto."""
    import numpy as np
    from PIL import Image as PILImage, ImageDraw, ImageFont

    base = image.convert("RGBA")
    for chave, score, box, mascara_bin in deteccoes:
        cor = CORES.get(chave, (255, 255, 255))
        camada = np.zeros((*mascara_bin.shape, 4), dtype=np.uint8)
        camada[mascara_bin] = (*cor, 110)
        base = PILImage.alpha_composite(base, PILImage.fromarray(camada, mode="RGBA"))

    draw = ImageDraw.Draw(base)
    for chave, score, box, mascara_bin in deteccoes:
        cor = CORES.get(chave, (255, 255, 255))
        draw.rectangle(box, outline=cor, width=2)

    vistos = sorted({chave for chave, score, box, mascara_bin in deteccoes})
    if vistos:
        try:
            fonte = ImageFont.truetype("arial.ttf", 13)
        except Exception:
            fonte = ImageFont.load_default()
        linha_h = 17
        largura_legenda = 24 + max(len(c) for c in vistos) * 7
        draw.rectangle([2, 2, 2 + largura_legenda, 4 + linha_h * len(vistos)], fill=(0, 0, 0, 160))
        y = 4
        for chave in vistos:
            cor = CORES.get(chave, (255, 255, 255))
            draw.rectangle([6, y + 2, 20, y + 14], fill=cor)
            draw.text((24, y), chave, fill=(255, 255, 255), font=fonte)
            y += linha_h

    return base.convert("RGB")


UNIDADES = {
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
    """.../Patches512/<regional>/19/<px>/<py>.jpg -> (regional, px, py)"""
    partes = caminho.replace("\\", "/").split("/")
    py = int(os.path.splitext(partes[-1])[0])
    px = int(partes[-2])
    regional = partes[-4]
    return regional, px, py


def main():
    ap = argparse.ArgumentParser(description="Inferencia SAM 3 zero-shot nos patches 512x512.")
    ap.add_argument("--dados", default=r"D:\CityVision\TCC-data")
    ap.add_argument("--unidade", choices=sorted(UNIDADES), default="tile",
                    help="tile = z20 cru 256x256/34,5m (mais zoom relativo, padrao); "
                         "patch = z19 512x512/69m (Estagio 4, mais contexto)")
    ap.add_argument("--regional", default=None,
                    help="ex.: matriz, boa_vista, cic... (padrao: todas)")
    ap.add_argument("--classes", default=None,
                    help="lista separada por virgula (padrao: todas). ex.: baldio,calcada")
    ap.add_argument("--amostra", type=int, default=0, help="limita a N patches (aleatorio, seed fixa)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--score-min", type=float, default=0.3)
    ap.add_argument("--saida", default=None, help="padrao: <dados>\\SAM3_<unidade>")
    ap.add_argument("--salvar-mascaras", action="store_true",
                    help="grava 1 imagem por patch com todas as deteccoes desenhadas por cima (mascara + caixa + rotulo)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    saida = args.saida or os.path.join(args.dados, "SAM3_%s" % args.unidade)
    dir_overlays = os.path.join(saida, "overlays")
    os.makedirs(saida, exist_ok=True)
    if args.salvar_mascaras:
        os.makedirs(dir_overlays, exist_ok=True)

    classes = CLASSES
    if args.classes:
        chaves = [c.strip() for c in args.classes.split(",")]
        classes = {k: CLASSES[k] for k in chaves}

    print("[setup] carregando SAM 3 (pode demorar na primeira vez - baixa/le o checkpoint)...")
    try:
        import numpy as np
        from PIL import Image
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as e:
        sys.exit("Faltou instalar o ambiente do SAM 3 (ver docstring deste arquivo, secao USO/1). "
                  "Erro original: %s" % e)

    model = build_sam3_image_model()
    model = model.to(args.device)
    processor = Sam3Processor(model)

    print("[setup] listando imagens (unidade=%s, regional=%s, amostra=%s)..."
          % (args.unidade, args.regional, args.amostra or "todas"))
    arquivos = listar_patches(args.dados, args.unidade, args.regional, args.amostra, args.seed)
    print("      %d imagens, %d classes -> %d chamadas ao modelo"
          % (len(arquivos), len(classes), len(arquivos) * len(classes)))

    destino_csv = os.path.join(saida, "deteccoes_sam3.csv")
    campos = ["patch", "regional", "px", "py", "classe", "prompt", "score",
              "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "area_px", "mascara"]
    f_csv = open(destino_csv, "w", newline="", encoding="utf-8-sig")
    w = csv.DictWriter(f_csv, fieldnames=campos, delimiter=";")
    w.writeheader()

    t0 = time.time()
    n_deteccoes = 0
    for i, caminho in enumerate(arquivos):
        regional, px, py = parse_patch_path(caminho)
        image = Image.open(caminho).convert("RGB")
        estado = processor.set_image(image)

        overlay_path = ""
        if args.salvar_mascaras:
            overlay_path = os.path.join(dir_overlays, "%s_%d_%d.jpg" % (regional, px, py))
        deteccoes_patch = []

        for chave, prompt in classes.items():
            saida_modelo = processor.set_text_prompt(state=estado, prompt=prompt)
            masks, boxes, scores = saida_modelo["masks"], saida_modelo["boxes"], saida_modelo["scores"]
            for j in range(len(scores)):
                score = float(scores[j])
                if score < args.score_min:
                    continue
                box = [float(v) for v in boxes[j]]
                mascara_bin = np.asarray(masks[j]).astype(bool)
                area = int(mascara_bin.sum())
                deteccoes_patch.append((chave, score, box, mascara_bin))
                w.writerow({
                    "patch": caminho, "regional": regional, "px": px, "py": py,
                    "classe": chave, "prompt": prompt, "score": "%.4f" % score,
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
