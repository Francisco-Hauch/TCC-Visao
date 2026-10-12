#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verificar_integridade.py — confere, tile a tile, se uma regional está
realmente completa no disco. Estágio 1.5 do pipeline do TCC.

O contador "err" que o baixador imprime no fim é da SESSÃO, não do disco:
um tile que falhou numa passada e veio numa seguinte continua contado como
erro no manifesto. E um download interrompido no meio da escrita pode
deixar arquivo de 0 byte, que o resume trata como ausente mas ninguém lista.
Este script reconta arquivo por arquivo contra a faixa de tiles (x0..x1 ×
y0..y1) e diz o que falta de verdade.

O que ele classifica:
  presente   — arquivo existe, >0 byte e (com --profundo) é um JPEG válido
  faltando   — não existe no disco
  vazio      — existe com 0 byte  (apagado antes de rebaixar)
  corrompido — existe mas não é JPEG íntegro (só com --profundo)
  ausente    — o servidor não tem esse tile (404 / fora da área fotografada);
               fica registrado em ausentes_*.txt e não é tentado de novo

Uso:
  # só conferir (não toca na rede)
  python verificar_integridade.py --regiao regional_matriz --zoom 20 --saida D:\\TCC-data

  # conferir e rebaixar só o que falta
  python verificar_integridade.py --regiao regional_matriz --zoom 20 --saida D:\\TCC-data --corrigir

  # checagem forte (abre cada JPEG); mais lenta, boa antes de fechar o dataset
  python verificar_integridade.py --regiao regional_matriz --zoom 20 --saida D:\\TCC-data --profundo

  # todas as regionais de uma vez
  python verificar_integridade.py --todas --zoom 20 --saida D:\\TCC-data --corrigir
"""

import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baixar_ortofotos_ippuc as bo   # noqa: E402  (reusa bbox, tile_range, download_one)


# --------------------------------------------------------------------------
# Faixa de tiles: do manifesto se houver, senão calculada do bbox da região
# --------------------------------------------------------------------------
def faixa_de_tiles(saida, acervo, regiao, z):
    mpath = os.path.join(saida, f"manifesto_{acervo}_{regiao}_z{z}.json")
    if os.path.exists(mpath):
        try:
            with open(mpath, encoding="utf-8") as f:
                m = json.load(f)
            r = m["tile_range"]
            return (r["x0"], r["y0"], r["x1"], r["y1"]), mpath
        except Exception as e:
            print(f"[aviso] manifesto ilegível ({e}) — usando o bbox da região.",
                  file=sys.stderr)
    if regiao not in bo.REGIOES:
        raise SystemExit(f"região desconhecida: {regiao}")
    return bo.tile_range(bo.REGIOES[regiao], z), None


def jpeg_integro(caminho):
    """Checagem barata de JPEG: assinatura no começo e marcador EOI no fim."""
    try:
        tam = os.path.getsize(caminho)
        if tam < 128:
            return False
        with open(caminho, "rb") as f:
            if f.read(3) != b"\xff\xd8\xff":
                return False
            f.seek(-2, os.SEEK_END)
            return f.read(2) == b"\xff\xd9"
    except OSError:
        return False


def carrega_ausentes(path):
    if not os.path.exists(path):
        return set()
    fora = set()
    with open(path, encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha or linha.startswith("#"):
                continue
            try:
                x, y = linha.split("/")
                fora.add((int(x), int(y)))
            except ValueError:
                pass
    return fora


# --------------------------------------------------------------------------
def verifica_regiao(args, regiao):
    rng, mpath = faixa_de_tiles(args.saida, args.acervo, regiao, args.zoom)
    x0, y0, x1, y1 = rng
    total = (x1 - x0 + 1) * (y1 - y0 + 1)
    cache = os.path.join(args.saida, args.acervo, str(args.zoom))

    aus_path = os.path.join(args.saida, f"ausentes_{args.acervo}_{regiao}_z{args.zoom}.txt")
    ausentes = carrega_ausentes(aus_path)

    print("=" * 68)
    print(f"{regiao}  z={args.zoom}  ({args.acervo})")
    print(f"  faixa   : x {x0}..{x1} × y {y0}..{y1} = {total:,}".replace(",", ".") + " tiles")
    print(f"  manifesto: {mpath or '(não encontrado — faixa calculada do bbox)'}")
    if ausentes:
        print(f"  ausentes conhecidos no servidor: {len(ausentes):,}".replace(",", "."))
    print(f"  varrendo {cache} ...")

    faltando, vazios, corrompidos, presentes, pulados = [], [], [], 0, 0
    t0 = time.time()
    for x in range(x0, x1 + 1):
        dcol = os.path.join(cache, str(x))
        # listar a coluna inteira de uma vez é MUITO mais rápido que um
        # os.path.exists por tile em disco mecânico / pasta grande
        try:
            nomes = set(os.listdir(dcol))
        except OSError:
            nomes = set()
        for y in range(y0, y1 + 1):
            if (x, y) in ausentes:
                pulados += 1
                continue
            nome = f"{y}.jpg"
            if nome not in nomes:
                faltando.append((x, y))
                continue
            p = os.path.join(dcol, nome)
            try:
                tam = os.path.getsize(p)
            except OSError:
                faltando.append((x, y))
                continue
            if tam == 0:
                vazios.append((x, y))
            elif args.profundo and not jpeg_integro(p):
                corrompidos.append((x, y))
            else:
                presentes += 1

    dt = time.time() - t0
    ruins = faltando + vazios + corrompidos
    print(f"  presentes : {presentes:,}".replace(",", "."))
    print(f"  faltando  : {len(faltando):,}".replace(",", "."))
    print(f"  vazios    : {len(vazios):,}".replace(",", "."))
    if args.profundo:
        print(f"  corrompidos: {len(corrompidos):,}".replace(",", "."))
    if pulados:
        print(f"  pulados (ausentes no servidor): {pulados:,}".replace(",", "."))
    print(f"  varredura em {dt:.1f}s")

    lista_path = os.path.join(args.saida, f"faltantes_{args.acervo}_{regiao}_z{args.zoom}.txt")
    if ruins:
        with open(lista_path, "w", encoding="utf-8") as f:
            f.write(f"# {regiao} z{args.zoom} — {len(ruins)} tiles a rebaixar "
                    f"({time.strftime('%Y-%m-%d %H:%M:%S')})\n")
            for x, y in ruins:
                f.write(f"{x}/{y}\n")
        print(f"  lista: {lista_path}")
    elif os.path.exists(lista_path):
        os.remove(lista_path)

    if not ruins:
        print("  >> COMPLETA.")
        return {"regiao": regiao, "total": total, "presentes": presentes,
                "pendentes": 0, "completa": True}

    if not args.corrigir:
        print("  >> INCOMPLETA. Rode de novo com --corrigir para rebaixar só esses.")
        return {"regiao": regiao, "total": total, "presentes": presentes,
                "pendentes": len(ruins), "completa": False}

    # ---------------------------------------------------------------- corrigir
    for x, y in vazios + corrompidos:
        try:
            os.remove(os.path.join(cache, str(x), f"{y}.jpg"))
        except OSError:
            pass

    print(f"  rebaixando {len(ruins):,}".replace(",", ".") +
          f" tiles com {args.threads} threads...")
    ctx = bo._ssl_ctx(args.insecure)
    cont = {"ok": 0, "skip": 0, "miss": 0, "err": 0}
    lock = threading.Lock()
    novos_ausentes = []
    feitos = 0

    def tarefa(x, y):
        dest = os.path.join(cache, str(x), f"{y}.jpg")
        status, nb = bo.download_one(args.acervo, args.zoom, x, y, dest,
                                     ctx, args.retries, 0.0)
        return (x, y, status)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = [ex.submit(tarefa, x, y) for (x, y) in ruins]
        for fut in concurrent.futures.as_completed(futs):
            x, y, status = fut.result()
            with lock:
                cont[status] += 1
                feitos += 1
                if status == "miss":
                    novos_ausentes.append((x, y))
                if feitos % 200 == 0 or feitos == len(ruins):
                    print(f"    {feitos:,}/{len(ruins):,}".replace(",", ".") +
                          f"  ok={cont['ok']} miss={cont['miss']} err={cont['err']}",
                          flush=True)

    if novos_ausentes:
        with open(aus_path, "a", encoding="utf-8") as f:
            if not ausentes:
                f.write(f"# tiles que o servidor não tem (404/vazio) — {regiao} z{args.zoom}\n")
            for x, y in sorted(novos_ausentes):
                f.write(f"{x}/{y}\n")
        print(f"  {len(novos_ausentes):,}".replace(",", ".") +
              f" tiles não existem no servidor — anotados em {os.path.basename(aus_path)}")

    pend = cont["err"]
    print(f"  >> corrigidos {cont['ok']:,}".replace(",", ".") +
          f" · ainda com erro: {pend}")
    if pend:
        print("     (rode de novo — os 10054 do IPPUC são intermitentes)")
    else:
        print("  >> COMPLETA (contando os ausentes no servidor).")
        if os.path.exists(lista_path):
            os.remove(lista_path)

    return {"regiao": regiao, "total": total, "presentes": presentes + cont["ok"],
            "pendentes": pend, "completa": pend == 0}


def main():
    ap = argparse.ArgumentParser(
        description="Confere no disco, tile a tile, se as regionais estão completas.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--regiao", help="Uma região (ex.: regional_matriz).")
    g.add_argument("--todas", action="store_true",
                   help="Todas as 8 regionais do dataset do TCC.")
    ap.add_argument("--acervo", default="Ortofotos2019")
    ap.add_argument("--zoom", type=int, default=20)
    ap.add_argument("--saida", default="dataset")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--corrigir", action="store_true",
                    help="Rebaixa os tiles faltantes/vazios/corrompidos.")
    ap.add_argument("--profundo", action="store_true",
                    help="Também valida a integridade de cada JPEG (mais lento).")
    ap.add_argument("--insecure", action="store_true")
    args = ap.parse_args()

    if args.todas:
        regioes = [r for r in sorted(bo.REGIOES) if r.startswith("regional_")]
    else:
        regioes = [args.regiao]

    resumo = [verifica_regiao(args, r) for r in regioes]

    print("=" * 68)
    print("RESUMO")
    print(f"  {'regional':<28} {'esperado':>10} {'no disco':>10} {'pendente':>9}")
    for r in resumo:
        print(f"  {r['regiao']:<28} {r['total']:>10,} {r['presentes']:>10,} "
              f"{r['pendentes']:>9,}".replace(",", "."))
    faltam = [r["regiao"] for r in resumo if not r["completa"]]
    print("  TODAS COMPLETAS." if not faltam else "  incompletas: " + ", ".join(faltam))
    return 1 if faltam else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrompido.", file=sys.stderr)
        sys.exit(130)
