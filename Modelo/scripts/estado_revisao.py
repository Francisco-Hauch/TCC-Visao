"""
Estado da revisão de 400 (revisao_400_xl): uma linha por patch, com a
contagem de cada classe segundo o modelo (predicoes_modelo/) e segundo a
revisão (.txt ao lado da imagem, que o rotulador sobrescreve).

Regras (decisão de 2026-10-18):
- area_NaoEdificada é BINÁRIA: só importa se o patch tem ou não tem. Caixa
  quebrada em várias, ou várias caixas na mesma área, não contam.
  Por patch: VP / FP / FN / VN de presença.
- ponto_onibus e faixa_pedestre são CONTAGEM: acerto = min(modelo, revisão),
  FP = excesso do modelo, FN = falta do modelo. É comparação por contagem,
  não por posição da caixa.

Estado do patch:
- pendente               ainda não revisado (posição depois de --ate)
- revisado_sem_mudanca   revisado e as contagens batem com o modelo
- revisado_alterado      alguma contagem mudou (marcado sozinho, mesmo sem --ate)
O rotulador não marca "revisado" (toda imagem pré-rotulada já aparece como
feita) e reescreve o .txt só de abrir a imagem, então "revisado" vem de
--ate: o último patch revisado, na mesma ordem da lista do rotulador. O
valor fica guardado no CSV e vale para as próximas execuções; --ate só avança.

Filtro de tamanho (2026-10-18, filtro_tamanho.py): caixa do MODELO menor que
o mínimo da classe no treino é descartada antes de contar (`_n_modelo`); a
contagem sem filtro fica em `_n_modelo_bruto`. A revisão (.txt) não é filtrada:
se uma caixa minúscula for objeto real e o revisor mantiver, conta como FN.

Confiança: maior confiança do modelo por classe no patch, de
deteccoes_lote_filtrado.csv (conf >= 0,25, já com filtro de tamanho), para medir precisão por faixa.

Saída: revisao_400_xl/estado_revisao.csv (;) + resumo na tela.

Uso:
  python estado_revisao.py                                   # atualiza contagens, mantém o --ate salvo
  python estado_revisao.py --ate bairro_novo_190512_300655   # revisei até esse patch (nome sem .jpg)
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

from filtro_tamanho import LOTE_FILTRADO, lados_obb, passa

RAIZ = Path(__file__).resolve().parent.parent
REVISAO = RAIZ / "revisao_400_xl"
PRED = REVISAO / "predicoes_modelo"
SAIDA = REVISAO / "estado_revisao.csv"
DETECCOES_CSV = LOTE_FILTRADO  # confiança máxima já sem as caixas abaixo do tamanho mínimo

CLASSES = ["area_NaoEdificada", "ponto_onibus", "faixa_pedestre"]  # id 0, 1, 2
BINARIAS = {"area_NaoEdificada"}
FAIXAS_CONF = [(0.25, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]


def ordem_rotulador(nome):
    """Mesma ordem do rotulador (Intl.Collator numeric): números comparados como número."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", nome)]


def contar(txt, filtrar=False):
    n = [0] * len(CLASSES)
    if txt.exists():
        for linha in txt.read_text(encoding="utf-8").split("\n"):
            v = linha.split()
            if not v:
                continue
            c = int(v[0])
            if filtrar and not passa(CLASSES[c], *lados_obb(list(map(float, v[1:])))):
                continue
            n[c] += 1
    return n


def conf_max_por_patch():
    """(x, y) -> {classe: maior confiança}."""
    conf = defaultdict(dict)
    with open(DETECCOES_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter=";"):
            k = (int(row["tile_x"]), int(row["tile_y"]))
            c, v = CLASSES[int(row["classe_id"])], float(row["confianca"])
            conf[k][c] = max(conf[k].get(c, 0.0), v)
    return conf


def comparar(classe, nm, nr):
    if classe in BINARIAS:
        m, r = nm > 0, nr > 0
        return {"vp": int(m and r), "fp": int(m and not r), "fn": int(r and not m)}
    return {"vp": min(nm, nr), "fp": max(nm - nr, 0), "fn": max(nr - nm, 0)}


def faixa(c):
    for lo, hi in FAIXAS_CONF:
        if lo <= c < hi:
            return f"{lo:.2f}-{min(hi, 1):.2f}"
    return "sem_det"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ate", help="último patch revisado (nome sem .jpg), na ordem do rotulador")
    args = ap.parse_args()

    nomes = sorted((p.stem for p in REVISAO.glob("*.jpg")), key=ordem_rotulador)
    ate_salvo = None
    if SAIDA.exists():
        with open(SAIDA, encoding="utf-8-sig") as f:
            ate_salvo = next((r["revisado_ate"] for r in csv.DictReader(f, delimiter=";")
                              if r["revisado_ate"]), None)
    pos = {n: i for i, n in enumerate(nomes)}
    for v in (args.ate, ate_salvo):
        if v and v not in pos:
            raise SystemExit(f"patch não encontrado na revisão: {v}")
    candidatos = [v for v in (args.ate, ate_salvo) if v]
    ate = max(candidatos, key=pos.get) if candidatos else None
    limite = pos[ate] if ate else -1

    conf = conf_max_por_patch()
    linhas = []
    for i, nome in enumerate(nomes):
        regional, x, y = nome.rsplit("_", 2)
        nb = contar(PRED / f"{nome}.txt")
        nm, nr = contar(PRED / f"{nome}.txt", filtrar=True), contar(REVISAO / f"{nome}.txt")
        alterado = nb != nr  # comparado com o que o revisor recebeu (sem filtro)
        estado = ("revisado_alterado" if alterado else
                  "revisado_sem_mudanca" if i <= limite else "pendente")
        lin = {"ordem": i + 1, "patch": nome, "regional": regional, "tile_x": x, "tile_y": y,
               "estado": estado}
        for j, c in enumerate(CLASSES):
            cm = conf.get((int(x), int(y)), {}).get(c, 0.0)
            lin[f"{c}_n_modelo_bruto"] = nb[j]
            lin[f"{c}_n_modelo"] = nm[j]
            lin[f"{c}_n_revisao"] = nr[j]
            lin[f"{c}_tem_modelo"] = int(nm[j] > 0)
            lin[f"{c}_tem_revisao"] = int(nr[j] > 0)
            lin[f"{c}_conf_max"] = f"{cm:.4f}" if cm else ""
            for k, v in comparar(c, nm[j], nr[j]).items():
                lin[f"{c}_{k}"] = v
        lin["revisado_ate"] = ate if i == 0 and ate else ""
        linhas.append(lin)

    with open(SAIDA, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]), delimiter=";")
        w.writeheader()
        w.writerows(linhas)

    rev = [l for l in linhas if l["estado"] != "pendente"]
    n_alt = sum(l["estado"] == "revisado_alterado" for l in linhas)
    print(f"{len(linhas)} patches | revisados: {len(rev)} (até {ate or '-'}; "
          f"{n_alt} alterados) | pendentes: {len(linhas) - len(rev)}")
    if ate and n_alt > sum(l["estado"] == "revisado_alterado" for l in linhas[:limite + 1]):
        print("  aviso: há patches alterados depois do --ate salvo — confira se o --ate está certo")
    if not rev:
        print(f"-> {SAIDA}")
        return

    print("\nsó patches revisados; area_NaoEdificada = presença por patch, "
          "ônibus/faixa = contagem")
    print(f"{'classe':20s} {'faixa conf':>10s} {'patches':>7s} {'VP':>5s} {'FP':>5s} {'FN':>5s} "
          f"{'precisão':>8s} {'recall':>6s}")
    for c in CLASSES:
        grupos = defaultdict(lambda: [0, 0, 0, 0])
        for l in rev:
            cm = float(l[f"{c}_conf_max"] or 0)
            for g in ("total", faixa(cm)):
                s = grupos[g]
                s[0] += 1
                s[1] += l[f"{c}_vp"]
                s[2] += l[f"{c}_fp"]
                s[3] += l[f"{c}_fn"]
        ordem = [f"{lo:.2f}-{min(hi, 1):.2f}" for lo, hi in FAIXAS_CONF] + ["sem_det", "total"]
        rotulo = c
        for g in ordem:
            if g not in grupos:
                continue
            n, vp, fp, fn = grupos[g]
            prec = f"{vp / (vp + fp):.2f}" if vp + fp else "-"
            rec = f"{vp / (vp + fn):.2f}" if vp + fn else "-"
            print(f"{rotulo:20s} {g:>10s} {n:7d} {vp:5d} {fp:5d} {fn:5d} {prec:>8s} {rec:>6s}")
            rotulo = ""
        print()
    print(f"-> {SAIDA}")


if __name__ == "__main__":
    main()
