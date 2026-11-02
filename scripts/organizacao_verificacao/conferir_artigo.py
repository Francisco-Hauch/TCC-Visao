"""
Confere, valor a valor, os números do artigo contra as SAÍDAS DOS SCRIPTS
(nenhum número é digitado aqui a não ser o que está escrito no artigo).

Cada linha: seção | o que é | valor no artigo | valor lido/derivado das saídas |
OK/DIFERE. O valor das saídas é arredondado para o mesmo número de casas que o
artigo usa antes de comparar. Itens que dependem só da revisão manual de 400
não entram.

Entradas (todas regeradas pelos scripts do pipeline):
  TCC-data/ruas_caracteristicas_ocorrencias_2009_2020.csv / _2009_2022.csv
  TCC-data/patches512_rua_objetos.csv, eixos_logradouro.geojson, quadras_cadastrais.geojson
  TCC-data/resultado_hipoteses_2009_2020.csv / _2009_2022.csv
  TCC-data/resultado_r3_*.csv, resultado_r4_todas_naturezas.csv, resultado_r5_onibus_area.csv
  Modelo/dataset (contagens)

Uso: python conferir_artigo.py [--csv conferencia_artigo.csv]
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "essenciais"))
from modelo_hipoteses_ruas import preparar  # mesmo recorte de ruas dos modelos

RAIZ = Path(__file__).resolve().parents[2] / "Modelo"  # Modelo/
DADOS = RAIZ.parent / "TCC-data"


def ler(nome):
    return pd.read_csv(DADOS / nome, sep=";", encoding="utf-8-sig")


def fmt(v, casas):
    """Arredonda meio-para-cima a partir do valor como gravado (as saídas CSV têm 4
    casas; float binário faria 1.1150 virar 1.11)."""
    if isinstance(v, str):
        return v
    from decimal import Decimal, ROUND_HALF_UP
    d = Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-casas), rounding=ROUND_HALF_UP)
    return f"{d:,.{casas}f}"


class Conferencia:
    def __init__(self):
        self.linhas = []

    def item(self, secao, desc, artigo, valor, casas=None, obs=""):
        if casas is None:
            casas = len(artigo.split(".")[1]) if "." in artigo.replace(",", "") else 0
        v = fmt(valor, casas)
        a = artigo
        ok = "OK" if v == a else "DIFERE"
        if ok == "DIFERE" and not isinstance(valor, str):
            esc = float(valor) * 10 ** casas  # empate exato em ,5: as duas convenções valem
            if abs(esc - math.floor(esc) - 0.5) < 1e-9 and a.replace(",", "") in (
                    f"{math.floor(esc) / 10 ** casas:.{casas}f}", f"{math.ceil(esc) / 10 ** casas:.{casas}f}"):
                ok, obs = "OK", (obs + "; " if obs else "") + f"empate exato {float(valor)} (meio)"
        self.linhas.append((secao, desc, a, v, ok, obs))

    def nao_verificavel(self, secao, desc, artigo, motivo):
        self.linhas.append((secao, desc, artigo, "—", "NÃO VERIFICÁVEL", motivo))

    def imprimir(self):
        sec = None
        for s, d, a, v, ok, obs in self.linhas:
            if s != sec:
                print(f"\n=== {s} ===")
                sec = s
            marca = {"OK": "  ok ", "DIFERE": " >>> ", "NÃO VERIFICÁVEL": "  ?  "}[ok]
            print(f"{marca}{d:58s} artigo: {a:>14s}   saída: {v:>14s}  {ok}" + (f"  ({obs})" if obs else ""))
        n = {k: sum(1 for l in self.linhas if l[4] == k) for k in ("OK", "DIFERE", "NÃO VERIFICÁVEL")}
        print(f"\nTOTAL: {len(self.linhas)} itens — OK {n['OK']}, DIFERE {n['DIFERE']}, "
              f"não verificável {n['NÃO VERIFICÁVEL']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    c = Conferencia()

    # ---------------- dataset e detector ----------------
    ds = RAIZ / "dataset"
    ntr = len(list((ds / "images" / "train").glob("*.jpg")))
    nva = len(list((ds / "images" / "val").glob("*.jpg")))
    c.item("Dataset", "imagens rotuladas (total)", "348", ntr + nva)
    c.item("Dataset", "imagens de treino", "272", ntr)
    c.item("Dataset", "imagens de validação", "76", nva)
    rc = pd.read_csv(RAIZ / "runs" / "yolo26x_obb_v4_xl" / "results.csv")
    rc.columns = rc.columns.str.strip()
    # critério de best.pt no Ultralytics 8.4.163 (gravado no próprio best.pt: fitness = mAP50-95)
    fit = rc["metrics/mAP50-95(B)"]
    best = rc.loc[fit.idxmax()]
    c.item("Treino (results.csv)", "época do best.pt", "82", int(best["epoch"]))
    c.item("Treino (results.csv)", "mAP50 (val) na melhor época", "0.736", best["metrics/mAP50(B)"])
    c.item("Treino (results.csv)", "mAP50-95 (val) na melhor época", "0.482", best["metrics/mAP50-95(B)"],
           obs="não está no artigo; CLAUDE.md")
    c.item("Treino (results.csv)", "precisão (val) na melhor época", "0.78", best["metrics/precision(B)"])
    c.item("Treino (results.csv)", "recall (val) na melhor época", "0.73", best["metrics/recall(B)"])
    c.nao_verificavel("Treino (revalidação)", "mAP50 por classe", "área 0.67 (menor)",
                      "ver saída 18 (validar_modelo.py): área 0.666, ônibus 0.657, faixa 0.883")
    p = pd.read_csv(DADOS / "patches512_rua_objetos.csv", sep=";", encoding="utf-8-sig",
                    usecols=["regional_pasta", "area_NaoEdificada"])
    c.item("Detector", "patches (tiles 512) na cidade", "45,523", len(p))
    tem = (p["area_NaoEdificada"] > 0)
    c.item("Detector", "% de patches com área não edificada", "54.7", 100 * tem.mean())
    por_reg = tem.groupby(p["regional_pasta"]).mean().drop("_sem_regional") * 100
    c.item("Detector", "regional com MENOR % de área (nome)", "Matriz",
           por_reg.idxmin().replace("_", " ").title(), obs=f"{por_reg.min():.1f}%")
    c.item("Detector", "% de área na Matriz", "38", por_reg["matriz"])
    c.item("Detector", "menor % entre regionais", "38", por_reg.min())
    c.item("Detector", "regional com MAIOR % (nome)", "Santa Felicidade",
           por_reg.idxmax().replace("_", " ").title())
    c.item("Detector", "maior % entre regionais", "81", por_reg.max())
    eixos = json.load(open(DADOS / "eixos_logradouro.geojson", encoding="utf-8"))
    c.item("Integração", "segmentos de eixo de logradouro", "42,236", len(eixos["features"]))

    # ---------------- recorte e casamento ----------------
    t20 = ler("ruas_caracteristicas_ocorrencias_2009_2020.csv")
    t22 = ler("ruas_caracteristicas_ocorrencias_2009_2022.csv")
    sem20 = ler("ocorrencias_sem_casamento_2009_2020.csv")
    casadas20 = int(t20["oc_total"].sum())
    nao20 = sem20.groupby("motivo")["ocorrencias"].sum()
    total20 = casadas20 + int(nao20.sum())
    c.item("Dados", "registros 2009–2020 (sistema antigo)", "323,007", total20)
    c.item("Dados", "registros casados com rua coberta", "234,600", casadas20)
    c.item("Dados", "% casados", "73", 100 * casadas20 / total20)
    c.item("Dados", "% rua sem ortofoto", "14", 100 * nao20["rua_sem_ortofoto"] / total20)
    c.item("Dados", "% nome fora do cadastro", "7.6", 100 * nao20["nome_fora_do_cadastro"] / total20)
    c.item("Dados", "% rua vista em outro bairro", "5.8", 100 * nao20["rua_vista_em_outro_bairro"] / total20)

    b = preparar((2009, 2020))
    c.item("Dados", "unidades (rua, bairro) modeladas", "5,712", len(b))
    c.item("Dados", "bairros", "64", b["bairro"].nunique())
    c.item("Dados", "km de rua nas unidades", "2,600", b["comprimento_m"].sum() / 1000, casas=0,
           obs="artigo arredonda para centena")
    c.item("Dados", "ocorrências nas unidades modeladas", "224,180", int(b["oc_total"].sum()))
    c.item("Dados", "ocorrências fora (ruas <3 tiles, sem renda/hierarquia)", "10,420",
           casadas20 - int(b["oc_total"].sum()))
    s = np.sort(b["oc_total"].values)[::-1]
    c.item("Dados", "% dos registros nos 5% de unidades com mais", "65",
           100 * s[:round(0.05 * len(s))].sum() / s.sum())
    c.item("Dados", "% de unidades sem queima", "80", 100 * (b["oc_queima_a_ceu_aberto"] == 0).mean())
    c.item("Dados", "registros de queima", "2,235", int(b["oc_queima_a_ceu_aberto"].sum()))
    c.item("Dados", "% de unidades sem depósito irregular", "97", 100 * (b["oc_deposito_irregular"] == 0).mean())
    c.item("Dados", "registros de depósito irregular", "294", int(b["oc_deposito_irregular"].sum()))

    # ---------------- Tabela I (médias anuais nas 5.712 unidades) ----------------
    b22 = t22.set_index(["rua_chave", "bairro"]).loc[b.set_index(["rua_chave", "bairro"]).index]
    b20 = b.set_index(["rua_chave", "bairro"])
    def media(col):
        a20 = b20[col].sum() / 12
        a2122 = (b22[col].sum() - b20[col].sum()) / 2
        return a20, a2122
    for rot, col, v20, v22 in [("todas", "oc_total", "18,682", "38,109"),
                               ("trânsito", "oc_transito", "998", "6,300"),
                               ("queima", "oc_queima_a_ceu_aberto", "186", "391"),
                               ("roubo", "oc_roubo", "1,050", "530")]:
        x20, x22 = media(col)
        c.item("Tabela I", f"média anual {rot} 2009–2020", v20, x20)
        c.item("Tabela I", f"média anual {rot} 2021–2022", v22, x22)
    tt20, tt22 = media("oc_transito"); to20, to22 = media("oc_total")
    c.item("Tabela I", "% trânsito 2009–2020", "5.3", 100 * tt20 / to20)
    c.item("Tabela I", "% trânsito 2021–2022", "16.5", 100 * tt22 / to22)

    # ---------------- Tabela II (três unidades) ----------------
    ex = {"CORONEL JOAO MARIA SOBRINHO|VISTA ALEGRE": ("12", "12", "0/0", "6,327", "28", "2", "0"),
          "BRASILIO PERY MOREIRA|PINHEIRINHO": ("11", "8", "1/3", "3,868", "52", "2", "8"),
          "PASTEUR|BATEL": ("8", "0", "0/3", "14,553", "12", "0", "0")}
    for k, (npat, narea, of, renda, tot, que, inv) in ex.items():
        rua, bai = k.split("|")
        r = t20[(t20["rua_chave"] == rua) & (t20["bairro"] == bai)].iloc[0]
        c.item("Tabela II", f"{rua[:22]}: tiles", npat, int(r["n_patches"]))
        c.item("Tabela II", f"{rua[:22]}: tiles com área", narea, int(r["area_NaoEdificada_bin"]))
        c.item("Tabela II", f"{rua[:22]}: ônibus/faixas", of, f"{int(r['ponto_onibus'])}/{int(r['faixa_pedestre'])}")
        c.item("Tabela II", f"{rua[:22]}: renda média", renda, float(r["renda_media"]))
        c.item("Tabela II", f"{rua[:22]}: ocorrências", tot, int(r["oc_total"]))
        c.item("Tabela II", f"{rua[:22]}: queima", que, int(r["oc_queima_a_ceu_aberto"]))
        c.item("Tabela II", f"{rua[:22]}: invasão", inv, int(r["oc_invasao"]))

    # ---------------- Figura 3 (sem controles) ----------------
    fr = b["area_NaoEdificada_bin"] / b["n_patches"]
    # intervalos fechados à direita: (0,20], (20,40], ... (uma rua com exatamente 20% fica no 1º grupo)
    grp = pd.cut(fr, [-0.001, .2, .4, .6, .8, 1.0], right=True, labels=["0-20", "20-40", "40-60", "60-80", "80-100"])
    g = b.groupby(grp, observed=False)
    perkm = g["oc_total"].sum() / (g["comprimento_m"].sum() / 1000)
    for lab, art in zip(perkm.index, ["98", "105", "90", "58", "57"]):
        c.item("Figura 3", f"ocorrências/km (2009–2020) grupo {lab}%", art, perkm[lab])
    q = g["oc_queima_a_ceu_aberto"].sum() / g["oc_total"].sum() * 100
    d = g["oc_deposito_irregular"].sum() / g["oc_total"].sum() * 100
    c.item("Figura 3", "% queima no grupo 0–20%", "0.7", q.iloc[0])
    c.item("Figura 3", "% queima no grupo 80–100%", "1.5", q.iloc[-1])
    c.item("Figura 3", "% depósito no grupo 0–20%", "0.13", d.iloc[0])
    c.item("Figura 3", "% depósito no grupo 80–100%", "0.25", d.iloc[-1])

    # ---------------- Tabela VI / modelos A e B ----------------
    def ef(arq, h, mod, campo="efeito"):
        r = ler(arq)
        r = r[(r["variavel"] == "area_nao_edificada") & r["hipotese"].str.startswith(h) & r["modelo"].str.startswith(mod)]
        return float(r[campo].iloc[0])
    A20 = "resultado_hipoteses_2009_2020.csv"
    tab6 = {"H1": ("1.09", "1.06", "1.13", "1.07", "1.03", "1.11"),
            "H2": ("1.15", "1.08", "1.23", "1.12", "1.03", "1.21"),
            "H3": ("1.03", "0.99", "1.07", "1.00", "0.96", "1.03"),
            "H4": ("1.03", "0.99", "1.06", "0.98", "0.96", "1.01"),
            "C ": ("1.01", "0.97", "1.04", "1.00", "0.97", "1.02")}
    for h, (ea, la, ua, eb, lb, ub) in tab6.items():
        for mod, e, l, u in (("A", ea, la, ua), ("B", eb, lb, ub)):
            c.item("Tabela VI", f"{h.strip()} modelo {mod} efeito", e, ef(A20, h, mod))
            c.item("Tabela VI", f"{h.strip()} modelo {mod} IC inf", l, ef(A20, h, mod, "ic95_inf"))
            c.item("Tabela VI", f"{h.strip()} modelo {mod} IC sup", u, ef(A20, h, mod, "ic95_sup"))
    c.item("Tabela VI", "H1 p no modelo B", "0.001", ef(A20, "H1", "B", "p"))
    c.item("Tabela VI", "H2 p no modelo B", "0.007", ef(A20, "H2", "B", "p"))
    A22 = "resultado_hipoteses_2009_2022.csv"
    for h, mod, art in [("H1", "A", "1.11"), ("H1", "B", "1.09"), ("H2", "A", "1.17"), ("H2", "B", "1.14")]:
        c.item("Sensibilidade 2009–2022", f"{h} modelo {mod}", art, ef(A22, h, mod))
    c.item("Sensibilidade 2009–2022", "trânsito modelo B (efeito)", "0.97", ef(A22, "C ", "B"),
           obs="artigo: 'leve efeito negativo'")

    # ---------------- Tabela VII / robustez ----------------
    mo = ler("resultado_r3_moran.csv")
    c.item("Tabela VII", "H1 Moran: menor p", "0.086", mo[mo["hipotese"].str.startswith("H1")]["p_perm"].min())
    c.item("Tabela VII", "H2 Moran: menor p", "0.18", mo[mo["hipotese"].str.startswith("H2")]["p_perm"].min())
    re_ = ler("resultado_r3_moran_reestimado.csv")
    def rr(h, ver, campo):
        x = re_[re_["hipotese"].str.startswith(h) & re_["versao"].str.startswith(ver)]
        return float(x[campo].iloc[0])
    for h, ver, e, pv in [("H1", "SLX", "1.06", "0.023"), ("H2", "SLX", "1.15", "0.002"),
                          ("H1", "cluster", "1.07", "0.003"), ("H2", "cluster", "1.12", "0.027")]:
        c.item("Tabela VII", f"{h} {ver} efeito B", e, rr(h, ver, "B_efeito"))
        c.item("Tabela VII", f"{h} {ver} p B", pv, rr(h, ver, "B_p"))
    co = ler("resultado_r3_composicao.csv")
    for h, e, pv in [("H1", "1.06", "0.004"), ("H2", "1.11", "0.012")]:
        x = co[co["desfecho"].str.startswith(h) & co["versao"].str.startswith("B2")].iloc[0]
        c.item("Tabela VII", f"{h} sem trânsito e perturbação (efeito)", e, x["efeito"])
        c.item("Tabela VII", f"{h} sem trânsito e perturbação (p)", pv, x["p"])
    cf = ler("resultado_r3_confianca.csv")
    def conf(cmin, h, mod, campo):
        x = cf[cf["conf_min"].astype(str).str.startswith(cmin) & cf["hipotese"].str.startswith(h)
               & cf["modelo"].str.startswith(mod)]
        return float(x[campo].iloc[0])
    for cmin, h, e, pv in [("0,5", "H1", "1.03", "0.12"), ("0,7", "H1", "1.05", "0.02"),
                           ("0,5", "H2", "1.11", "0.004")]:
        c.item("Tabela VII", f"{h} conf ≥{cmin} efeito B", e, conf(cmin, h, "B", "efeito"))
        c.item("Tabela VII", f"{h} conf ≥{cmin} p B", pv, conf(cmin, h, "B", "p"))
    c.item("Tabela VII", "H2 conf ≥0,7 efeito B", "1.15", conf("0,7", "H2", "B", "efeito"))
    c.item("Tabela VII", "H2 conf ≥0,7 p B < 0.001", "sim", "sim" if conf("0,7", "H2", "B", "p") < 0.001 else "não")
    for cmin, art in [("0,25", "44"), ("0,5", "31"), ("0,7", "19")]:
        c.item("Robustez (texto)", f"fração média de área por rua, conf ≥{cmin} (%)", art,
               conf(cmin, "H1", "B", "fracao_media_pct"))
    for cmin, h, art in [("0,25", "H1", "1.21"), ("0,5", "H1", "1.08"), ("0,7", "H1", "1.12"),
                         ("0,25", "H2", "1.38"), ("0,5", "H2", "1.33"), ("0,7", "H2", "1.40")]:
        c.item("Robustez (texto)", f"{h} efeito por DP, conf ≥{cmin}", art, conf(cmin, h, "B", "efeito_por_dp"))
    pe = ler("resultado_r3_fragilidade_permutacao.csv")
    for h in ("H1", "H2"):
        x = pe[pe["hipotese"].str.startswith(h)].iloc[0]
        c.item("Tabela VII", f"{h} permutação: nenhuma ≥ observado", "sim",
               "sim" if x["nulo_max"] < x["efeito_observado"] else "não",
               obs=f"p_perm = {x['p_perm']:.4f} = 1/1001")
    lo = ler("resultado_r3_fragilidade_lobo.csv")
    for h, mn, mx in [("H1", "1.05", "1.08"), ("H2", "1.09", "1.14")]:
        x = lo[lo["hipotese"].str.startswith(h)]
        c.item("Tabela VII", f"{h} deixa-um-bairro: menor efeito B", mn, x["B_efeito"].min())
        c.item("Tabela VII", f"{h} deixa-um-bairro: maior efeito B", mx, x["B_efeito"].max())
    x = lo[lo["hipotese"].str.startswith("H2")].sort_values("B_p").iloc[-1]
    c.item("Tabela VII", "H2 deixa-um-bairro: maior p (bairro)", "CIDADE INDUSTRIAL DE CURITIBA", x["bairro_fora"],
           obs="revisão do autor")
    c.item("Tabela VII", "H2 deixa-um-bairro: maior p", "0.031", x["B_p"], obs="revisão do autor")
    te = ler("resultado_r3_tamanho_efeito.csv")
    h1 = te[te["hipotese"].str.startswith("H1")].iloc[0]
    h2 = te[te["hipotese"].str.startswith("H2")].iloc[0]
    c.item("Tamanho do efeito", "rua local típica (m)", "273", h1["km_rua_tipica"] * 1000)
    c.item("Tamanho do efeito", "H1 razão 80%/20%", "1.7", h1["razao_80_20"])
    c.item("Tamanho do efeito", "H1 IC inf", "1.4", h1["razao_ic_inf"])
    c.item("Tamanho do efeito", "H1 IC sup", "2.1", h1["razao_ic_sup"])
    c.item("Tamanho do efeito", "H2 razão 80%/20%", "2.3", h2["razao_80_20"])
    c.item("Tamanho do efeito", "H2 IC inf", "1.5", h2["razao_ic_inf"])
    c.item("Tamanho do efeito", "H2 IC sup", "3.3", h2["razao_ic_sup"])
    c.item("Tamanho do efeito", "queimas a mais por 100 ruas/ano", "1", h1["a_mais_por_100_ruas_ano"],
           obs="artigo: 'cerca de uma'")
    loc = b[b["hierarquia"] == 4]
    c.item("Tamanho do efeito", "renda da rua local típica (R$)", "3,990", math.exp(loc["log_renda"].median()))
    so = ler("resultado_r3_sem_onibus.csv")
    for h, art in [("H1", "1.07"), ("H2", "1.12")]:
        x = so[so["desfecho"].str.startswith(h) & (so["versao"] == "sem ônibus")].iloc[0]
        c.item("Robustez (texto)", f"{h} modelo B sem controle de ônibus", art, x["B_efeito"])

    # ---------------- exploratório (r4, r5) ----------------
    r4 = ler("resultado_r4_todas_naturezas.csv")
    c.item("Tabela VIII", "naturezas testadas", "64", len(r4))
    tab8 = {"G1": ("7", "4", "0", "3"), "G2": ("5", "1", "2", "2"), "G3": ("16", "5", "1", "10"),
            "G4": ("12", "0", "1", "11"), "G5": ("24", "1", "3", "20")}
    for gk, (n, mais, menos, nulo) in tab8.items():
        x = r4[r4["grupo"].str.startswith(gk)]
        c.item("Tabela VIII", f"{gk} n", n, len(x))
        c.item("Tabela VIII", f"{gk} positivos", mais, int((x["sinal"] == "+").sum()))
        c.item("Tabela VIII", f"{gk} negativos", menos, int((x["sinal"] == "-").sum()))
        c.item("Tabela VIII", f"{gk} nulos", nulo, int((x["sinal"] == "nulo").sum()))
    for nat, art in [("crime_ambiental", "1.16"), ("corte_irregular_arvore", "1.14"),
                     ("deposito_irregular", "1.12"), ("queima_a_ceu_aberto", "1.07")]:
        c.item("Exploratório (texto)", f"{nat} efeito B", art, float(r4[r4["natureza"] == nat]["B_efeito"].iloc[0]))
    r5 = ler("resultado_r5_onibus_area.csv")
    def it(desf, mod):
        x = r5[r5["desfecho"].str.startswith(desf) & r5["modelo"].str.startswith(mod) & r5["termo"].str.startswith("intera")]
        return x.iloc[0]
    c.item("Exploratório (texto)", "roubo: interação ônibus×área (modelo A)", "1.01", it("Roubo", "A")["efeito"])
    c.item("Exploratório (texto)", "roubo: p da interação (modelo A)", "0.95", it("Roubo", "A")["p"])
    c.item("Exploratório (texto)", "trânsito: interação (modelo A)", "1.17", it("Trânsito", "A")["efeito"],
           obs="revisão do autor")
    c.item("Exploratório (texto)", "trânsito: p da interação (modelo A)", "0.04", it("Trânsito", "A")["p"],
           obs="revisão do autor")

    # ---------------- não verificáveis aqui ----------------
    c.nao_verificavel("Não verificáveis nesta rodada", "tiles z20 baixados", "180,319",
                      "exige a árvore Ortofotos2019 (180 mil arquivos)")
    c.nao_verificavel("Não verificáveis nesta rodada", "quadrantes vazios", "1,773 / 182,092",
                      "idem; 45.523×4 = 182.092 confere com o nº de patches")
    c.nao_verificavel("Não verificáveis nesta rodada", "tamanho do download", "2.4 GB", "idem")
    c.nao_verificavel("Não verificáveis nesta rodada", "conexões paralelas", "16 (autor: 8)",
                      "sem log do download; scripts têm padrão 16, README diz 8")
    c.nao_verificavel("Não verificáveis nesta rodada", "mAP50 do v1 nano", "0.718",
                      "pesos/run do v1 não estão na pasta (só o README)")

    c.imprimir()
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["secao", "item", "artigo", "saida_script", "status", "obs"])
            w.writerows(c.linhas)
        print(f"-> {a.csv}")


if __name__ == "__main__":
    main()
