"""
2ª rodada do modelo de ocorrências por rua: testa as hipóteses H1-H4 e o
controle negativo C de hipoteses-ocorrencias-ruas.md (escritas antes de rodar).

Para cada hipótese, dois modelos com os MESMOS controles:
  A  volume      binomial negativa: ocorrências do tipo X na rua,
                 offset log(km de rua coberta pelas ortofotos)
  B  composição  binomial: fração das ocorrências da rua que é do tipo X
                 (só ruas com >= 1 ocorrência). Teste principal.

Variável de interesse: area_NaoEdificada_bin / n_patches * 10, ou seja, o
efeito é por +10 pontos percentuais da rua com área não edificada.
Controles: ponto_onibus e faixa_pedestre por 10 patches, hierarquia viária
(via principal = 1 ou 2, coletora = 3, local = 4 como referência),
log(renda média do setor 2022) e efeito fixo de bairro.
Erro-padrão agrupado por bairro. Bairro sem nenhuma ocorrência do tipo X sai
daquele modelo (não informa a comparação dentro do bairro).

Critério (Bonferroni, 4 hipóteses): área com p < 0,0125.

Entrada: TCC-data/ruas_caracteristicas_ocorrencias_INI_FIM.csv
Saída:   TCC-data/resultado_hipoteses_INI_FIM.csv (todos os coeficientes
         menos os efeitos fixos de bairro)

Uso: .venv-analise\\Scripts\\python.exe Modelo\\scripts\\modelo_hipoteses_ruas.py [--anos 2009 2020]
(padrão 2009-2022; 2026-10-18: 2009-2020 passou a ser o recorte usado, §6-6)
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

DADOS = Path("D:/CityVision/TCC-data")
ENTRADA = DADOS / "ruas_caracteristicas_ocorrencias_{}_{}.csv"
SAIDA = DADOS / "resultado_hipoteses_{}_{}.csv"

MIN_PATCHES = 3
POR = 10
ALFA = 0.05 / 4

CESTA = ["oc_dano", "oc_invasao", "oc_perturbacao_sossego", "oc_deposito_irregular",
         "oc_crime_ambiental", "oc_furto", "oc_substancia_ilicita"]
HIPOTESES = {
    "H1 Queima a céu aberto": ["oc_queima_a_ceu_aberto"],
    "H2 Depósito irregular": ["oc_deposito_irregular"],
    "H3 Invasão": ["oc_invasao"],
    "H4 Cesta desordem": CESTA,
    "C  Trânsito (controle negativo)": ["oc_transito"],
}
INTERESSE = "area_nao_edificada"
CONTROLES = ["ponto_onibus", "faixa_pedestre", "via_principal", "coletora", "log_renda"]


def preparar(anos=(2009, 2022), conf=""):
    """conf: "" (detecções conf >= 0,25) ou "_c050" / "_c070" (colunas da mesma tabela
    recontadas com conf >= 0,5 / 0,7; teste de corte de confiança, robustez_r3.py)."""
    t = pd.read_csv(str(ENTRADA).format(*anos), sep=";", encoding="utf-8-sig")
    t = t[(t["n_patches"] >= MIN_PATCHES) & (t["comprimento_m"] > 0)
          & t["renda_media"].notna() & t["hierarquia"].notna()].copy()
    t[INTERESSE] = t["area_NaoEdificada_bin" + conf] / t["n_patches"] * POR
    t["ponto_onibus"] = t["ponto_onibus" + conf] / t["n_patches"] * POR
    t["faixa_pedestre"] = t["faixa_pedestre" + conf] / t["n_patches"] * POR
    t["via_principal"] = t["hierarquia"].isin([1, 2]).astype(float)
    t["coletora"] = (t["hierarquia"] == 3).astype(float)
    t["log_renda"] = np.log(t["renda_media"])
    t["log_km"] = np.log(t["comprimento_m"] / 1000)
    return t.reset_index(drop=True)


def matriz(t):
    fe = pd.get_dummies(t["bairro"], prefix="b", drop_first=True, dtype=float)
    return sm.add_constant(pd.concat([t[[INTERESSE] + CONTROLES], fe], axis=1))


def volume(t, y):
    X = matriz(t)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inicio = sm.Poisson(y, X, offset=t["log_km"]).fit(disp=0, maxiter=500).params
        for metodo in ("newton", "bfgs"):
            try:
                m = sm.NegativeBinomial(y, X, offset=t["log_km"]).fit(
                    start_params=np.append(inicio, 1.0), method=metodo, disp=0, maxiter=3000,
                    cov_type="cluster", cov_kwds={"groups": pd.factorize(t["bairro"])[0]})
            except np.linalg.LinAlgError:
                continue
            if m.mle_retvals["converged"] and np.isfinite(m.bse[INTERESSE]):
                return m
    return None


def composicao(t, y):
    ok = t["oc_total"] > 0
    t, y = t[ok].reset_index(drop=True), y[ok].reset_index(drop=True)
    X = matriz(t)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sm.GLM(np.column_stack([y, t["oc_total"] - y]), X,
                      family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(t["bairro"])[0]})


def leitura(ef, p):
    if p >= ALFA:
        return "sem evidência"
    return f"{(ef - 1) * 100:+.0f}% por +10 p.p. ({'apoia' if ef > 1 else 'contraria'})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anos", nargs=2, type=int, default=[2009, 2022], metavar=("INI", "FIM"))
    anos = ap.parse_args().anos
    base = preparar(anos)
    print(f"{len(base)} ruas (rua, bairro), {base['bairro'].nunique()} bairros; "
          f"critério p < {ALFA:.4f}\n")
    linhas = []
    for nome, cols in HIPOTESES.items():
        y_all = base[cols].sum(axis=1)
        # bairro sem nenhuma ocorrência do tipo X não informa a comparação dentro do bairro
        com = base.groupby("bairro")[cols].sum().sum(axis=1)
        t = base[base["bairro"].isin(com[com > 0].index)].reset_index(drop=True)
        y = y_all[base["bairro"].isin(com[com > 0].index)].reset_index(drop=True)
        print(f"=== {nome} ===  {int(y.sum())} ocorrências, {int((y > 0).sum())} ruas com >=1, "
              f"{len(t)} ruas em {t['bairro'].nunique()} bairros")
        for modelo, ajustar in (("A volume", volume), ("B composição", composicao)):
            m = ajustar(t, y)
            if m is None:
                print(f"  {modelo:14s} não convergiu")
                continue
            ic = m.conf_int()
            for v in [INTERESSE] + CONTROLES:
                ef, p = np.exp(m.params[v]), m.pvalues[v]
                lo, hi = np.exp(ic.loc[v])
                linhas.append({"hipotese": nome, "modelo": modelo, "variavel": v,
                               "efeito": ef, "ic95_inf": lo, "ic95_sup": hi, "p": p,
                               "n_ruas": int(m.nobs), "n_ocorrencias": int(y.sum())})
            ef, p = np.exp(m.params[INTERESSE]), m.pvalues[INTERESSE]
            lo, hi = np.exp(ic.loc[INTERESSE])
            print(f"  {modelo:14s} área: {ef:5.2f} ({lo:.2f}-{hi:.2f})  p={p:.4f}  {leitura(ef, p)}")
        print()

    saida = Path(str(SAIDA).format(*anos))
    pd.DataFrame(linhas).to_csv(saida, sep=";", index=False, encoding="utf-8-sig",
                                float_format="%.4f")
    print(f"-> {saida}")


if __name__ == "__main__":
    main()
