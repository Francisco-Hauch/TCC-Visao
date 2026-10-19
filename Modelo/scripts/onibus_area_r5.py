"""
5ª rodada (exploratória, hipoteses-ocorrencias-ruas.md, escrita antes de rodar):
ponto de ônibus + área não edificada no MESMO patch aumentam roubo além da soma
dos dois separados?

Cada patch da rua é de um tipo: nenhum (referência), só área, só ônibus, os
dois (colunas patch_* de montar_tabela_ruas.py). Entram as frações de patches
de cada tipo, por +10 p.p., com os controles da 2ª rodada menos o ônibus (que
já está nos tipos): faixa, hierarquia, log renda, efeito fixo de bairro, erro
agrupado por bairro. Modelos A (volume) e B (composição, principal), reaproveitados
de modelo_hipoteses_ruas.py.

Teste principal: contraste β_dois − β_área − β_ônibus > 0 (interação
multiplicativa), p < 0,05 em B e mesmo sinal em A. Trânsito = controle negativo.

Saída: TCC-data/resultado_r5_onibus_area.csv
Uso: .venv-analise\\Scripts\\python.exe Modelo\\scripts\\onibus_area_r5.py
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

import modelo_hipoteses_ruas as mh

ANOS = (2009, 2020)
SAIDA = mh.DADOS / "resultado_r5_onibus_area.csv"
TIPOS = {"so_area": "patch_so_area", "so_onibus": "patch_so_onibus",
         "os_dois": "patch_onibus_e_area"}
CONTROLES = ["faixa_pedestre", "via_principal", "coletora", "log_renda"]
DESFECHOS = {"Roubo": "oc_roubo", "Trânsito (controle negativo)": "oc_transito"}
ALFA = 0.05


def matriz(t):
    fe = pd.get_dummies(t["bairro"], prefix="b", drop_first=True, dtype=float)
    return sm.add_constant(pd.concat([t[list(TIPOS) + CONTROLES], fe], axis=1))


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
            if m.mle_retvals["converged"] and np.isfinite(m.bse["os_dois"]):
                return m
    return None


def composicao(t, y):
    ok = t["oc_total"] > 0
    t, y = t[ok].reset_index(drop=True), y[ok].reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sm.GLM(np.column_stack([y, t["oc_total"] - y]), matriz(t),
                      family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(t["bairro"])[0]})


def main():
    base = mh.preparar(ANOS)
    for v, col in TIPOS.items():
        base[v] = base[col] / base["n_patches"] * mh.POR
    nenhum = base["n_patches"] - base[list(TIPOS.values())].sum(axis=1)
    print(f"{len(base)} ruas, {base['bairro'].nunique()} bairros, {ANOS[0]}-{ANOS[1]}")
    print("patches por tipo: nenhum {}, ".format(int(nenhum.sum()))
          + ", ".join(f"{v} {int(base[c].sum())}" for v, c in TIPOS.items())
          + f"; ruas com >=1 'os dois': {(base['patch_onibus_e_area'] > 0).sum()}\n")

    linhas = []
    for nome, col in DESFECHOS.items():
        com = base.groupby("bairro")[col].sum()
        dentro = base["bairro"].isin(com[com > 0].index)
        t, y = base[dentro].reset_index(drop=True), base.loc[dentro, col].reset_index(drop=True)
        print(f"=== {nome} ===  {int(y.sum())} ocorrências em {len(t)} ruas")
        for modelo, ajustar in (("A volume", volume), ("B composição", composicao)):
            m = ajustar(t, y)
            if m is None:
                print(f"  {modelo:14s} não convergiu")
                continue
            ic = m.conf_int()
            efeitos = []
            for v in TIPOS:
                ef, p = np.exp(m.params[v]), m.pvalues[v]
                lo, hi = np.exp(ic.loc[v])
                efeitos.append(f"{v} {ef:.2f} (p={p:.3f})")
                linhas.append({"desfecho": nome, "modelo": modelo, "termo": v + " vs nenhum",
                               "efeito": ef, "ic95_inf": lo, "ic95_sup": hi, "p": p,
                               "n_ruas": int(m.nobs)})
            # interação: os dois além da soma dos dois separados (escala log)
            c = np.zeros(len(m.params))
            nomes = list(m.params.index)
            c[nomes.index("os_dois")], c[nomes.index("so_area")], c[nomes.index("so_onibus")] = 1, -1, -1
            tt = m.t_test(c)
            ef = float(np.exp(tt.effect[0]))
            lo, hi = np.exp(np.asarray(tt.conf_int()).ravel())
            p = float(tt.pvalue)
            linhas.append({"desfecho": nome, "modelo": modelo,
                           "termo": "interação (dois − área − ônibus)", "efeito": ef,
                           "ic95_inf": lo, "ic95_sup": hi, "p": p, "n_ruas": int(m.nobs)})
            print(f"  {modelo:14s} vs nenhum: " + "; ".join(efeitos))
            print(f"  {'':14s} INTERAÇÃO {ef:.3f} ({lo:.3f}-{hi:.3f}) p={p:.4f}"
                  f"{'  <- p<0,05' if p < ALFA else ''}")
        print()

    pd.DataFrame(linhas).to_csv(SAIDA, sep=";", index=False, encoding="utf-8-sig",
                                float_format="%.4f")
    print(f"-> {SAIDA}")


if __name__ == "__main__":
    main()
