"""
Modelo de ocorrências por rua (modelo-ocorrencias-ruas.md, versão simplificada).

Um modelo por natureza: regressão binomial negativa da contagem de
ocorrências de 2019 em cada (rua, bairro) contra as três características
detectadas pelo v4 XL. Entrada: ruas_caracteristicas_ocorrencias.csv
(montar_tabela_ruas.py).

- Offset log(n_patches): compara ocorrências por patch visto, para rua vista
  inteira e rua vista pela metade não serem tratadas igual.
- Características em densidade: objetos por 10 patches (~690 m de rua), para
  o efeito não ser só "rua maior tem mais de tudo".
- Erro-padrão agrupado por bairro (ruas do mesmo bairro não são independentes).
- Só pares com >= MIN_PATCHES patches.

Sem controle de renda (CLAUDE.md §7-f) e com area_NaoEdificada ainda não
validada (revisão de 400 pendente) — resultado preliminar.

Saída: TCC-data/resultado_modelo_ocorrencias_2019.csv

Uso (ambiente da análise):
  .venv-analise\\Scripts\\python.exe Modelo\\scripts\\modelo_ocorrencias_ruas.py
  .venv-analise\\Scripts\\python.exe Modelo\\scripts\\modelo_ocorrencias_ruas.py --anos 2009 2022
  (--anos lê ruas_caracteristicas_ocorrencias_INI_FIM.csv e grava
   resultado_modelo_ocorrencias_INI_FIM.csv)
  --area-binaria: area_NaoEdificada entra como presença (patches com >=1 por
   10 patches) em vez de contagem de caixas; saída com sufixo _bin
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

DADOS = Path("D:/CityVision/TCC-data")
ENTRADA = DADOS / "ruas_caracteristicas_ocorrencias{}.csv"
SAIDA = DADOS / "resultado_modelo_ocorrencias_{}.csv"

MIN_PATCHES = 3
POR = 10  # densidade = objetos por POR patches

CARACTERISTICAS = ["ponto_onibus", "faixa_pedestre", "area_NaoEdificada"]
NATUREZAS = {  # escolhidas em 2026-10-15
    "oc_dano": "Dano",
    "oc_invasao": "Invasão",
    "oc_roubo": "Roubo",
    "oc_furto": "Furto",
    "oc_transito": "Trânsito",
    "oc_substancia_ilicita": "Substância ilícita",
    "oc_perturbacao_sossego": "Perturbação do sossego",
    "oc_queima_a_ceu_aberto": "Queima a céu aberto",
    "oc_veiculo": "Veículo",
}


def leitura(efeito, p):
    if p >= 0.05:
        return "sem relação clara"
    pct = (efeito - 1) * 100
    return f"{pct:+.0f}% por +1 a cada {POR} patches"


def ajustar(y, X, offset, grupos):
    """Binomial negativa partindo do ajuste de Poisson; tenta newton, depois bfgs e nm."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inicio = sm.Poisson(y, X, offset=offset).fit(disp=0, maxiter=200).params
        inicio = np.append(inicio, 1.0)  # alpha inicial
        for metodo in ("newton", "bfgs", "nm"):
            try:
                m = sm.NegativeBinomial(y, X, offset=offset).fit(
                    start_params=inicio, method=metodo, disp=0, maxiter=2000,
                    cov_type="cluster", cov_kwds={"groups": grupos})
            except np.linalg.LinAlgError:
                continue
            if m.mle_retvals["converged"] and np.all(np.isfinite(m.bse)):
                return m
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anos", nargs=2, type=int, metavar=("INI", "FIM"))
    ap.add_argument("--area-binaria", action="store_true")
    args = ap.parse_args()
    anos = args.anos
    if anos:
        entrada = Path(str(ENTRADA).format(f"_{anos[0]}_{anos[1]}"))
        saida = Path(str(SAIDA).format(f"{anos[0]}_{anos[1]}"))
    else:
        entrada, saida = Path(str(ENTRADA).format("")), Path(str(SAIDA).format("2019"))
    t = pd.read_csv(entrada, sep=";", encoding="utf-8-sig")
    t = t[t["n_patches"] >= MIN_PATCHES].copy()
    caracteristicas = list(CARACTERISTICAS)
    if args.area_binaria:
        caracteristicas[caracteristicas.index("area_NaoEdificada")] = "area_NaoEdificada_bin"
        saida = saida.with_name(saida.stem + "_bin" + saida.suffix)
    print(f"{len(t)} pares (rua, bairro) com >= {MIN_PATCHES} patches, "
          f"{t['bairro'].nunique()} bairros\n")

    X = pd.DataFrame({c: t[c] / t["n_patches"] * POR for c in caracteristicas})
    X = sm.add_constant(X)
    offset = np.log(t["n_patches"])
    grupos = pd.factorize(t["bairro"])[0]

    linhas = []
    for col, nome in NATUREZAS.items():
        y = t[col]
        m = ajustar(y, X, offset, grupos)
        if m is None:
            print(f"=== {nome} ===  não convergiu em nenhum método, pulado\n")
            continue
        print(f"=== {nome} ===  ({int(y.sum())} ocorrências, {int((y > 0).sum())} pares com >=1,"
              f" alpha={m.params['alpha']:.2f}{'' if m.mle_retvals['converged'] else ', NÃO CONVERGIU'})")
        print(f"{'':20s} {'efeito':>7s} {'IC95%':>13s} {'p':>6s}")
        ic = m.conf_int()
        for c in caracteristicas:
            ef, p = np.exp(m.params[c]), m.pvalues[c]
            lo, hi = np.exp(ic.loc[c])
            print(f"{c:20s} {ef:7.2f} {lo:6.2f}-{hi:<6.2f} {p:6.3f}   {leitura(ef, p)}")
            linhas.append({"natureza": nome, "caracteristica": c, "efeito": ef,
                           "ic95_inf": lo, "ic95_sup": hi, "p": p,
                           "n_ocorrencias": int(y.sum()), "n_pares_com_ocorrencia": int((y > 0).sum()),
                           "n_pares": len(t), "alpha": m.params["alpha"],
                           "convergiu": m.mle_retvals["converged"]})
        print()

    pd.DataFrame(linhas).to_csv(saida, sep=";", index=False, encoding="utf-8-sig",
                                float_format="%.4f")
    print(f"-> {saida}")


if __name__ == "__main__":
    main()
