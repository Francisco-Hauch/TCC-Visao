"""
4ª rodada: área não edificada x TODAS as naturezas do Sigesguarda (2009-2020).
Critério e grupos escritos antes de rodar em hipoteses-ocorrencias-ruas.md,
seção "4ª rodada — todas as naturezas". Exploratório: correção FDR
(Benjamini-Hochberg) sobre os p do modelo B.

Mesmos modelos e controles da 2ª/3ª rodada (robustez_r3.py):
  A   volume (binomial negativa, offset log km)
  B   composição (fração das ocorrências da rua)
  B2  composição sem Trânsito e Perturbação no denominador

Entrada: TCC-data/ruas_caracteristicas_ocorrencias_2009_2020.csv (montar_tabela_ruas.py,
         uma coluna por natureza com >= 100 registros no período)
Saída:   TCC-data/resultado_r4_todas_naturezas.csv
         TCC-data/figuras_r4/todas_naturezas.png (rodar de novo sobrescreve)

Uso: .venv-analise\\Scripts\\python.exe Modelo\\scripts\\todas_naturezas_r4.py
"""
import numpy as np
import pandas as pd

from modelo_hipoteses_ruas import preparar
from robustez_r3 import DADOS, ajustar_A, ajustar_B, efeito, recorte

SAIDA = DADOS / "resultado_r4_todas_naturezas.csv"
FIGURAS = DADOS / "figuras_r4"
MIN_OC, MIN_RUAS = 100, 50

GRUPOS = {
    "G1 abandono físico/ambiental": [
        "queima_a_ceu_aberto", "deposito_irregular", "crime_ambiental",
        "poluicao_visual_ambiental", "corte_irregular_arvore", "incendio", "invasao"],
    "G2 vegetação e água": [
        "queda_arvore", "animais", "banho_em_local_improprio", "alagamento", "erosao"],
    "G3 crime e desordem de rua": [
        "roubo", "furto", "dano", "tentativa", "arrastao", "veiculo", "substancia_ilicita",
        "substancia_licita", "porte_ilegal", "disparo_arma", "atos_obscenos_libidinosos",
        "importunacao_sexual", "fundada_suspeita_abordagem", "atitude_suspeita", "embriaguez",
        "conduta_inconveniente"],
    "G4 interpessoal/doméstico": [
        "ameaca", "agressao_fisica_verbal", "vias_fato", "desinteligencia",
        "violacao_medida_protetiva_lei_maria_penha", "lesao_corporal", "perturbacao_sossego",
        "abandono_incapaz", "crianca_perdida_desaparecida", "desaparecimento",
        "paciente_usuario_alterado", "obito"],
    "G5 operacional/controle negativo": [
        "apoio", "orientacao", "alarmes", "encaminhamento", "vistoria", "ronda", "averiguacao",
        "escolta", "aifu", "achado", "desacato", "desobediencia", "manifestacao",
        "uso_indevido_cartao_transporte", "transito", "risco_acidente_a_vida", "obstrucao_via",
        "avaria_em_equipamento_patrimonio_nao_intencional", "fornecimento_lona", "destelhamento",
        "desabamento", "risco_desabamento_desmoronamento", "situacao_risco",
        "vazamento_ou_derramamento_produto_perigoso_ou_infectante"],
}


def bh(p):
    """q-valores de Benjamini-Hochberg."""
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def figura(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURAS.mkdir(exist_ok=True)
    r = r.sort_values(["grupo", "B_efeito"], ascending=[False, True]).reset_index(drop=True)
    cores = dict(zip(GRUPOS, ["#c0392b", "#27ae60", "#8e44ad", "#2980b9", "#7f8c8d"]))
    fig, ax = plt.subplots(figsize=(8, 0.22 * len(r) + 1))
    for i, l in r.iterrows():
        c = cores[l["grupo"]]
        ax.plot([l["B_ic95_inf"], l["B_ic95_sup"]], [i, i], color=c, lw=1.2)
        ax.plot(l["B_efeito"], i, "o", color=c, mfc=c if l["B_q"] < 0.05 else "white", ms=5)
    ax.axvline(1, color="k", lw=0.8)
    ax.set_yticks(range(len(r)), [n[:38] for n in r["natureza"]], fontsize=7)
    ax.set_xlabel("efeito na composição por +10 p.p. de área não edificada (cheio = FDR q<0,05)")
    for g, c in cores.items():
        ax.plot([], [], "o", color=c, label=g)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURAS / "todas_naturezas.png", dpi=150)
    print(f"-> {FIGURAS / 'todas_naturezas.png'}")


def main():
    b = preparar((2009, 2020))
    grupo_de = {"oc_" + n: g for g, ns in GRUPOS.items() for n in ns}
    cols = [c for c in b.columns if c.startswith("oc_") and c != "oc_total"
            and b[c].sum() >= MIN_OC and (b[c] > 0).sum() >= MIN_RUAS]
    sem_grupo = [c for c in cols if c not in grupo_de]
    if sem_grupo:
        raise SystemExit(f"natureza sem grupo definido antes de rodar: {sem_grupo}")
    n_b2 = b["oc_total"] - b["oc_transito"] - b["oc_perturbacao_sossego"]
    print(f"{len(b)} ruas; {len(cols)} naturezas testadas")
    linhas = []
    for c in cols:
        t, y = recorte(b, [c])
        n2 = n_b2[b["bairro"].isin(t["bairro"].unique())].reset_index(drop=True)
        try:  # A pode não convergir em natureza rara (fica NaN; o teste é o B)
            ra = efeito(ajustar_A(t, y)[0])
        except np.linalg.LinAlgError:
            ra = efeito(None)
        rb = efeito(ajustar_B(t, y)[0])
        rb2 = efeito(ajustar_B(t, y, n=n2)[0]) if c not in ("oc_transito", "oc_perturbacao_sossego") \
            else dict.fromkeys(rb, np.nan)
        linhas.append({"natureza": c[3:], "grupo": grupo_de[c], "n_ocorrencias": int(y.sum()),
                       "ruas_com_oc": int((y > 0).sum()),
                       **{f"A_{k}": v for k, v in ra.items()},
                       **{f"B_{k}": v for k, v in rb.items()},
                       **{f"B2_{k}": v for k, v in rb2.items()}})
        print(f"  {c[3:40]:38s} A {ra['efeito']:.3f} p={ra['p']:.4f} | "
              f"B {rb['efeito']:.3f} p={rb['p']:.4f}")
    r = pd.DataFrame(linhas)
    r["B_q"] = bh(r["B_p"].fillna(1))
    r["B_bonferroni"] = (r["B_p"] * len(r)).clip(upper=1)
    r["sinal"] = np.where(r["B_q"] >= 0.05, "nulo", np.where(r["B_efeito"] > 1, "+", "-"))
    r = r.sort_values(["grupo", "B_efeito"], ascending=[True, False])
    r.to_csv(SAIDA, sep=";", index=False, encoding="utf-8-sig", float_format="%.4f")
    print(f"-> {SAIDA}")

    print("\nresumo por grupo (B, FDR q<0,05):")
    for g, d in r.groupby("grupo"):
        print(f"  {g:34s} n={len(d):2d}  + {(d['sinal'] == '+').sum():2d}  "
              f"- {(d['sinal'] == '-').sum():2d}  nulo {(d['sinal'] == 'nulo').sum():2d}")
    g5 = r[r["grupo"].str.startswith("G5")]
    print(f"\nG5 positivos significativos: {(g5['sinal'] == '+').mean():.0%} (limite 10%)")
    figura(r)


if __name__ == "__main__":
    main()
