"""
3ª rodada: testes de robustez de H1 (queima) e H2 (depósito irregular).
Critérios escritos antes de rodar em hipoteses-ocorrencias-ruas.md, seção
"3ª rodada — testes de robustez". As hipóteses e os controles são os da 2ª
rodada (modelo_hipoteses_ruas.py); recorte 2009-2020.

  1 moran        I de Moran dos resíduos (kNN 8 e banda 500 m, 999 permutações);
                 H1/H2 reestimados com SLX (área média dos 8 vizinhos) e com
                 cluster em células de 1 km
  2 composicao   modelo B sem Trânsito no denominador (B1) e sem Trânsito +
                 Perturbação (B2), para H1-H4 e para os exploratórios
  3 confianca    área recalculada só com detecções conf >= 0,5 e >= 0,7
  4 renda        precisão do v4 XL na revisão de 400 por tercil de renda
  5 fragilidade  permutação da área dentro do bairro (1.000, modelo B) e
                 deixa-um-bairro-de-fora (A e B), H2 e H1
  6 tamanho      ocorrências previstas por km por ano numa rua local típica com
                 20% e 80% de área, bootstrap por bairro (500)

Entrada: TCC-data/ruas_caracteristicas_ocorrencias_2009_2020.csv (montar_tabela_ruas.py;
         lat_media/lon_media para o Moran, colunas _c050/_c070 para o corte de confiança)
Saídas:  TCC-data/resultado_r3_<teste>*.csv, figuras em TCC-data/figuras_r3/
         (rodar de novo sobrescreve)

Uso: .venv-analise\\Scripts\\python.exe Modelo\\scripts\\robustez_r3.py --teste 1 2 3 4
     (5 e 6 são lentos: rodar à parte)
"""
import argparse
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import norm

from modelo_hipoteses_ruas import ALFA, CONTROLES, HIPOTESES, INTERESSE, matriz, preparar

DADOS = Path("D:/CityVision/TCC-data")
RAIZ = Path(__file__).resolve().parent.parent
SAIDA = DADOS / "resultado_r3_{}.csv"
FIGURAS = DADOS / "figuras_r3"
ANOS = 12  # 2009-2020
SEMENTE = 20261018

H12 = ["H1 Queima a céu aberto", "H2 Depósito irregular"]
EXPLORATORIOS = {"Roubo": ["oc_roubo"], "Veículo": ["oc_veiculo"],
                 "Substância ilícita": ["oc_substancia_ilicita"], "Furto": ["oc_furto"]}


# --- modelos (mesma especificação de modelo_hipoteses_ruas.py) ------------------

def base(conf=""):
    t = preparar((2009, 2020), conf)
    lat0 = t["lat_media"].mean()
    t["x_m"] = (t["lon_media"] - t["lon_media"].mean()) * 111_320 * math.cos(math.radians(lat0))
    t["y_m"] = (t["lat_media"] - lat0) * 111_320
    return t


def recorte(b, cols):
    """Bairro sem nenhuma ocorrência do tipo X sai (mesma regra da 2ª rodada)."""
    com = b.groupby("bairro")[cols].sum().sum(axis=1)
    t = b[b["bairro"].isin(com[com > 0].index)].reset_index(drop=True)
    return t, t[cols].sum(axis=1)


def mat(t, extras=(), sem=()):
    X = matriz(t).drop(columns=list(sem))
    for c in extras:
        X[c] = t[c].values
    return X


def grupos_de(t, col):
    return pd.factorize(t[col])[0]


def ajustar_A(t, y, extras=(), cluster="bairro", robusto=True, sem=()):
    X = mat(t, extras, sem)
    kw = {"cov_type": "cluster", "cov_kwds": {"groups": grupos_de(t, cluster)}} if robusto else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inicio = sm.Poisson(y, X, offset=t["log_km"]).fit(disp=0, maxiter=500).params
        for metodo in ("newton", "bfgs"):
            try:
                m = sm.NegativeBinomial(y, X, offset=t["log_km"]).fit(
                    start_params=np.append(inicio, 1.0), method=metodo, disp=0, maxiter=3000, **kw)
            except np.linalg.LinAlgError:
                continue
            if m.mle_retvals["converged"] and np.isfinite(m.bse[INTERESSE]):
                return m, X
    return None, X


def ajustar_B(t, y, n=None, extras=(), cluster="bairro", robusto=True, sem=()):
    """Composição: y de n (padrão oc_total). Devolve (modelo, linhas usadas)."""
    n = t["oc_total"] if n is None else n
    ok = (n > 0).values
    t, y, n = t[ok].reset_index(drop=True), y[ok].reset_index(drop=True), n[ok].reset_index(drop=True)
    X = mat(t, extras, sem)
    kw = {"cov_type": "cluster", "cov_kwds": {"groups": grupos_de(t, cluster)}} if robusto else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = sm.GLM(np.column_stack([y, n - y]), X, family=sm.families.Binomial()).fit(**kw)
    return m, ok


def efeito(m, v=INTERESSE):
    if m is None:
        return {"efeito": np.nan, "ic95_inf": np.nan, "ic95_sup": np.nan, "p": np.nan}
    lo, hi = np.exp(m.conf_int().loc[v])
    return {"efeito": np.exp(m.params[v]), "ic95_inf": lo, "ic95_sup": hi, "p": m.pvalues[v]}


def veredito(efB, pB, efA=None):
    """Critério da 3ª rodada (escrito antes de rodar)."""
    if not np.isfinite(pB) or efB <= 1 or pB >= 0.05:
        return "cai"
    if pB >= ALFA or (efA is not None and efA <= 1):
        return "enfraquece"
    return "se mantém"


def pior(vs):
    ordem = ["se mantém", "enfraquece", "cai"]
    return max(vs, key=ordem.index)


def resid_A(m, X, t, y):
    mu = m.predict(X, offset=t["log_km"])
    return ((y - mu) / np.sqrt(mu + m.params["alpha"] * mu ** 2)).values


def salvar(df, nome):
    caminho = Path(str(SAIDA).format(nome))
    df.to_csv(caminho, sep=";", index=False, encoding="utf-8-sig", float_format="%.4f")
    print(f"-> {caminho}")


def fmt(r):
    return f"{r['efeito']:.3f} ({r['ic95_inf']:.2f}-{r['ic95_sup']:.2f}) p={r['p']:.4f}"


# --- 1. Moran ---------------------------------------------------------------------

def w_knn(xy, k=8):
    _, idx = cKDTree(xy).query(xy, k + 1)
    lin, col = [], []
    for i, viz in enumerate(idx):
        viz = [j for j in viz if j != i][:k]
        lin += [i] * len(viz)
        col += viz
    W = sparse.csr_matrix((np.ones(len(lin)), (lin, col)), shape=(len(xy), len(xy)))
    return sparse.diags(1 / np.asarray(W.sum(axis=1)).ravel()) @ W


def w_banda(xy, d=500.0):
    """Pesos binários até d metros, padronizados por linha; devolve (W, linhas sem ilha)."""
    pares = cKDTree(xy).query_pairs(d, output_type="ndarray")
    n = len(xy)
    W = sparse.csr_matrix((np.ones(2 * len(pares)),
                           (np.r_[pares[:, 0], pares[:, 1]], np.r_[pares[:, 1], pares[:, 0]])),
                          shape=(n, n))
    soma = np.asarray(W.sum(axis=1)).ravel()
    ok = soma > 0
    W = W[ok][:, ok]
    return sparse.diags(1 / np.asarray(W.sum(axis=1)).ravel()) @ W, ok


def moran(z, W, rng, nperm=999):
    z = z - z.mean()
    I = z @ (W @ z) / (z @ z)
    sim = np.array([(zp @ (W @ zp)) / (z @ z) for zp in (rng.permutation(z) for _ in range(nperm))])
    maiores = (sim >= I).sum()
    maiores = min(maiores, nperm - maiores)
    return I, -1 / (len(z) - 1), (maiores + 1) / (nperm + 1)


def teste_moran():
    rng = np.random.default_rng(SEMENTE)
    b = base()
    b["area_vizinhos"] = w_knn(b[["x_m", "y_m"]].values) @ b[INTERESSE].values
    b["celula_1km"] = (np.floor(b["x_m"] / 1000).astype(int).astype(str) + "_"
                       + np.floor(b["y_m"] / 1000).astype(int).astype(str))
    print(f"{len(b)} ruas; {b['celula_1km'].nunique()} células de 1 km")
    linhas, reest = [], []
    for nome, cols in HIPOTESES.items():
        t, y = recorte(b, cols)
        mA, XA = ajustar_A(t, y)
        mB, okB = ajustar_B(t, y)
        res = {"A volume": (resid_A(mA, XA, t, y), t), "B composição": (mB.resid_pearson, t[okB])}
        for modelo, (z, tt) in res.items():
            xy = tt[["x_m", "y_m"]].values
            for pesos, (W, ok) in (("knn8", (w_knn(xy), np.ones(len(z), bool))),
                                   ("banda500m", w_banda(xy))):
                I, EI, p = moran(np.asarray(z)[ok], W, rng)
                linhas.append({"hipotese": nome, "modelo": modelo, "pesos": pesos,
                               "n_ruas": int(ok.sum()), "I": I, "E_I": EI, "p_perm": p})
                print(f"  {nome[:24]:24s} {modelo:13s} {pesos:9s} n={ok.sum():5d} "
                      f"I={I:+.4f} p={p:.3f}")
        if nome in H12:
            for versao, kw in (("original (cluster bairro)", {}),
                               ("SLX (área dos 8 vizinhos)", {"extras": ["area_vizinhos"]}),
                               ("cluster célula 1 km", {"cluster": "celula_1km"})):
                ra = efeito(ajustar_A(t, y, **kw)[0])
                mb = ajustar_B(t, y, **kw)[0]
                rb = efeito(mb)
                viz = efeito(mb, "area_vizinhos") if "extras" in kw else None
                reest.append({"hipotese": nome, "versao": versao,
                              **{f"A_{k}": v for k, v in ra.items()},
                              **{f"B_{k}": v for k, v in rb.items()},
                              "B_area_vizinhos": viz["efeito"] if viz else np.nan,
                              "B_area_vizinhos_p": viz["p"] if viz else np.nan,
                              "veredito": veredito(rb["efeito"], rb["p"], ra["efeito"])})
                print(f"    {versao:28s} A {fmt(ra)} | B {fmt(rb)} -> {reest[-1]['veredito']}")
    salvar(pd.DataFrame(linhas), "moran")
    salvar(pd.DataFrame(reest), "moran_reestimado")


# --- 2. composição sem soma 100% --------------------------------------------------

def teste_composicao():
    b = base()
    linhas = []
    desfechos = [(n, c, "hipótese") for n, c in HIPOTESES.items() if not n.startswith("C")]
    desfechos += [(n, c, "exploratório") for n, c in EXPLORATORIOS.items()]
    for nome, cols, tipo in desfechos:
        t, y = recorte(b, cols)
        res = {}
        for versao, tira in (("B", []), ("B1 sem Trânsito", ["oc_transito"]),
                             ("B2 sem Trânsito e Perturbação",
                              ["oc_transito", "oc_perturbacao_sossego"])):
            num = [c for c in cols if c not in tira]
            yy = t[num].sum(axis=1)
            n = t["oc_total"] - t[tira].sum(axis=1)
            m, ok = ajustar_B(t, yy, n)
            r = efeito(m)
            res[versao] = r
            linhas.append({"desfecho": nome, "tipo": tipo, "versao": versao,
                           "numerador": "+".join(num), **r, "n_ruas": int(ok.sum()),
                           "n_ocorrencias": int(yy.sum())})
            print(f"  {nome[:26]:26s} {versao:30s} {fmt(r)}")
        b1, b2 = res["B1 sem Trânsito"], res["B2 sem Trânsito e Perturbação"]
        if tipo == "hipótese":
            v = pior([veredito(b1["efeito"], b1["p"]), veredito(b2["efeito"], b2["p"])])
        elif all(r["efeito"] > 1 and r["p"] < ALFA for r in (b1, b2)):
            v = "sobrevive"
        elif any(r["p"] >= 0.05 for r in (b1, b2)):
            v = "artefato da composição"
        else:
            v = "ambíguo"
        for l in linhas[-3:]:
            l["veredito"] = v
        print(f"  {'':26s} -> {v}")
    salvar(pd.DataFrame(linhas), "composicao")


# --- 3. corte de confiança --------------------------------------------------------

def veredito_conf(rb, ra):
    """Critério próprio do teste 3: só "cai" se a direção inverter."""
    if rb["efeito"] <= 1:
        return "cai"
    if rb["p"] >= ALFA or ra["efeito"] <= 1:
        return "enfraquece"
    return "se mantém"


def teste_confianca():
    linhas = []
    for corte, conf in (("0,25 (atual)", ""), ("0,5", "_c050"), ("0,7", "_c070")):
        b = base(conf)
        dp = b[INTERESSE].std()
        print(f"conf >= {corte}: fração média da rua com área = {b[INTERESSE].mean() * 10:.1f}%, "
              f"DP = {dp * 10:.1f} p.p.")
        for nome, cols in HIPOTESES.items():
            t, y = recorte(b, cols)
            mA = ajustar_A(t, y)[0]
            mB = ajustar_B(t, y)[0]
            ra, rb = efeito(mA), efeito(mB)
            for modelo, m, r in (("A volume", mA, ra), ("B composição", mB, rb)):
                linhas.append({"conf_min": corte, "hipotese": nome, "modelo": modelo, **r,
                               "efeito_por_dp": np.exp(m.params[INTERESSE] * dp),
                               "fracao_media_pct": b[INTERESSE].mean() * 10, "dp_pp": dp * 10,
                               "veredito": veredito_conf(rb, ra) if nome in H12 else ""})
            print(f"  {nome[:26]:26s} A {fmt(ra)} | B {fmt(rb)}")
    salvar(pd.DataFrame(linhas), "confianca")


# --- 4. viés do detector por renda ------------------------------------------------

def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, c - h, c + h


def teste_renda():
    est = pd.read_csv(RAIZ / "revisao_400_xl" / "estado_revisao.csv", sep=";", encoding="utf-8-sig")
    renda = pd.read_csv(DADOS / "patches512_renda.csv", sep=";", encoding="utf-8-sig")
    cortes = renda["renda_media"].dropna().quantile([1 / 3, 2 / 3]).values
    print(f"tercis de renda (todos os patches da cidade): {cortes[0]:.0f} | {cortes[1]:.0f}")
    est = est.merge(renda.rename(columns={"x": "tile_x", "y": "tile_y"}),
                    on=["tile_x", "tile_y"], how="left")
    est["tercil"] = pd.cut(est["renda_media"], [-np.inf, *cortes, np.inf],
                           labels=["1 baixa", "2 média", "3 alta"]).astype(str)
    est.loc[est["renda_media"].isna(), "tercil"] = "sem renda"
    linhas = []
    for c in ["area_NaoEdificada", "ponto_onibus", "faixa_pedestre"]:
        tm, tr = est[f"{c}_tem_modelo"] == 1, est[f"{c}_tem_revisao"] == 1
        est[f"{c}_pvp"], est[f"{c}_pfp"], est[f"{c}_pfn"] = tm & tr, tm & ~tr, ~tm & tr
        medidas = [("presença", "pvp", "pfp", "pfn")]
        if c != "area_NaoEdificada":
            medidas.append(("contagem", "vp", "fp", "fn"))
        for medida, vp, fp, fn in medidas:
            for terc, g in [("todos", est), *sorted(est.groupby("tercil"))]:
                VP, FP, FN = (int(g[f"{c}_{k}"].sum()) for k in (vp, fp, fn))
                pr, pl, ph = wilson(VP, VP + FP)
                rc, rl, rh = wilson(VP, VP + FN)
                linhas.append({"classe": c, "medida": medida, "tercil": terc, "patches": len(g),
                               "VP": VP, "FP": FP, "FN": FN, "precisao": pr, "precisao_ic_inf": pl,
                               "precisao_ic_sup": ph, "recall_400": rc, "recall_ic_inf": rl,
                               "recall_ic_sup": rh, "saldo_fp_menos_fn": FP - FN})
                print(f"  {c:18s} {medida:9s} {terc:9s} n={len(g):3d} VP={VP:3d} FP={FP:3d} "
                      f"FN={FN:3d} prec={pr:.2f} ({pl:.2f}-{ph:.2f}) rec400={rc:.2f} ({rl:.2f}-{rh:.2f})")
    salvar(pd.DataFrame(linhas), "renda_detector")


# --- 5. fragilidade ---------------------------------------------------------------

def teste_fragilidade(nperm=1000):
    rng = np.random.default_rng(SEMENTE)
    b = base()
    perm, lobo = [], []
    for nome in ["H2 Depósito irregular", "H1 Queima a céu aberto"]:
        cols = HIPOTESES[nome]
        t, y = recorte(b, cols)
        mB, ok = ajustar_B(t, y)
        obs = mB.params[INTERESSE]
        tb, yb = t[ok].reset_index(drop=True), y[ok].reset_index(drop=True)
        idx_grupo = [np.flatnonzero(tb["bairro"].values == g) for g in tb["bairro"].unique()]
        nulos, t0 = [], time.time()
        for r in range(nperm):
            tp = tb.copy()
            v = tp[INTERESSE].values.copy()
            for ix in idx_grupo:
                v[ix] = rng.permutation(v[ix])
            tp[INTERESSE] = v
            m = ajustar_B(tp, yb, robusto=False)[0]
            nulos.append(m.params[INTERESSE])
            if r % 200 == 199:
                print(f"  {nome[:20]} permutação {r + 1}/{nperm} ({time.time() - t0:.0f}s)")
        nulos = np.array(nulos)
        p = (1 + (nulos >= obs).sum()) / (nperm + 1)
        perm.append({"hipotese": nome, "efeito_observado": np.exp(obs),
                     "nulo_mediana": np.exp(np.median(nulos)), "nulo_p2_5": np.exp(np.percentile(nulos, 2.5)),
                     "nulo_p97_5": np.exp(np.percentile(nulos, 97.5)), "nulo_max": np.exp(nulos.max()),
                     "p_perm": p, "n_perm": nperm,
                     "veredito": "se mantém" if p < ALFA else "enfraquece" if p < 0.05 else "cai"})
        print(f"  {nome}: observado {np.exp(obs):.3f}, nulo 95% {perm[-1]['nulo_p2_5']:.3f}-"
              f"{perm[-1]['nulo_p97_5']:.3f}, p_perm={p:.4f}")
        _figura_perm(nome, nulos, obs)

        for bairro in sorted(t["bairro"].unique()):
            fora = (t["bairro"] != bairro).values
            tt, yy = t[fora].reset_index(drop=True), y[fora].reset_index(drop=True)
            ra, rb = efeito(ajustar_A(tt, yy)[0]), efeito(ajustar_B(tt, yy)[0])
            lobo.append({"hipotese": nome, "bairro_fora": bairro,
                         "ocorrencias_do_bairro": int(y[~fora].sum()),
                         **{f"A_{k}": v for k, v in ra.items()}, **{f"B_{k}": v for k, v in rb.items()}})
        lb = pd.DataFrame([l for l in lobo if l["hipotese"] == nome])
        print(f"  {nome} deixa-um-bairro-de-fora ({len(lb)}): B de {lb['B_efeito'].min():.3f} "
              f"a {lb['B_efeito'].max():.3f}; p máx {lb['B_p'].max():.4f} "
              f"(sem {lb.loc[lb['B_p'].idxmax(), 'bairro_fora']})")
    salvar(pd.DataFrame(perm), "fragilidade_permutacao")
    salvar(pd.DataFrame(lobo), "fragilidade_lobo")


def _figura_perm(nome, nulos, obs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURAS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(np.exp(nulos), bins=40, color="#999")
    ax.axvline(np.exp(obs), color="#c0392b", lw=2, label=f"observado {np.exp(obs):.3f}")
    ax.set_xlabel("efeito na composição por +10 p.p. de área (nulo: área embaralhada no bairro)")
    ax.set_title(nome)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURAS / f"permutacao_{nome[:2]}.png", dpi=150)
    plt.close(fig)


# --- 6. tamanho do efeito ---------------------------------------------------------

def taxas(m, X, tipico):
    """Ocorrências por km por ano na rua típica com 20% e 80% de área."""
    fe = np.append(0.0, m.params[[c for c in X.columns if c.startswith("b_")]].values)
    eta = m.params["const"] + np.median(fe) + sum(m.params[c] * v for c, v in tipico.items())
    return [math.exp(eta + m.params[INTERESSE] * a) / ANOS for a in (2.0, 8.0)]


def teste_tamanho(nboot=500):
    rng = np.random.default_rng(SEMENTE)
    b = base()
    loc = b[b["hierarquia"] == 4]
    tipico = {"ponto_onibus": loc["ponto_onibus"].median(), "faixa_pedestre": loc["faixa_pedestre"].median(),
              "via_principal": 0.0, "coletora": 0.0, "log_renda": loc["log_renda"].median()}
    km_rua = loc["comprimento_m"].median() / 1000
    print(f"rua local típica: {km_rua * 1000:.0f} m, renda {math.exp(tipico['log_renda']):.0f}, "
          f"ônibus {tipico['ponto_onibus']:.2f} e faixa {tipico['faixa_pedestre']:.2f} por 10 patches")
    linhas = []
    for nome in H12:
        t, y = recorte(b, HIPOTESES[nome])
        m, X = ajustar_A(t, y, robusto=False)
        r20, r80 = taxas(m, X, tipico)
        bairros = t["bairro"].unique()
        boot, falhas, t0 = [], 0, time.time()
        for r in range(nboot):
            partes = []
            for j, g in enumerate(rng.choice(bairros, len(bairros))):
                p = t[t["bairro"] == g].copy()
                p["bairro"] = f"{g}#{j}"
                partes.append(p)
            tb = pd.concat(partes, ignore_index=True)
            mb, Xb = ajustar_A(tb, tb[HIPOTESES[nome]].sum(axis=1), robusto=False)
            if mb is None:
                falhas += 1
                continue
            boot.append(taxas(mb, Xb, tipico))
            if r % 100 == 99:
                print(f"  {nome[:20]} bootstrap {r + 1}/{nboot} ({time.time() - t0:.0f}s)")
        bt = np.array(boot)
        dif = bt[:, 1] - bt[:, 0]
        ic = lambda v: np.percentile(v, [2.5, 97.5])
        linhas.append({"hipotese": nome, "km_rua_tipica": km_rua,
                       "taxa_20pct": r20, "taxa_20pct_ic_inf": ic(bt[:, 0])[0], "taxa_20pct_ic_sup": ic(bt[:, 0])[1],
                       "taxa_80pct": r80, "taxa_80pct_ic_inf": ic(bt[:, 1])[0], "taxa_80pct_ic_sup": ic(bt[:, 1])[1],
                       "diferenca_km_ano": r80 - r20, "dif_ic_inf": ic(dif)[0], "dif_ic_sup": ic(dif)[1],
                       "razao_80_20": r80 / r20, "razao_ic_inf": ic(bt[:, 1] / bt[:, 0])[0],
                       "razao_ic_sup": ic(bt[:, 1] / bt[:, 0])[1],
                       "a_mais_por_100_ruas_ano": 100 * km_rua * (r80 - r20),
                       "a_mais_100_ic_inf": 100 * km_rua * ic(dif)[0],
                       "a_mais_100_ic_sup": 100 * km_rua * ic(dif)[1],
                       "n_boot_ok": len(bt), "n_boot_falhou": falhas})
        l = linhas[-1]
        print(f"  {nome}: {r20:.3f} -> {r80:.3f} por km/ano (x{r80 / r20:.2f}); "
              f"+{l['a_mais_por_100_ruas_ano']:.1f} por 100 ruas/ano "
              f"(IC {l['a_mais_100_ic_inf']:.1f}-{l['a_mais_100_ic_sup']:.1f}); boot ok {len(bt)}")
    salvar(pd.DataFrame(linhas), "tamanho_efeito")


def sem_onibus():
    """Extra, fora do critério escrito antes: ponto_onibus está contaminado por
    caminhões e telhados (triagem visual da 3ª rodada), então H1-H4, C e os exploratórios
    são reestimados sem esse controle."""
    b = base()
    linhas = []
    for nome, cols in [*HIPOTESES.items(), *EXPLORATORIOS.items()]:
        t, y = recorte(b, cols)
        for versao, sem in (("com ônibus", ()), ("sem ônibus", ("ponto_onibus",))):
            ra, rb = efeito(ajustar_A(t, y, sem=sem)[0]), efeito(ajustar_B(t, y, sem=sem)[0])
            linhas.append({"desfecho": nome, "versao": versao, **{f"A_{k}": v for k, v in ra.items()},
                           **{f"B_{k}": v for k, v in rb.items()}})
            print(f"  {nome[:26]:26s} {versao:10s} A {fmt(ra)} | B {fmt(rb)}")
    salvar(pd.DataFrame(linhas), "sem_onibus")


TESTES = {1: teste_moran, 2: teste_composicao, 3: teste_confianca, 4: teste_renda,
          5: teste_fragilidade, 6: teste_tamanho, 42: sem_onibus}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teste", nargs="+", type=int, default=[1, 2, 3, 4])
    for n in ap.parse_args().teste:
        print(f"\n===== teste {n}: {TESTES[n].__name__} =====")
        TESTES[n]()


if __name__ == "__main__":
    main()
