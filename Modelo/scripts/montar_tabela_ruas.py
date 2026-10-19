"""
Monta a tabela rua x características x ocorrências para o modelo de
ocorrências por rua (modelo-ocorrencias-ruas.md, versão simplificada).

Características: patches512_rua_objetos.csv (contagens do v4 XL, conf >= 0,5).
Cada patch conta para TODAS as ruas da coluna `ruas` (esquina/cruzamento
conta para as duas — decisão de 2026-10-15). Patch sem rua cruzando usa a
rua confrontante da quadra (`rua_quadra`).

Ocorrências: um único arquivo do Sigesguarda (sistema antigo, acumulado),
recorte só 2019 (ano das ortofotos), NATUREZA1_DESCRICAO, casado por LOGRADOURO_NOME +
ATENDIMENTO_BAIRRO_NOME.

Chave de casamento = (rua normalizada, bairro) — decisão de 2026-10-15.
Rua: sem acento, maiúsculas, sem tipo (RUA/AV./...), abreviações por
extenso, sem DE/DA/DO/DOS/DAS. Bairro do lado das características: o do
trecho de eixo DAQUELA rua mais próximo do centro do patch (não o bairro da
rua principal do patch). A ocorrência só entra se o par (rua, bairro) foi
visto em algum patch — é o mesmo filtro espacial das imagens: bairro ou
trecho sem ortofoto sai sozinho. Rua de divisa: ocorrência registrada no
bairro do outro lado do eixo é atribuída ao par do lado que foi visto.

Saídas (TCC-data/):
  ruas_caracteristicas_ocorrencias.csv  — uma linha por (rua, bairro), inclui 0
  ocorrencias_sem_casamento.csv         — (logradouro, bairro) que não casaram, com motivo

Uso:
  python montar_tabela_ruas.py                    # só 2019 (saídas sem sufixo)
  python montar_tabela_ruas.py --anos 2009 2020   # recorte principal (saídas com _2009_2020)
  python montar_tabela_ruas.py --anos 2009 2022   # sensibilidade

2026-10-18 (limpeza): UMA tabela por período, com tudo que as análises usam —
  - uma coluna oc_* por natureza com >= MIN_NATUREZA registros no período
    (além das NATUREZAS fixas);
  - contagens das 3 classes também com conf >= 0,5 e >= 0,7 (colunas _c050,
    _c070), recontadas de deteccoes_lote_filtrado.csv, para o teste de corte de
    confiança (robustez_r3.py) sem tabelas à parte;
  - lat_media/lon_media = média dos centros dos patches do par (I de Moran).
Rodar de novo sobrescreve a tabela do período.
"""
import argparse
import csv
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

DADOS = Path("D:/CityVision/TCC-data")
PATCHES = DADOS / "patches512_rua_objetos.csv"
EIXOS = DADOS / "eixos_logradouro.geojson"
RENDA = DADOS / "patches512_renda.csv"  # renda_por_patch.py (setor censitário 2022)
DETECCOES = Path("D:/CityVision/Modelo/runs/yolo26x_obb_v4_xl/deteccoes_lote_filtrado.csv")
SIGESGUARDA = DADOS / "Sigesguarda" / "2024-04-01_sigesguarda_-_Base_de_Dados.csv"
SAIDA = DADOS / "ruas_caracteristicas_ocorrencias{}.csv"
SAIDA_SEM = DADOS / "ocorrencias_sem_casamento{}.csv"

# padrão: mesmo ano das ortofotos (decisão de 2026-10-15); --anos amplia o período
# (2026-10-17: base completa 2009-2022, assumindo que a rua muda pouco no tempo)
ANO_INI, ANO_FIM = 2019, 2019
CLASSES = ["ponto_onibus", "faixa_pedestre", "area_NaoEdificada"]
# área não edificada também como presença (patches com >=1), não só contagem: uma área
# grande costuma sair em várias caixas no mesmo patch (decisão de 2026-10-18)
BINARIAS = ["area_NaoEdificada"]
COLS_BIN = [c + "_bin" for c in BINARIAS]
# cortes de confiança extras (o principal, 0,25, é o de patches512_rua_objetos.csv)
CORTES = {"_c050": 0.5, "_c070": 0.7}
COLS_CORTE = [c + s for s in CORTES for c in CLASSES + COLS_BIN]
# 5ª rodada (2026-10-19): tipo do patch por ônibus × área (conf 0,25); nenhum = n_patches - soma
COLS_TIPO = ["patch_so_area", "patch_so_onibus", "patch_onibus_e_area"]
MIN_NATUREZA = 100  # 4ª rodada: toda natureza com >= 100 registros no período vira coluna
# naturezas que entram no modelo (decisão de 2026-10-15); as demais não vão para a tabela
NATUREZAS = ["Dano", "Invasão", "Roubo", "Furto", "Trânsito", "Substância Ilícita",
             "Perturbação do sossego", "Queima a céu aberto", "Veículo",
             # 2026-10-18: para H2 e a cesta de desordem (hipoteses-ocorrencias-ruas.md)
             "Depósito irregular", "Crime ambiental"]
# 2026-10-18, controles da 2ª rodada (hipoteses-ocorrencias-ruas.md):
#   oc_total       todas as naturezas casadas na rua (base do modelo de composição)
#   comprimento_m  metros de eixo da rua cujo ponto médio cai num patch com ortofoto
#   hierarquia     hierarquia viária (1 via grande ... 4 local) de maior comprimento coberto
#   renda_media    média da renda do setor 2022 dos patches da rua
ZOOM_PATCH = 19

TIPOS = {
    "RUA", "R", "AVENIDA", "AV", "TRAVESSA", "TV", "TRAV", "ALAMEDA", "AL",
    "PRACA", "PCA", "PC", "RODOVIA", "ROD", "ESTRADA", "EST", "LARGO", "LGO",
    "VIA", "VIADUTO", "PASSAGEM", "LINHA", "JARDINETE", "BOSQUE", "PARQUE",
    "MARGINAL", "VIELA", "SERVIDAO", "CONTORNO", "TREVO", "ACESSO",
}
ABREV = {
    "MAL": "MARECHAL", "DR": "DOUTOR", "DRA": "DOUTORA", "PROF": "PROFESSOR",
    "PROFA": "PROFESSORA", "GAL": "GENERAL", "GEN": "GENERAL", "CEL": "CORONEL",
    "STA": "SANTA", "STO": "SANTO", "S": "SAO", "PE": "PADRE", "ENG": "ENGENHEIRO",
    "DES": "DESEMBARGADOR", "CAP": "CAPITAO", "TEN": "TENENTE", "SGT": "SARGENTO",
    "PRES": "PRESIDENTE", "VER": "VEREADOR", "DEP": "DEPUTADO", "SEN": "SENADOR",
    "MONS": "MONSENHOR", "BRIG": "BRIGADEIRO", "ALM": "ALMIRANTE", "COM": "COMENDADOR",
    "CONS": "CONSELHEIRO", "GOV": "GOVERNADOR", "MAJ": "MAJOR", "IR": "IRMA",
    "FREI": "FREI", "N": "NOSSA", "SRA": "SENHORA", "VISC": "VISCONDE",
    "BAR": "BARAO", "MIN": "MINISTRO", "ARQ": "ARQUITETO", "JORN": "JORNALISTA",
    "PREF": "PREFEITO", "ENGS": "ENGENHEIROS",
}
SUFIXOS = {"LD", "LE", "EC"}  # lado direito/esquerdo, eixo central da mesma via
LIGACAO = {"DE", "DA", "DO", "DOS", "DAS", "E", "D"}

# bairro escrito diferente no Sigesguarda e no cadastro
ALIAS_BAIRRO = {"CIDADE INDUSTRIAL": "CIDADE INDUSTRIAL DE CURITIBA"}

MOTIVOS = ["casou", "casou_divisa", "rua_vista_em_outro_bairro",
           "rua_sem_ortofoto", "nome_fora_do_cadastro"]


def normalizar(nome):
    n = (nome or "").replace("º", " ").replace("ª", " ")  # 1º DE MAIO -> 1 DE MAIO
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode("ascii")
    n = re.sub(r"[^A-Z0-9 ]", " ", n.upper())
    palavras = n.split()
    while len(palavras) > 1 and palavras[0] in TIPOS:
        palavras = palavras[1:]
    if len(palavras) == 1:  # RUA A, RUA E, RUA S: a letra é o nome, não abreviação
        return palavras[0]
    while palavras and palavras[-1] in SUFIXOS:
        palavras = palavras[:-1]
    palavras = [ABREV.get(p, p) for p in palavras]
    return " ".join(p for p in palavras if p not in LIGACAO)


def normalizar_bairro(nome):
    n = unicodedata.normalize("NFKD", nome or "").encode("ascii", "ignore").decode("ascii")
    n = " ".join(n.upper().split())
    return ALIAS_BAIRRO.get(n, n)


def ruas_do_patch(row):
    if row["ruas"]:
        return {r.rsplit(":", 1)[0] for r in row["ruas"].split("|") if r}
    if row["rua_quadra"]:
        return {row["rua_quadra"]}
    return set()


def carregar_eixos():
    """rua normalizada -> lista de (lat, lon, bairro_e, bairro_d), um por vértice."""
    pts = defaultdict(list)
    for ft in json.load(open(EIXOS, encoding="utf-8"))["features"]:
        p, g = ft["properties"], ft["geometry"]
        if not g:
            continue
        k = normalizar(p["gtm_nm_logradouro"])
        be = normalizar_bairro(p["geo_nm_bairro_e"])
        bd = normalizar_bairro(p["geo_nm_bairro_d"])
        linhas = g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]
        for linha in linhas:
            for lon, lat, *_ in linha:
                pts[k].append((lat, lon, be, bd))
    return pts


def bairro_mais_proximo(pts, lat, lon):
    coslat = math.cos(math.radians(lat))
    return min(pts, key=lambda t: (t[0] - lat) ** 2 + ((t[1] - lon) * coslat) ** 2)[2]


def tile_xy(lat, lon, z=ZOOM_PATCH):
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return x, y


def metros(lat1, lon1, lat2, lon2):
    dy = (lat2 - lat1) * 111_320
    dx = (lon2 - lon1) * 111_320 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(dx, dy)


def comprimento_coberto(patches_xy):
    """(rua, bairro) -> (metros cobertos, hierarquia de maior comprimento)."""
    comp = defaultdict(Counter)  # chave -> {hierarquia: metros}
    for ft in json.load(open(EIXOS, encoding="utf-8"))["features"]:
        p, g = ft["properties"], ft["geometry"]
        if not g:
            continue
        chave = (normalizar(p["gtm_nm_logradouro"]), normalizar_bairro(p["geo_nm_bairro_e"]))
        hier = p["hierarquia_viaria"] or ""
        linhas = g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]
        for linha in linhas:
            for (lon1, lat1, *_), (lon2, lat2, *_) in zip(linha, linha[1:]):
                if tile_xy((lat1 + lat2) / 2, (lon1 + lon2) / 2) in patches_xy:
                    comp[chave][hier] += metros(lat1, lon1, lat2, lon2)
    return {k: (sum(c.values()), c.most_common(1)[0][0]) for k, c in comp.items()}


def contar_cortes():
    """{sufixo: {(x, y, classe): n}} com conf >= cada corte de CORTES (mesma regra de
    contar_objetos_por_patch.py, sobre o lote filtrado por tamanho)."""
    corte = {s: Counter() for s in CORTES}
    with open(DETECCOES, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            cf = float(r["confianca"])
            for s, minimo in CORTES.items():
                if cf >= minimo:
                    corte[s][(r["tile_x"], r["tile_y"], r["classe"])] += 1
    return corte


def sufixo(ini, fim):
    return "" if (ini, fim) == (ANO_INI, ANO_FIM) else f"_{ini}_{fim}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anos", nargs=2, type=int, default=[ANO_INI, ANO_FIM],
                    metavar=("INI", "FIM"))
    args = ap.parse_args()
    ano_ini, ano_fim = args.anos
    saida = Path(str(SAIDA).format(sufixo(ano_ini, ano_fim)))
    saida_sem = Path(str(SAIDA_SEM).format(sufixo(ano_ini, ano_fim)))
    eixos = carregar_eixos()

    # rua de divisa: (rua, bairro de um lado) -> bairros do outro lado
    divisa = defaultdict(set)
    for k, pts in eixos.items():
        for _, _, be, bd in pts:
            if be and bd and be != bd:
                divisa[(k, bd)].add(be)
                divisa[(k, be)].add(bd)

    # --- características por (rua, bairro) -----------------------------------
    feat = defaultdict(lambda: {"n_patches": 0, "renda_soma": 0.0, "renda_n": 0,
                                "lat_soma": 0.0, "lon_soma": 0.0,
                                **{c: 0 for c in CLASSES + COLS_BIN + COLS_CORTE + COLS_TIPO}})
    corte = contar_cortes()
    renda = {}
    with open(RENDA, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            if row["renda_media"]:
                renda[(row["x"], row["y"])] = float(row["renda_media"])
    patches_xy = set()
    sem_rua = sem_eixo = 0
    with open(PATCHES, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            patches_xy.add((int(row["x"]), int(row["y"])))
            ruas = ruas_do_patch(row)
            if not ruas:
                sem_rua += 1
                continue
            lat, lon = float(row["lat_centro"]), float(row["lon_centro"])
            for rua in ruas:
                k = normalizar(rua)
                if k not in eixos:
                    sem_eixo += 1
                    continue
                chave = (k, bairro_mais_proximo(eixos[k], lat, lon))
                d = feat[chave]
                d["n_patches"] += 1
                d["lat_soma"] += lat
                d["lon_soma"] += lon
                for c in CLASSES:
                    d[c] += int(row[c] or 0)
                for c in BINARIAS:
                    d[c + "_bin"] += int(row[c] or 0) > 0
                tem_area = int(row["area_NaoEdificada"] or 0) > 0
                tem_onibus = int(row["ponto_onibus"] or 0) > 0
                if tem_area or tem_onibus:
                    d[COLS_TIPO[2] if tem_area and tem_onibus
                      else COLS_TIPO[0] if tem_area else COLS_TIPO[1]] += 1
                for s in CORTES:
                    for c in CLASSES:
                        n = corte[s].get((row["x"], row["y"], c), 0)
                        d[c + s] += n
                        if c in BINARIAS:
                            d[c + "_bin" + s] += n > 0
                r = renda.get((row["x"], row["y"]))
                if r is not None:
                    d["renda_soma"] += r
                    d["renda_n"] += 1
    ruas_vistas = {k for k, _ in feat}
    print(f"características: {len(feat)} pares (rua, bairro), {len(ruas_vistas)} ruas "
          f"distintas; {sem_rua} patches sem rua (fora), {sem_eixo} rua-patch sem eixo")
    comp = comprimento_coberto(patches_xy)
    sem_comp = sum(1 for k in feat if k not in comp)
    print(f"comprimento coberto: {len(feat) - sem_comp} pares com eixo dentro das ortofotos, "
          f"{sem_comp} sem")

    # --- ocorrências ---------------------------------------------------------
    ocorr = defaultdict(Counter)
    total_nat = Counter()
    sem_casar = Counter()
    motivos = Counter()
    n_total = 0
    with open(SIGESGUARDA, encoding="latin-1", newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            data = row["OCORRENCIA_DATA"]
            if not data[:4].isdigit() or not (ano_ini <= int(data[:4]) <= ano_fim):
                continue
            n_total += 1
            nat = row["NATUREZA1_DESCRICAO"].strip()
            total_nat[nat] += 1
            k = normalizar(row["LOGRADOURO_NOME"])
            b = normalizar_bairro(row["ATENDIMENTO_BAIRRO_NOME"])
            chave = (k, b)
            if chave in feat:
                motivo = "casou"
            else:
                outro = [bb for bb in divisa.get(chave, ()) if (k, bb) in feat]
                if len(outro) == 1:
                    chave, motivo = (k, outro[0]), "casou_divisa"
                elif k not in eixos:
                    motivo = "nome_fora_do_cadastro"
                elif k in ruas_vistas:
                    motivo = "rua_vista_em_outro_bairro"
                else:
                    motivo = "rua_sem_ortofoto"
            motivos[motivo] += 1
            if motivo.startswith("casou"):
                ocorr[chave][nat] += 1
            else:
                sem_casar[(row["LOGRADOURO_NOME"].strip(), b, motivo)] += 1
    print(f"ocorrências {ano_ini}-{ano_fim}: {n_total}")
    for m in MOTIVOS:
        print(f"  {m:27s} {motivos[m]:7d} ({motivos[m] / n_total:.1%})")

    # --- saída ---------------------------------------------------------------
    faltando = [n for n in NATUREZAS if n not in total_nat]
    if faltando:
        raise SystemExit(f"natureza não encontrada no Sigesguarda: {faltando}")
    naturezas = NATUREZAS + [n for n, v in total_nat.most_common()
                             if v >= MIN_NATUREZA and n not in NATUREZAS]
    col_nat = {n: "oc_" + re.sub(r"[^a-z0-9]+", "_", normalizar(n).lower()).strip("_")
               for n in naturezas}
    with open(saida, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["rua_chave", "bairro", "n_patches", "comprimento_m", "hierarquia",
                    "renda_media", *CLASSES, *COLS_BIN, *col_nat.values(), "oc_total",
                    "lat_media", "lon_media", *COLS_CORTE, *COLS_TIPO])
        for chave in sorted(feat):
            d, oc = feat[chave], ocorr.get(chave, Counter())
            m, h = comp.get(chave, (0.0, ""))
            r = f"{d['renda_soma'] / d['renda_n']:.2f}" if d["renda_n"] else ""
            w.writerow([chave[0], chave[1], d["n_patches"], f"{m:.1f}", h, r,
                        *(d[c] for c in CLASSES + COLS_BIN), *(oc[n] for n in naturezas),
                        sum(oc.values()), f"{d['lat_soma'] / d['n_patches']:.6f}",
                        f"{d['lon_soma'] / d['n_patches']:.6f}",
                        *(d[c] for c in COLS_CORTE + COLS_TIPO)])
    with open(saida_sem, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["logradouro_sigesguarda", "bairro", "motivo", "ocorrencias"])
        for (nome, b, motivo), v in sem_casar.most_common():
            w.writerow([nome, b, motivo, v])

    com_oc = sum(1 for k in feat if any(ocorr.get(k, Counter())[n] for n in naturezas))
    print(f"tabela: {len(feat)} pares ({com_oc} com >=1 ocorrência das {len(naturezas)} "
          f"naturezas, {len(feat) - com_oc} com 0)")
    print(f"-> {saida}\n-> {saida_sem}")
    print("\nmais frequentes sem casamento:")
    for (nome, b, motivo), v in sem_casar.most_common(20):
        print(f"  {v:6d}  {nome} / {b}  [{motivo}]")


if __name__ == "__main__":
    main()
