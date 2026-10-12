#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baixar_ortofotos_ippuc.py
=========================
Baixador de ortofotos do IPPUC (GeoCuritiba) para o TCC "Quadra a Quadra".

Estágio 1 do pipeline: baixa o cache de tiles de um acervo (por padrão
Ortofotos2019) para um bounding box e nível de zoom, gravando em disco como
    <saida>/<Acervo>/<z>/<x>/<y>.jpg
com resume (pula o que já existe), retry com backoff e um manifesto JSON.

Os tiles são um cache XYZ Web Mercator (EPSG:3857), JPEG 256x256, sem
autenticação. ATENÇÃO: a URL do ArcGIS é /tile/{z}/{y}/{x} — row (y) ANTES
de col (x). O esquema é o XYZ padrão (verificado), então a matemática
slippy-map padrão vale para converter lat/lon <-> tile.

Requisitos: só a biblioteca padrão do Python (>=3.8). A costura opcional
(--stitch) usa Pillow (pip install Pillow), mas o download em si não precisa
de nada além do Python.

Exemplos
--------
# 1) Ver o que seria baixado, sem tocar na rede (recomendado 1o):
python baixar_ortofotos_ippuc.py --regiao centro --zoom 21 --dry-run
python baixar_ortofotos_ippuc.py --regiao sitio_cercado --zoom 21 --dry-run

# 2) Baixar o piloto de fato e costurar um mosaico pra olhar:
python baixar_ortofotos_ippuc.py --regiao centro --zoom 21 --stitch
python baixar_ortofotos_ippuc.py --regiao sitio_cercado --zoom 21 --stitch

# 3) Cidade inteira (depois de validar o piloto) — leva horas:
python baixar_ortofotos_ippuc.py --regiao curitiba --zoom 21

# 4) Um bbox arbitrário (lat/lon em graus decimais):
python baixar_ortofotos_ippuc.py --bbox -25.4315 -49.2770 -25.4255 -49.2695 --zoom 21

# Outra época (o esquema de tiles é co-registrado entre anos):
python baixar_ortofotos_ippuc.py --acervo Ortofotos2012 --regiao centro --zoom 21
"""

import argparse
import concurrent.futures
import json
import math
import os
import ssl
import sys
import threading
import time
import urllib.request
import urllib.error

# --------------------------------------------------------------------------
# Configuração de serviço
# --------------------------------------------------------------------------
HOST = "https://geocuritiba.ippuc.org.br"
# Padrão dos acervos hospedados de imagem no ArcGIS Server do IPPUC:
#   .../server/rest/services/Hosted/<Acervo>/MapServer
SERVICE_TMPL = HOST + "/server/rest/services/Hosted/{acervo}/MapServer"

# Regiões-piloto e cidade (lat_sul, lon_oeste, lat_norte, lon_leste) em WGS84.
# As caixas do piloto têm ~600-800 m de lado: rápidas de baixar e suficientes
# para validar resolução e qualidade. A caixa "curitiba" cobre o município
# com folga (a ortofoto vai um pouco além dos limites administrativos).
REGIOES = {
    # Centro / Praça Tiradentes (~-25.4284, -49.2733)
    "centro":        (-25.4320, -49.2775, -25.4250, -49.2690),
    # Sítio Cercado (bairro do sul, ~-25.5490, -49.2790)
    "sitio_cercado": (-25.5525, -49.2830, -25.5455, -49.2750),
    # Município de Curitiba (bbox generoso)
    "curitiba":      (-25.6510, -49.3960, -25.3400, -49.1850),

    # ------------------------------------------------------------------
    # Núcleos urbanos das regionais administrativas (bbox APROXIMADO do
    # miolo urbano, cortando as franjas rurais). NÃO são os polígonos
    # oficiais do IPPUC — são retângulos ancorados em pontos conhecidos
    # da cidade, para reduzir o volume focando na malha urbana. Ajuste à
    # vontade. Onde as caixas se encostam, o resume evita baixar em dobro.
    # ------------------------------------------------------------------
    "regional_matriz":           (-25.4550, -49.3000, -25.4050, -49.2550),
    "regional_boa_vista":        (-25.4150, -49.2550, -25.3550, -49.2050),
    "regional_santa_felicidade": (-25.4350, -49.3400, -25.3850, -49.2900),
    "regional_portao":           (-25.5100, -49.3100, -25.4700, -49.2700),
    "regional_boqueirao":        (-25.5450, -49.2750, -25.4950, -49.2300),
    "regional_cajuru":           (-25.5000, -49.2450, -25.4500, -49.1950),
    "regional_cic":              (-25.5400, -49.3600, -25.4700, -49.3100),
    "regional_pinheirinho":      (-25.5650, -49.3200, -25.5200, -49.2750),
}

# Constantes Web Mercator (EPSG:3857) — origem do canto superior-esquerdo.
ORIGIN = 20037508.342787            # metros; -ORIGIN..+ORIGIN
R_MAJOR = 6378137.0                 # raio da esfera auxiliar

USER_AGENT = "TCC-QuadraAQuadra/1.0 (uso academico; contato: fran.hauch@gmail.com)"

# --------------------------------------------------------------------------
# Matemática de tiles (slippy-map padrão XYZ)
# --------------------------------------------------------------------------
def deg2tile(lat, lon, z):
    """lat/lon (graus) -> (x, y) do tile no zoom z (origem canto sup-esq)."""
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    x = min(max(x, 0), n - 1)
    y = min(max(y, 0), n - 1)
    return x, y


def tile2deg(x, y, z):
    """(x, y) do tile -> lat/lon (graus) do canto superior-esquerdo do tile."""
    n = 2 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


def tile_range(bbox, z):
    """bbox=(lat_s, lon_o, lat_n, lon_l) -> (x0, y0, x1, y1) inclusivo."""
    lat_s, lon_o, lat_n, lon_l = bbox
    x0, y0 = deg2tile(lat_n, lon_o, z)   # noroeste -> menor x, menor y
    x1, y1 = deg2tile(lat_s, lon_l, z)   # sudeste  -> maior x, maior y
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def ground_resolution(lat, z):
    """Metros por pixel no zoom z, na latitude lat."""
    return math.cos(math.radians(lat)) * 2 * math.pi * R_MAJOR / (256 * 2 ** z)


def tile_bounds_3857(x, y, z):
    """Retângulo do tile em EPSG:3857 (metros): (minx, miny, maxx, maxy)."""
    res = 2 * ORIGIN / (2 ** z)
    minx = -ORIGIN + x * res
    maxx = -ORIGIN + (x + 1) * res
    maxy = ORIGIN - y * res
    miny = ORIGIN - (y + 1) * res
    return minx, miny, maxx, maxy


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
def _ssl_ctx(insecure):
    ctx = ssl.create_default_context()
    if insecure:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


def fetch_service_meta(acervo, insecure, timeout=30):
    """Lê ?f=json do MapServer: LODs, formato, extensão. Retorna dict ou None."""
    url = SERVICE_TMPL.format(acervo=acervo) + "?f=json"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, context=_ssl_ctx(insecure), timeout=timeout) as r:
            return json.load(r)
    except Exception as e:
        print(f"[meta] aviso: não consegui ler metadados do serviço ({e}). "
              f"Sigo com o zoom pedido.", file=sys.stderr)
        return None


def tile_url(acervo, z, x, y):
    # ORDEM ArcGIS: /tile/{level}/{row=y}/{col=x}
    return SERVICE_TMPL.format(acervo=acervo) + f"/tile/{z}/{y}/{x}"


def download_one(acervo, z, x, y, dest, ctx, retries, sleep):
    """Baixa um tile. Retorna ('ok'|'skip'|'miss'|'err', bytes)."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return "skip", 0
    url = tile_url(acervo, z, x, y)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=30) as r:
                ctype = r.headers.get("Content-Type", "")
                body = r.read()
                if "image" not in ctype or len(body) < 128:
                    # Tile ausente do cache costuma vir como erro JSON / vazio.
                    return "miss", 0
                tmp = dest + ".part"
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(tmp, "wb") as f:
                    f.write(body)
                os.replace(tmp, dest)
                if sleep:
                    time.sleep(sleep)
                return "ok", len(body)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return "miss", 0          # fora do cache: não adianta tentar de novo
            last = e
        except Exception as e:
            last = e
        # backoff exponencial: 0.5s, 1s, 2s, ...
        time.sleep(0.5 * (2 ** attempt))
    print(f"[err] {z}/{y}/{x}: {last}", file=sys.stderr)
    return "err", 0


# --------------------------------------------------------------------------
# Costura opcional (mosaico para visualização)
# --------------------------------------------------------------------------
PRJ_3857 = ('PROJCS["WGS_1984_Web_Mercator_Auxiliary_Sphere",'
            'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",'
            'SPHEROID["WGS_1984",6378137.0,298.257223563]],'
            'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],'
            'PROJECTION["Mercator_Auxiliary_Sphere"],'
            'PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],'
            'PARAMETER["Central_Meridian",0.0],'
            'PARAMETER["Standard_Parallel_1",0.0],'
            'PARAMETER["Auxiliary_Sphere_Type",0.0],'
            'UNIT["Meter",1.0]]')


def _write_worldfile(png_path, minx_m, maxy_m, res_m):
    """Grava .pgw (world file EPSG:3857) + .prj ao lado do PNG."""
    stem = os.path.splitext(png_path)[0]
    with open(stem + ".pgw", "w") as f:
        f.write(f"{res_m:.10f}\n0.0\n0.0\n{-res_m:.10f}\n{minx_m:.6f}\n{maxy_m:.6f}\n")
    with open(stem + ".prj", "w") as f:
        f.write(PRJ_3857)


def stitch_region(acervo, z, rng, cache_dir, out_png, grade=1):
    """Monta os tiles do range num PNG georreferenciado. Se grade>1, também
    corta o mosaico numa grade grade×grade (ex.: grade=2 -> 4 partes), cada
    pedaço com seu próprio world file."""
    try:
        from PIL import Image
    except ImportError:
        print("[stitch] Pillow não instalado — pulei a costura. "
              "Rode 'pip install Pillow' e repita com --stitch.", file=sys.stderr)
        return None
    x0, y0, x1, y1 = rng
    cols, rows = (x1 - x0 + 1), (y1 - y0 + 1)
    W, H = cols * 256, rows * 256
    if W * H > 400_000_000:  # ~400 MP: evita estourar memória por acidente
        print(f"[stitch] mosaico muito grande ({W}x{H}px) — pulei. "
              f"Costure sub-regiões ou use um SIG.", file=sys.stderr)
        return None
    mosaic = Image.new("RGB", (W, H), (0, 0, 0))
    faltando = 0
    for xi, x in enumerate(range(x0, x1 + 1)):
        for yi, y in enumerate(range(y0, y1 + 1)):
            p = os.path.join(cache_dir, str(z), str(x), f"{y}.jpg")
            if os.path.exists(p) and os.path.getsize(p) > 0:
                try:
                    mosaic.paste(Image.open(p), (xi * 256, yi * 256))
                except Exception:
                    faltando += 1
            else:
                faltando += 1

    res_m = 2 * ORIGIN / (2 ** z) / 256.0          # metros por pixel
    minx, _, _, maxy = tile_bounds_3857(x0, y0, z)  # canto sup-esq do mosaico

    if grade <= 1:
        mosaic.save(out_png)
        _write_worldfile(out_png, minx, maxy, res_m)
        print(f"[stitch] {out_png}  ({W}x{H}px, {faltando} tiles faltando)")
        return out_png

    # --- grade grade×grade -----------------------------------------------
    stem = os.path.splitext(out_png)[0]
    cw, ch = W // grade, H // grade   # tamanho de cada célula em pixels
    saidas = []
    for r in range(grade):            # linha (norte->sul)
        for c in range(grade):        # coluna (oeste->leste)
            left, top = c * cw, r * ch
            # última coluna/linha absorve o resto da divisão inteira
            right = W if c == grade - 1 else (c + 1) * cw
            bottom = H if r == grade - 1 else (r + 1) * ch
            parte = mosaic.crop((left, top, right, bottom))
            pnome = f"{stem}_r{r+1}c{c+1}.png"
            parte.save(pnome)
            _write_worldfile(pnome, minx + left * res_m, maxy - top * res_m, res_m)
            saidas.append(pnome)
    print(f"[stitch] {len(saidas)} partes {cw}x{ch}px (grade {grade}x{grade}), "
          f"{faltando} tiles faltando:")
    for s in saidas:
        print("   " + s)
    return saidas


# --------------------------------------------------------------------------
# Principal
# --------------------------------------------------------------------------
def human_bytes(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.1f} {u}"
        n /= 1024


def main():
    ap = argparse.ArgumentParser(description="Baixador de ortofotos do IPPUC (GeoCuritiba).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--regiao", choices=sorted(REGIOES), help="Região pré-definida.")
    g.add_argument("--bbox", nargs=4, type=float, metavar=("LAT_S", "LON_O", "LAT_N", "LON_L"),
                   help="Bounding box em graus decimais WGS84.")
    ap.add_argument("--acervo", default="Ortofotos2019", help="Acervo (padrão: Ortofotos2019).")
    ap.add_argument("--zoom", type=int, default=20,
                    help="Nível de zoom (padrão: 20 — ~0,135 m/px, o padrão do TCC).")
    ap.add_argument("--saida", default="dataset", help="Diretório de saída (padrão: ./dataset).")
    ap.add_argument("--threads", type=int, default=16,
                    help="Threads paralelas (padrão: 16 — medido pela sonda_vazao).")
    ap.add_argument("--retries", type=int, default=3, help="Tentativas por tile (padrão: 3).")
    ap.add_argument("--sleep", type=float, default=0.0, help="Pausa por tile por thread (s).")
    ap.add_argument("--insecure", action="store_true",
                    help="Ignora verificação de certificado TLS (use se der erro de SSL).")
    ap.add_argument("--stitch", action="store_true", help="Costura um mosaico PNG ao final.")
    ap.add_argument("--grade", type=int, default=1, metavar="N",
                    help="Corta o mosaico numa grade N×N (ex.: 2 = 4 partes). Requer --stitch.")
    ap.add_argument("--dry-run", action="store_true", help="Só calcula; não baixa nada.")
    args = ap.parse_args()

    bbox = REGIOES[args.regiao] if args.regiao else tuple(args.bbox)
    nome = args.regiao or "bbox"
    z = args.zoom

    # Metadados do serviço (valida zoom, formato) — best effort.
    meta = None if args.dry_run else fetch_service_meta(args.acervo, args.insecure)
    if meta:
        ti = meta.get("tileInfo", {})
        lods = [l.get("level") for l in ti.get("lods", [])]
        if lods:
            print(f"[meta] {meta.get('name', args.acervo)} — LODs {min(lods)}..{max(lods)}, "
                  f"formato {ti.get('format')}, {ti.get('rows')}x{ti.get('cols')}px/tile")
            if z > max(lods):
                print(f"[meta] AVISO: zoom {z} acima do máximo {max(lods)} do serviço.",
                      file=sys.stderr)

    rng = tile_range(bbox, z)
    x0, y0, x1, y1 = rng
    cols, rows = (x1 - x0 + 1), (y1 - y0 + 1)
    total = cols * rows
    lat_c = (bbox[0] + bbox[2]) / 2
    res = ground_resolution(lat_c, z)
    est_bytes = total * 11800          # ~11,8 KB/tile medido
    est_seg = total / 38.0             # ~38 tiles/s medido (8 threads)

    print("=" * 64)
    print(f"Acervo : {args.acervo}")
    print(f"Região : {nome}  bbox={bbox}")
    print(f"Zoom   : {z}   (~{res:.3f} m/px em lat {lat_c:.3f})")
    print(f"Tiles  : x {x0}..{x1} ({cols}) × y {y0}..{y1} ({rows}) = {total:,}".replace(",", "."))
    print(f"Estimado: ~{human_bytes(est_bytes)}  ·  ~{est_seg/60:.1f} min a 38 tiles/s")
    print(f"Saída  : {os.path.abspath(os.path.join(args.saida, args.acervo))}")
    print("=" * 64)

    if args.dry_run:
        print("[dry-run] nada foi baixado.")
        return

    cache_dir = os.path.join(args.saida, args.acervo)
    os.makedirs(cache_dir, exist_ok=True)
    ctx = _ssl_ctx(args.insecure)

    counters = {"ok": 0, "skip": 0, "miss": 0, "err": 0, "bytes": 0}
    lock = threading.Lock()
    t0 = time.time()
    done = 0

    def task(x, y):
        dest = os.path.join(cache_dir, str(z), str(x), f"{y}.jpg")
        status, nb = download_one(args.acervo, z, x, y, dest, ctx, args.retries, args.sleep)
        return status, nb

    jobs = [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = [ex.submit(task, x, y) for (x, y) in jobs]
        for fut in concurrent.futures.as_completed(futs):
            status, nb = fut.result()
            with lock:
                counters[status] += 1
                counters["bytes"] += nb
                done += 1
                if done % 200 == 0 or done == total:
                    el = time.time() - t0
                    rate = done / el if el else 0
                    eta = (total - done) / rate if rate else 0
                    print(f"  {done:,}/{total:,}  ".replace(",", ".") +
                          f"ok={counters['ok']} skip={counters['skip']} "
                          f"miss={counters['miss']} err={counters['err']}  "
                          f"{rate:.0f} t/s  ETA {eta/60:.1f} min", flush=True)

    dt = time.time() - t0
    print("-" * 64)
    print(f"Concluído em {dt/60:.1f} min · baixados {human_bytes(counters['bytes'])} · "
          f"ok={counters['ok']} skip={counters['skip']} miss={counters['miss']} err={counters['err']}")

    # Manifesto (reprodutibilidade / retomada)
    manifest = {
        "acervo": args.acervo, "regiao": nome, "bbox_wgs84": list(bbox),
        "zoom": z, "tile_range": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        "tiles_total": total, "resolucao_m_px": round(res, 4),
        "contadores": counters, "segundos": round(dt, 1),
        "gerado_em": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "servico": SERVICE_TMPL.format(acervo=args.acervo),
    }
    os.makedirs(args.saida, exist_ok=True)
    mpath = os.path.join(args.saida, f"manifesto_{args.acervo}_{nome}_z{z}.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Manifesto: {mpath}")

    if counters["err"]:
        print(f"[aviso] {counters['err']} tiles falharam. Rode o mesmo comando de novo "
              f"para retomar (o resume pula o que já baixou).", file=sys.stderr)

    if args.stitch:
        out_png = os.path.join(args.saida, f"mosaico_{args.acervo}_{nome}_z{z}.png")
        stitch_region(args.acervo, z, rng, cache_dir, out_png, grade=max(1, args.grade))


if __name__ == "__main__":
    main()
