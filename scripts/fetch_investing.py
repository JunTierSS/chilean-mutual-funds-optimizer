#!/usr/bin/env python3
"""
scripts/fetch_investing.py
--------------------------
Descarga datos históricos de fondos mutuos chilenos desde Investing.com.
Usa cloudscraper para bypassear la protección anti-bot de Cloudflare.

Flujo automático completo:
    python3 scripts/fetch_investing.py --interval both
    → descarga CSVs → git commit → git push → Streamlit Cloud redeploya

Uso:
    python3 scripts/fetch_investing.py                          # mensual, todos
    python3 scripts/fetch_investing.py --interval daily         # diario, todos
    python3 scripts/fetch_investing.py --interval both          # ambos (recomendado)
    python3 scripts/fetch_investing.py --broker santander       # solo Santander
    python3 scripts/fetch_investing.py --dry-run                # sin escribir ni pushear
    python3 scripts/fetch_investing.py --no-push                # descarga pero no pushea

Brokers: santander, larrain_vial, bci, bice, bancochile, security

Author: Junwei He — MSc Data Science (c), Universidad de Chile
"""

import argparse
import re
import subprocess
import sys
import time
import random
from datetime import datetime, date
from io import StringIO
from pathlib import Path

import pandas as pd

try:
    import cloudscraper
except ImportError:
    print("❌ Instala dependencias: pip3 install cloudscraper")
    sys.exit(1)

# ── Configuración ─────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent.parent / "data"
DATE_FROM = "01/01/2020"
SLEEP_MIN = 1.5
SLEEP_MAX = 3.5

# ── Catálogo completo (slugs verificados desde Investing.com Chile) ────────────
ALL_FONDOS: dict = {
    "santander": {
        "0P0000KBXJ": {"nombre": "Santander Renta Mediano Plazo APV",         "slug": "santander-renta-mediano-plazo-apv"},
        "0P0000KBZ5": {"nombre": "Santander Renta Selecta Chile APV",          "slug": "santander-renta-selecta-chile-apv"},
        "0P0000KBZ6": {"nombre": "Santander Renta Selecta Chile GLOBA",        "slug": "santander-renta-selecta-chile-global"},
        "0P0000KBZQ": {"nombre": "Santander Acciones Global Emergente APV",    "slug": "santander-acciones-global-emergente-apv"},
        "0P0000KBZR": {"nombre": "Santander Acciones Global Emergente EJECU",  "slug": "santander-acciones-global-emergente-ejecutivo"},
        "0P0000KBZS": {"nombre": "Santander Acciones Global Emergente INVER",  "slug": "santander-acciones-global-emergente-inversionista"},
        "0P0000KC0A": {"nombre": "Santander GO Acciones Globales ESG APV",     "slug": "santander-go-acciones-globales-esg-apv"},
        "0P0000KC0E": {"nombre": "Santander Renta Largo Plazo UF APV",         "slug": "santander-renta-largo-plazo-uf-apv"},
        "0P0000KC1P": {"nombre": "Santander GO Acciones Asia Emergente APV",   "slug": "santander-go-acciones-asia-emergente-apv"},
        "0P0000KC1Q": {"nombre": "Santander GO Acciones Asia Emergente EJECU", "slug": "santander-go-acciones-asia-emergente-ejecutivo"},
        "0P0000KC2J": {"nombre": "Santander B APV",                            "slug": "santander-b-apv"},
        "0P0000KC2T": {"nombre": "Santander Bonos Nacionales APV",             "slug": "santander-bonos-nacionales-apv"},
        "0P0000KC3F": {"nombre": "Santander Acciones Chilenas APV",            "slug": "santander-acciones-chilenas-apv"},
        "0P0000KC3M": {"nombre": "Santander Acciones Selectas Chile APV",      "slug": "santander-acciones-selectas-chile-apv"},
        "0P0000KC3N": {"nombre": "Santander Acciones Selectas Chile PATRI",    "slug": "santander-acciones-selectas-chile-patrimonial"},
        "0P0000KNOT": {"nombre": "Santander Renta Extra Largo Plazo UF APV",   "slug": "santander-renta-extra-largo-plazo-uf-apv"},
        "0P0000N9A0": {"nombre": "Santander Renta Extra Largo Plazo UF INVER", "slug": "santander-renta-extra-largo-plazo-uf-inversionista"},
        "0P0000V0AL": {"nombre": "Santander Private Banking Agresivo APV",     "slug": "santander-private-banking-agresivo-apv"},
        "0P0000V0AN": {"nombre": "Santander Private Banking Equilibrio APV",   "slug": "santander-private-banking-equilibrio-apv"},
        "0P000102JP": {"nombre": "Santander Renta Selecta Chile PATRI",        "slug": "santander-renta-selecta-chile-patrimonial"},
    },
    "larrain_vial": {
        "0P0000K9XA": {"nombre": "LV Cuenta Activa Defensiva Dolar P",   "slug": "larrainvial-cuenta-activa-defensiva-dolar-p"},
        "0P0000K9XC": {"nombre": "LV Cuenta Activa Defensiva Dolar A",   "slug": "larrainvial-cuenta-activa-defensiva-dolar-a"},
        "0P0000K9XE": {"nombre": "LV Cuenta Activa Defensiva Dolar APV", "slug": "larrainvial-cuenta-activa-defensiva-dolar-apv"},
        "0P0000KCCW": {"nombre": "LV Portfolio Lider F",                 "slug": "larrainvial-portfolio-lider-f"},
        "0P0000KCD2": {"nombre": "LV Ahorro Corporativo APV",            "slug": "larrainvial-ahorro-corporativo-apv"},
        "0P0000KCDG": {"nombre": "LV Latinoamericano APV",               "slug": "larrainvial-latinoamericano-apv"},
        "0P0000KCDI": {"nombre": "LV Latinoamericano F",                 "slug": "larrainvial-latinoamericano-f"},
        "0P0000KCE0": {"nombre": "LV Bonos Latam A",                     "slug": "larrainvial-bonos-latam-a"},
        "0P0000KCE1": {"nombre": "LV Bonos Latam APV",                   "slug": "larrainvial-bonos-latam-apv"},
        "0P0000KCE3": {"nombre": "LV Asia APV",                          "slug": "larrainvial-asia-apv"},
        "0P0000KCE5": {"nombre": "LV Asia F",                            "slug": "larrainvial-asia-f"},
        "0P0000KCEF": {"nombre": "LV Enfoque A",                         "slug": "larrainvial-enfoque-a"},
        "0P0000KCEK": {"nombre": "LV Enfoque APV",                       "slug": "larrainvial-enfoque-apv"},
        "0P0000KCHS": {"nombre": "LV Acciones Nacionales APV",           "slug": "larrainvial-acciones-nacionales-apv"},
        "0P0000KCIT": {"nombre": "LV Ahorro Capital APV",                "slug": "larrainvial-ahorro-capital-apv"},
        "0P0000TFHA": {"nombre": "LV Acciones Nacionales I",             "slug": "larrainvial-acciones-nacionales-i"},
        "0P0000TFVX": {"nombre": "LV Ahorro Estrategico I",              "slug": "larrainvial-ahorro-estrategico-i"},
        "0P0000TFW3": {"nombre": "LV Ahorro Capital I",                  "slug": "larrainvial-ahorro-capital-i"},
        "0P0000TG5U": {"nombre": "LV Bonos Latam F",                     "slug": "larrainvial-bonos-latam-f"},
        "0P0000TG5W": {"nombre": "LV Bonos Latam P",                     "slug": "larrainvial-bonos-latam-p"},
        "0P0000TG61": {"nombre": "LV Enfoque F",                         "slug": "larrainvial-enfoque-f"},
        "0P0000TG62": {"nombre": "LV Enfoque I",                         "slug": "larrainvial-enfoque-i"},
    },
    "bci": {
        "0P0000KA6F": {"nombre": "BCI Selección Bursátil APV",          "slug": "bci-seleccion-bursatil-apv"},
        "0P0000KA7G": {"nombre": "BCI de Personas APV",                 "slug": "bci-de-personas-apv"},
        "0P0000KA7B": {"nombre": "BCI Gestión Global Dinámica 20 APV",  "slug": "bci-gestion-global-dinamica-20-apv"},
        "0P0000KA71": {"nombre": "BCI Gestión Global Dinámica 50 APV",  "slug": "bci-gestion-global-dinamica-50-apv"},
        "0P0000KA78": {"nombre": "BCI Gestión Global Dinámica 80 APV",  "slug": "bci-gestion-global-dinamica-80-apv"},
    },
    "bice": {
        "0P0000KA40": {"nombre": "BICE Extra G",        "slug": "bice-extra-g"},
        "0P0000N3YH": {"nombre": "BICE Extra D",        "slug": "bice-extra-d"},
        "0P0000ZNUT": {"nombre": "BICE Extra Gestion G","slug": "bice-extra-gestion-g"},
        "0P0000N0W0": {"nombre": "BICE Target D",       "slug": "bice-target-d"},
        "0P0000X2CX": {"nombre": "BICE Target G",       "slug": "bice-target-g"},
    },
    "bancochile": {
        "0P0000Z8US": {"nombre": "BancoEstado Protección I",   "slug": "proteccion-bancoestado-i"},
        "0P0000Z8UW": {"nombre": "BancoEstado Compromiso I",   "slug": "compromiso-bancoestado-i"},
        "0P0000KASM": {"nombre": "BancoEstado Compromiso A",   "slug": "compromiso-bancoestado-a"},
        "0P0000SHS5": {"nombre": "BancoEstado Perfil E PATRI", "slug": "bancoestado-perfil-e-patri"},
    },
    "security": {
        "0P0000KBJZ": {"nombre": "Security Gold B",                   "slug": "security-gold-b"},
        "0P0000KBK0": {"nombre": "Security Gold I-APV",               "slug": "security-gold-i-apv"},
        "0P0000KBV4": {"nombre": "Security Crecimiento Estratégico B","slug": "fondo-mutuo-security-crecimiento-es"},
        "0P0000KBK3": {"nombre": "Security Equilibrio Estratégico B", "slug": "security-equilibrio-estrategico-b"},
        "0P000087W9": {"nombre": "Security Mid Term UF B",            "slug": "security-mid-term-uf-b"},
    },
}

INTERVAL_MAP = {
    "monthly": ("Monthly", "Datos_historicos_", ""),
    "daily":   ("Daily",   "Datos_diarios_",    "daily"),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sleep():
    time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))


def needs_update(path: Path, interval: str) -> bool:
    if not path.exists():
        return True
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    now   = datetime.today()
    return not (mtime.year == now.year and mtime.month == now.month) if interval == "monthly" \
           else mtime.date() < date.today()


def save_csv(df: pd.DataFrame, broker: str, mid: str, prefix: str, subdir: str) -> Path:
    out_dir = BASE_DIR / broker / subdir if subdir else BASE_DIR / broker
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{prefix}{mid}.csv"
    rename = {"Price": "Último", "Precio": "Último", "Close": "Último",
              "Open": "Apertura", "High": "Máximo", "Low": "Mínimo",
              "Change %": "% var.", "Var. %": "% var."}
    df = df.rename(columns=rename)
    for c in df.columns:
        if "fecha" in c.lower() or "date" in c.lower():
            df = df.rename(columns={c: "Fecha"})
            break
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# ── Cloudscraper fetcher ───────────────────────────────────────────────────────

def _make_scraper():
    return cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "darwin", "desktop": True}
    )


def fetch_fund(scraper, mid: str, info: dict, interval_sec: str) -> object:
    """
    1. GET la página histórica para establecer cookies de sesión.
    2. Extrae pairId del HTML.
    3. POST a HistoricalDataAjax con las cookies activas.
    """
    slug     = info.get("slug", "")
    end_date = datetime.today().strftime("%m/%d/%Y")
    base_url = "https://es.investing.com/funds/"

    page_url = f"{base_url}{slug}-historical-data"

    try:
        resp = scraper.get(page_url, timeout=20)
        if resp.status_code != 200:
            return None

        html = resp.text
        match = (re.search(r'data-pair-id=["\'](\d+)["\']', html) or
                 re.search(r'"pairId"\s*:\s*(\d+)', html) or
                 re.search(r'curr_id["\s:=]+(\d+)', html))
        if not match:
            return None

        pair_id = match.group(1)
        _sleep()

        r2 = scraper.post(
            "https://es.investing.com/instruments/HistoricalDataAjax",
            data={
                "curr_id":      pair_id,
                "st_date":      DATE_FROM,
                "end_date":     end_date,
                "interval_sec": interval_sec,
                "action":       "historical_data",
            },
            headers={
                "X-Requested-With": "XMLHttpRequest",
                "Referer":          page_url,
            },
            timeout=20,
        )

        if r2.status_code != 200 or "<table" not in r2.text:
            return None

        tables = pd.read_html(StringIO(r2.text))
        return tables[0] if tables and not tables[0].empty else None

    except Exception as e:
        print(f"    ⚠️  {slug}: {e}")
        return None


# ── Lógica principal ──────────────────────────────────────────────────────────

def update_all(brokers: dict, interval: str, dry_run: bool) -> dict:
    stats = {"ok": 0, "error": 0, "skip": 0}
    iv_sec, prefix, subdir = INTERVAL_MAP[interval]

    scraper = _make_scraper()

    # Warm-up session
    print("  🌐 Estableciendo sesión en Investing.com...")
    try:
        scraper.get("https://es.investing.com/", timeout=15)
        _sleep()
    except Exception as e:
        print(f"  ⚠️  Warm-up fallido: {e}")

    for broker, fondos in brokers.items():
        print(f"\n{'─'*65}")
        print(f"📦  {broker.upper()} — {len(fondos)} fondos — [{interval}]")
        print(f"{'─'*65}")

        for mid, info in fondos.items():
            nombre   = info["nombre"]
            out_dir  = BASE_DIR / broker / subdir if subdir else BASE_DIR / broker
            out_path = out_dir / f"{prefix}{mid}.csv"

            if not needs_update(out_path, interval):
                print(f"  ⏭️  {nombre[:55]:<55}  ya actualizado")
                stats["skip"] += 1
                continue

            print(f"  🔄  {nombre[:55]:<55}", end="  ", flush=True)

            df = fetch_fund(scraper, mid, info, iv_sec)

            if df is None or df.empty:
                print("❌  sin datos")
                stats["error"] += 1
                _sleep()
                continue

            if dry_run:
                print(f"✅  {len(df):>5} filas  (dry-run)")
            else:
                path = save_csv(df, broker, mid, prefix, subdir)
                print(f"✅  {len(df):>5} filas  → {path.parent.name}/{path.name}")

            stats["ok"] += 1
            _sleep()

    return stats


def git_commit_push(dry_run: bool):
    """Commit y push automático si hay cambios en data/."""
    result = subprocess.run(
        ["git", "diff", "--quiet", "data/"],
        cwd=Path(__file__).parent.parent,
        capture_output=True
    )
    # Check also untracked files in data/
    result2 = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "data/"],
        cwd=Path(__file__).parent.parent,
        capture_output=True, text=True
    )
    if result.returncode == 0 and not result2.stdout.strip():
        print("\n  ℹ️  Sin cambios en data/ — nada que commitear")
        return

    fecha = datetime.today().strftime("%Y-%m-%d")
    cmds = [
        ["git", "add", "data/"],
        ["git", "commit", "-m", f"chore: update fund data {fecha}"],
        ["git", "push", "origin", "main"],
    ]
    for cmd in cmds:
        if dry_run:
            print(f"  [dry-run] {' '.join(cmd)}")
            continue
        r = subprocess.run(cmd, cwd=Path(__file__).parent.parent,
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  ❌ {' '.join(cmd)}: {r.stderr.strip()}")
            return
    print("  ✅ Pusheado a GitHub — Streamlit Cloud redesplegando...")


def main():
    parser = argparse.ArgumentParser(
        description="Actualiza fondos mutuos desde Investing.com y pushea a GitHub"
    )
    parser.add_argument("--broker", default="all",
                        choices=["all"] + list(ALL_FONDOS.keys()))
    parser.add_argument("--interval", default="monthly",
                        choices=["monthly", "daily", "both"])
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--no-push",  action="store_true",
                        help="Descarga datos pero no hace git push")
    args = parser.parse_args()

    brokers   = ALL_FONDOS if args.broker == "all" else {args.broker: ALL_FONDOS[args.broker]}
    intervals = ["monthly", "daily"] if args.interval == "both" else [args.interval]

    print("=" * 65)
    print("🇨🇱  Actualizador de Fondos Mutuos — Investing.com")
    print(f"    Fecha     : {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    print(f"    Intervalo : {args.interval}")
    print(f"    Brokers   : {', '.join(brokers.keys())}")
    print(f"    Modo      : {'dry-run' if args.dry_run else 'escritura'}")
    print("=" * 65)

    for interval in intervals:
        total = update_all(brokers, interval, args.dry_run)
        print(f"\n[{interval}] ✅ {total['ok']} ok · ❌ {total['error']} errores · ⏭️  {total['skip']} sin cambios")

    if not args.dry_run and not args.no_push:
        print("\n📤 Commiteando y pusheando a GitHub...")
        git_commit_push(dry_run=False)


if __name__ == "__main__":
    main()
