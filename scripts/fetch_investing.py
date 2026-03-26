#!/usr/bin/env python3
"""
scripts/fetch_investing.py
--------------------------
Descarga datos históricos de fondos mutuos chilenos desde Investing.com.
Soporta frecuencia mensual y diaria para todos los brokers.

Estructura de archivos:
    data/{broker}/Datos_historicos_{id}.csv   ← mensual
    data/{broker}/daily/Datos_diarios_{id}.csv ← diario

Uso:
    python3 scripts/fetch_investing.py                         # mensual, todos
    python3 scripts/fetch_investing.py --interval daily        # diario, todos
    python3 scripts/fetch_investing.py --interval both         # ambos
    python3 scripts/fetch_investing.py --broker bci            # solo BCI
    python3 scripts/fetch_investing.py --dry-run               # sin escribir

Author: Junwei He — MSc Data Science (c), Universidad de Chile
"""

import argparse
import sys
import time
import random
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import requests

# ── Configuración ─────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent.parent / "data"
DATE_FROM = "01/01/2020"
SLEEP_MIN = 1.5
SLEEP_MAX = 3.0
MAX_RETRY = 3

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept":           "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language":  "es-ES,es;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding":  "gzip, deflate, br",
    "X-Requested-With": "XMLHttpRequest",
    "Referer":          "https://es.investing.com/",
    "Origin":           "https://es.investing.com",
})

# ── Catálogo completo de fondos (todos los brokers) ───────────────────────────
ALL_FONDOS: dict[str, dict] = {

    # ── Santander AM ──────────────────────────────────────────────────────────
    "santander": {
        "0P0000KBXJ": {"nombre": "Santander Renta Mediano Plazo APV",         "perfil": "conservador"},
        "0P0000KBZ5": {"nombre": "Santander Renta Selecta Chile APV",          "perfil": "conservador"},
        "0P0000KBZ6": {"nombre": "Santander Renta Selecta Chile GLOBA",        "perfil": "conservador"},
        "0P0000KBZQ": {"nombre": "Santander Acciones Global Emergente APV",    "perfil": "agresivo"},
        "0P0000KBZR": {"nombre": "Santander Acciones Global Emergente EJECU",  "perfil": "agresivo"},
        "0P0000KBZS": {"nombre": "Santander Acciones Global Emergente INVER",  "perfil": "agresivo"},
        "0P0000KC0A": {"nombre": "Santander GO Acciones Globales ESG APV",     "perfil": "agresivo"},
        "0P0000KC0E": {"nombre": "Santander Renta Largo Plazo UF APV",         "perfil": "conservador"},
        "0P0000KC1P": {"nombre": "Santander GO Acciones Asia Emergente APV",   "perfil": "agresivo"},
        "0P0000KC1Q": {"nombre": "Santander GO Acciones Asia Emergente EJECU", "perfil": "agresivo"},
        "0P0000KC2J": {"nombre": "Santander B APV",                            "perfil": "moderado"},
        "0P0000KC2T": {"nombre": "Santander Bonos Nacionales APV",             "perfil": "conservador"},
        "0P0000KC3F": {"nombre": "Santander Acciones Chilenas APV",            "perfil": "agresivo"},
        "0P0000KC3M": {"nombre": "Santander Acciones Selectas Chile APV",      "perfil": "agresivo"},
        "0P0000KC3N": {"nombre": "Santander Acciones Selectas Chile PATRI",    "perfil": "agresivo"},
        "0P0000KNOT": {"nombre": "Santander Renta Extra Largo Plazo UF APV",   "perfil": "conservador"},
        "0P0000N9A0": {"nombre": "Santander Renta Extra Largo Plazo UF INVER", "perfil": "conservador"},
        "0P0000V0AL": {"nombre": "Santander Private Banking Agresivo APV",     "perfil": "agresivo"},
        "0P0000V0AN": {"nombre": "Santander Private Banking Equilibrio APV",   "perfil": "moderado"},
        "0P000102JP": {"nombre": "Santander Renta Selecta Chile PATRI",        "perfil": "conservador"},
    },

    # ── LarrainVial AM ────────────────────────────────────────────────────────
    "larrain_vial": {
        "0P0000K9XA": {"nombre": "LV Cuenta Activa Defensiva Dolar P",   "perfil": "conservador"},
        "0P0000K9XC": {"nombre": "LV Cuenta Activa Defensiva Dolar A",   "perfil": "conservador"},
        "0P0000K9XE": {"nombre": "LV Cuenta Activa Defensiva Dolar APV", "perfil": "conservador"},
        "0P0000KCCW": {"nombre": "LV Portfolio Lider F",                 "perfil": "moderado"},
        "0P0000KCD2": {"nombre": "LV Ahorro Corporativo APV",            "perfil": "conservador"},
        "0P0000KCDG": {"nombre": "LV Latinoamericano APV",               "perfil": "agresivo"},
        "0P0000KCDI": {"nombre": "LV Latinoamericano F",                 "perfil": "agresivo"},
        "0P0000KCE0": {"nombre": "LV Bonos Latam A",                     "perfil": "moderado"},
        "0P0000KCE1": {"nombre": "LV Bonos Latam APV",                   "perfil": "moderado"},
        "0P0000KCE3": {"nombre": "LV Asia APV",                          "perfil": "agresivo"},
        "0P0000KCE5": {"nombre": "LV Asia F",                            "perfil": "agresivo"},
        "0P0000KCEF": {"nombre": "LV Enfoque A",                         "perfil": "moderado"},
        "0P0000KCEK": {"nombre": "LV Enfoque APV",                       "perfil": "moderado"},
        "0P0000KCHS": {"nombre": "LV Acciones Nacionales APV",           "perfil": "agresivo"},
        "0P0000KCIT": {"nombre": "LV Ahorro Capital APV",                "perfil": "conservador"},
        "0P0000TFHA": {"nombre": "LV Acciones Nacionales I",             "perfil": "agresivo"},
        "0P0000TFVX": {"nombre": "LV Ahorro Estrategico I",              "perfil": "conservador"},
        "0P0000TFW3": {"nombre": "LV Ahorro Capital I",                  "perfil": "conservador"},
        "0P0000TG5U": {"nombre": "LV Bonos Latam F",                     "perfil": "moderado"},
        "0P0000TG5W": {"nombre": "LV Bonos Latam P",                     "perfil": "moderado"},
        "0P0000TG61": {"nombre": "LV Enfoque F",                         "perfil": "moderado"},
        "0P0000TG62": {"nombre": "LV Enfoque I",                         "perfil": "moderado"},
    },

    # ── BCI ───────────────────────────────────────────────────────────────────
    "bci": {
        "0P0000KA6F": {"nombre": "BCI Selección Bursátil APV",            "perfil": "agresivo"},
        "0P0000KA6R": {"nombre": "BCI Emergente Global APV",              "perfil": "agresivo"},
        "0P0000KA7G": {"nombre": "BCI de Personas APV",                   "perfil": "moderado"},
        "0P0000KA71": {"nombre": "BCI Cartera Dinámica Balanceada APV",   "perfil": "moderado"},
        "0P0000V2S1": {"nombre": "BCI Estrategia UF Hasta 3 años",        "perfil": "conservador"},
    },

    # ── Bice Inversiones ──────────────────────────────────────────────────────
    "bice": {
        "0P0000KA3H": {"nombre": "Bice Acciones Chile Mid Cap A",         "perfil": "agresivo"},
        "0P0000KA3L": {"nombre": "Bice Acciones Chile Mid Cap I",         "perfil": "agresivo"},
        "0P0000KA3T": {"nombre": "Bice Renta UF A",                       "perfil": "conservador"},
        "0P0000KA3U": {"nombre": "Bice Renta UF B",                       "perfil": "conservador"},
        "0P0000KA37": {"nombre": "Bice Estrategia Balanceada B",          "perfil": "moderado"},
        "0P0000N0W0": {"nombre": "Bice Estrategia Balanceada D",          "perfil": "moderado"},
        "0P0000RT12": {"nombre": "Bice Chile Activo B",                   "perfil": "agresivo"},
    },

    # ── BancoEstado ───────────────────────────────────────────────────────────
    "bancochile": {
        "0P0000MOIN": {"nombre": "BancoEstado Acciones Desarrolladas APV","perfil": "agresivo"},
        "0P0000Z8US": {"nombre": "BancoEstado Protección I",              "perfil": "conservador"},
        "0P0000Z8UU": {"nombre": "BancoEstado Más Renta Bicentenario I",  "perfil": "moderado"},
        "0P0000Z8UW": {"nombre": "BancoEstado Compromiso I",              "perfil": "moderado"},
        "0P0000KASM": {"nombre": "BancoEstado Compromiso A",              "perfil": "moderado"},
    },

    # ── Security ──────────────────────────────────────────────────────────────
    "security": {
        "0P0000N3XH": {"nombre": "Security Global APV1",                  "perfil": "agresivo"},
        "0P0000KBV4": {"nombre": "Security Crecimiento Estratégico B",    "perfil": "moderado"},
        "0P0000KNC5": {"nombre": "Security Index Fund US I-APV",          "perfil": "agresivo"},
        "0P0000KBJZ": {"nombre": "Security Gold B",                       "perfil": "agresivo"},
        "0P0000KBKO": {"nombre": "Security Deuda Corp. Latinoam. I-APV",  "perfil": "moderado"},
    },
}

INTERVAL_MAP = {
    "monthly": ("Monthly", "Datos_historicos_", ""),
    "daily":   ("Daily",   "Datos_diarios_",    "daily"),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _sleep():
    time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))


def init_session() -> bool:
    try:
        r = SESSION.get("https://es.investing.com/", timeout=15)
        r.raise_for_status()
        _sleep()
        return True
    except Exception as e:
        print(f"  ⚠️  No se pudo inicializar sesión: {e}")
        return False


def get_pair_id(morningstar_id: str, fund_name: str) -> str | None:
    """Busca el pairId interno de Investing.com para un fondo."""
    for query in [morningstar_id, fund_name]:
        for attempt in range(MAX_RETRY):
            try:
                r = SESSION.post(
                    "https://es.investing.com/search/service/searchTopBar",
                    data={"search_text": query},
                    timeout=15,
                )
                r.raise_for_status()
                data = r.json()
                for section in data.values():
                    if not isinstance(section, list):
                        continue
                    for item in section:
                        if not isinstance(item, dict):
                            continue
                        ticker = str(item.get("symbol", "") or item.get("ticker", "")).upper()
                        if morningstar_id.upper() in ticker or morningstar_id.upper() in str(item):
                            pid = item.get("pairId") or item.get("pair_ID") or item.get("id")
                            if pid:
                                return str(pid)
                _sleep()
                break
            except Exception as e:
                if attempt < MAX_RETRY - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"    ⚠️  Error buscando '{query}': {e}")
    return None


def fetch_historical(pair_id: str, interval_sec: str) -> pd.DataFrame | None:
    """Descarga datos históricos desde Investing.com para un intervalo dado."""
    end_date = datetime.today().strftime("%m/%d/%Y")
    for attempt in range(MAX_RETRY):
        try:
            r = SESSION.post(
                "https://es.investing.com/instruments/HistoricalDataAjax",
                data={
                    "curr_id":      pair_id,
                    "st_date":      DATE_FROM,
                    "end_date":     end_date,
                    "interval_sec": interval_sec,
                    "action":       "historical_data",
                },
                timeout=25,
            )
            r.raise_for_status()
            tables = pd.read_html(r.text)
            if tables:
                return tables[0]
        except Exception as e:
            if attempt < MAX_RETRY - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    ⚠️  Error descargando pairId={pair_id}: {e}")
    return None


def needs_update(path: Path, interval: str) -> bool:
    """True si el archivo necesita actualizarse."""
    if not path.exists():
        return True
    mtime   = datetime.fromtimestamp(path.stat().st_mtime)
    now     = datetime.today()
    if interval == "monthly":
        return not (mtime.year == now.year and mtime.month == now.month)
    else:  # daily
        return mtime.date() < date.today()


def save_csv(df: pd.DataFrame, broker: str, mid: str, prefix: str, subdir: str) -> Path:
    """Guarda DataFrame en data/{broker}/{subdir}/{prefix}{mid}.csv"""
    out_dir = BASE_DIR / broker / subdir if subdir else BASE_DIR / broker
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{prefix}{mid}.csv"

    rename = {
        "Price":    "Último", "Precio": "Último", "Close": "Último",
        "Open":     "Apertura", "High": "Máximo", "Low": "Mínimo",
        "Change %": "% var.",   "Var. %": "% var.",
    }
    df = df.rename(columns=rename)
    for c in df.columns:
        if "fecha" in c.lower() or "date" in c.lower():
            df = df.rename(columns={c: "Fecha"})
            break

    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# ── Lógica principal ──────────────────────────────────────────────────────────

def update_broker(broker: str, fondos: dict, interval: str, dry_run: bool) -> dict:
    stats = {"ok": 0, "error": 0, "skip": 0}
    iv_sec, prefix, subdir = INTERVAL_MAP[interval]

    print(f"\n{'─'*65}")
    print(f"📦  {broker.upper()} — {len(fondos)} fondos — [{interval}]")
    print(f"{'─'*65}")

    for mid, info in fondos.items():
        nombre   = info["nombre"]
        out_dir  = BASE_DIR / broker / subdir if subdir else BASE_DIR / broker
        out_path = out_dir / f"{prefix}{mid}.csv"

        if not needs_update(out_path, interval):
            print(f"  ⏭️  {nombre[:52]:<52}  ya actualizado")
            stats["skip"] += 1
            continue

        print(f"  🔄  {nombre[:52]:<52}", end="  ", flush=True)

        pair_id = get_pair_id(mid, nombre)
        if not pair_id:
            print("❌  pairId no encontrado")
            stats["error"] += 1
            _sleep()
            continue

        df = fetch_historical(pair_id, iv_sec)
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


def run(brokers: dict, interval: str, dry_run: bool):
    total = {"ok": 0, "error": 0, "skip": 0}
    for broker, fondos in brokers.items():
        s = update_broker(broker, fondos, interval, dry_run)
        for k in total:
            total[k] += s[k]
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Actualiza datos de fondos mutuos chilenos desde Investing.com"
    )
    parser.add_argument(
        "--broker", default="all",
        choices=["all"] + list(ALL_FONDOS.keys()),
        help="Broker a actualizar (default: all)",
    )
    parser.add_argument(
        "--interval", default="monthly",
        choices=["monthly", "daily", "both"],
        help="Frecuencia de datos (default: monthly)",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="No escribir archivos")
    args = parser.parse_args()

    brokers = ALL_FONDOS if args.broker == "all" else {args.broker: ALL_FONDOS[args.broker]}
    intervals = ["monthly", "daily"] if args.interval == "both" else [args.interval]

    print("=" * 65)
    print("🇨🇱  Actualizador de Fondos Mutuos — Investing.com")
    print(f"    Fecha     : {datetime.today().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"    Intervalo : {args.interval}")
    print(f"    Brokers   : {', '.join(brokers.keys())}")
    print(f"    Modo      : {'dry-run' if args.dry_run else 'escritura'}")
    print("=" * 65)

    if not init_session():
        print("⛔  No se pudo conectar a Investing.com")
        sys.exit(1)

    any_error = False
    for interval in intervals:
        total = run(brokers, interval, args.dry_run)
        print(f"\n{'='*65}")
        print(f"[{interval}] ✅ {total['ok']} ok · ❌ {total['error']} errores · ⏭️  {total['skip']} sin cambios")
        if total["error"] > 0 and total["ok"] == 0:
            any_error = True

    if any_error:
        sys.exit(1)


if __name__ == "__main__":
    main()
