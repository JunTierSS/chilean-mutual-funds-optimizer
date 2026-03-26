#!/usr/bin/env python3
"""
scripts/fetch_investing.py
--------------------------
Descarga datos históricos mensuales de fondos mutuos chilenos desde Investing.com.
Guarda en data/{broker}/Datos_historicos_{morningstar_id}.csv (mismo formato que los
archivos existentes de Santander/LarrainVial).

Uso:
    python3 scripts/fetch_investing.py              # actualiza todos los brokers
    python3 scripts/fetch_investing.py --broker bci # solo BCI
    python3 scripts/fetch_investing.py --dry-run    # sin escribir archivos

Author: Junwei He — MSc Data Science (c), Universidad de Chile
"""

import argparse
import json
import sys
import time
import random
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# ── Configuración ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent / "data"
DATE_FROM  = "01/01/2020"
SLEEP_MIN  = 1.5   # segundos entre requests (cortesía)
SLEEP_MAX  = 3.0
MAX_RETRY  = 3

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "es-ES,es;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "X-Requested-With": "XMLHttpRequest",
    "Referer":  "https://es.investing.com/",
    "Origin":   "https://es.investing.com",
})

# ── Catálogo de nuevos fondos ─────────────────────────────────────────────────
# Morningstar IDs extraídos de Investing.com (es.investing.com/funds/chile-funds)
FONDOS_NUEVOS: dict[str, dict] = {

    "bci": {
        "0P0000KA6F": {"nombre": "BCI Selección Bursátil APV",            "perfil": "agresivo",    "moneda": "CLP"},
        "0P0000KA6R": {"nombre": "BCI Emergente Global APV",              "perfil": "agresivo",    "moneda": "CLP"},
        "0P0000KA7G": {"nombre": "BCI de Personas APV",                   "perfil": "moderado",    "moneda": "CLP"},
        "0P0000KA71": {"nombre": "BCI Cartera Dinámica Balanceada APV",   "perfil": "moderado",    "moneda": "CLP"},
        "0P0000V2S1": {"nombre": "BCI Estrategia UF Hasta 3 años",        "perfil": "conservador", "moneda": "CLP"},
    },

    "bice": {
        "0P0000KA3H": {"nombre": "Bice Acciones Chile Mid Cap A",         "perfil": "agresivo",    "moneda": "CLP"},
        "0P0000KA3L": {"nombre": "Bice Acciones Chile Mid Cap I",         "perfil": "agresivo",    "moneda": "CLP"},
        "0P0000KA3T": {"nombre": "Bice Renta UF A",                       "perfil": "conservador", "moneda": "CLP"},
        "0P0000KA3U": {"nombre": "Bice Renta UF B",                       "perfil": "conservador", "moneda": "CLP"},
        "0P0000KA37": {"nombre": "Bice Estrategia Balanceada B",          "perfil": "moderado",    "moneda": "CLP"},
        "0P0000N0W0": {"nombre": "Bice Estrategia Balanceada D",          "perfil": "moderado",    "moneda": "CLP"},
        "0P0000RT12": {"nombre": "Bice Chile Activo B",                   "perfil": "agresivo",    "moneda": "CLP"},
    },

    "bancochile": {
        "0P0000MOIN": {"nombre": "BancoEstado Acciones Desarrolladas APV","perfil": "agresivo",    "moneda": "CLP"},
        "0P0000Z8US": {"nombre": "BancoEstado Protección I",              "perfil": "conservador", "moneda": "CLP"},
        "0P0000Z8UU": {"nombre": "BancoEstado Más Renta Bicentenario I",  "perfil": "moderado",    "moneda": "CLP"},
        "0P0000Z8UW": {"nombre": "BancoEstado Compromiso I",              "perfil": "moderado",    "moneda": "CLP"},
        "0P0000KASM": {"nombre": "BancoEstado Compromiso A",              "perfil": "moderado",    "moneda": "CLP"},
    },

    "security": {
        "0P0000N3XH": {"nombre": "Security Global APV1",                  "perfil": "agresivo",    "moneda": "CLP"},
        "0P0000KBV4": {"nombre": "Security Crecimiento Estratégico B",    "perfil": "moderado",    "moneda": "CLP"},
        "0P0000KNC5": {"nombre": "Security Index Fund US I-APV",          "perfil": "agresivo",    "moneda": "USD"},
        "0P0000KBJZ": {"nombre": "Security Gold B",                       "perfil": "agresivo",    "moneda": "CLP"},
        "0P0000KBKO": {"nombre": "Security Deuda Corp. Latinoam. I-APV",  "perfil": "moderado",    "moneda": "CLP"},
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _sleep():
    time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))


def init_session() -> bool:
    """Visita la página principal para inicializar cookies."""
    try:
        r = SESSION.get("https://es.investing.com/", timeout=15)
        r.raise_for_status()
        _sleep()
        return True
    except Exception as e:
        print(f"  ⚠️  No se pudo inicializar sesión: {e}")
        return False


def get_pair_id(morningstar_id: str, fund_name: str) -> str | None:
    """
    Busca el pairId interno de Investing.com para un fondo.
    Intenta primero por Morningstar ID, luego por nombre.
    """
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

                # La respuesta tiene secciones: quotes, funds, equities, etc.
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


def fetch_historical(pair_id: str) -> pd.DataFrame | None:
    """
    Descarga datos históricos mensuales desde Investing.com.
    Retorna DataFrame con columnas: Fecha, Último, Apertura, Máximo, Mínimo, % var.
    """
    end_date = datetime.today().strftime("%m/%d/%Y")

    for attempt in range(MAX_RETRY):
        try:
            r = SESSION.post(
                "https://es.investing.com/instruments/HistoricalDataAjax",
                data={
                    "curr_id":      pair_id,
                    "st_date":      DATE_FROM,
                    "end_date":     end_date,
                    "interval_sec": "Monthly",
                    "action":       "historical_data",
                },
                timeout=20,
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


def already_current(path: Path) -> bool:
    """True si el archivo fue actualizado este mes."""
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    now   = datetime.today()
    return mtime.year == now.year and mtime.month == now.month


def save_csv(df: pd.DataFrame, broker: str, mid: str) -> Path:
    """Guarda en data/{broker}/Datos_historicos_{mid}.csv."""
    out_dir = BASE_DIR / broker
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"Datos_historicos_{mid}.csv"

    # Normalizar nombres de columnas al formato estándar del proyecto
    rename = {
        "Price":   "Último",
        "Precio":  "Último",
        "Close":   "Último",
        "Open":    "Apertura",
        "High":    "Máximo",
        "Low":     "Mínimo",
        "Change %":"% var.",
        "Var. %":  "% var.",
    }
    df = df.rename(columns=rename)

    # Asegurar columna Fecha
    for c in df.columns:
        if "fecha" in c.lower() or "date" in c.lower():
            df = df.rename(columns={c: "Fecha"})
            break

    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# ── Lógica principal ──────────────────────────────────────────────────────────

def update_broker(broker: str, fondos: dict, dry_run: bool = False) -> dict:
    stats = {"ok": 0, "error": 0, "skip": 0}
    print(f"\n{'─'*60}")
    print(f"📦  {broker.upper()} — {len(fondos)} fondos")
    print(f"{'─'*60}")

    for mid, info in fondos.items():
        nombre   = info["nombre"]
        out_path = BASE_DIR / broker / f"Datos_historicos_{mid}.csv"

        if already_current(out_path):
            print(f"  ⏭️  {nombre[:50]:<50}  ya actualizado")
            stats["skip"] += 1
            continue

        print(f"  🔄  {nombre[:50]:<50}", end="  ", flush=True)

        pair_id = get_pair_id(mid, nombre)
        if not pair_id:
            print("❌  pairId no encontrado")
            stats["error"] += 1
            _sleep()
            continue

        df = fetch_historical(pair_id)
        if df is None or df.empty:
            print("❌  sin datos")
            stats["error"] += 1
            _sleep()
            continue

        if dry_run:
            print(f"✅  {len(df):>4} filas  (dry-run, no guardado)")
        else:
            path = save_csv(df, broker, mid)
            print(f"✅  {len(df):>4} filas  → {path.name}")

        stats["ok"] += 1
        _sleep()

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Actualiza datos de fondos mutuos chilenos desde Investing.com"
    )
    parser.add_argument(
        "--broker", default="all",
        choices=["all"] + list(FONDOS_NUEVOS.keys()),
        help="Broker a actualizar (default: all)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Descarga datos pero no escribe archivos",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("🇨🇱  Actualizador de Fondos Mutuos — Investing.com")
    print(f"    Fecha : {datetime.today().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"    Modo  : {'dry-run' if args.dry_run else 'escritura'}")
    print("=" * 60)

    if not init_session():
        print("⛔  No se pudo conectar a Investing.com")
        sys.exit(1)

    brokers = (
        FONDOS_NUEVOS
        if args.broker == "all"
        else {args.broker: FONDOS_NUEVOS[args.broker]}
    )

    total = {"ok": 0, "error": 0, "skip": 0}
    for broker, fondos in brokers.items():
        s = update_broker(broker, fondos, dry_run=args.dry_run)
        for k in total:
            total[k] += s[k]

    print(f"\n{'='*60}")
    print(f"✅  Completado: {total['ok']} ok · {total['error']} errores · {total['skip']} sin cambios")
    print(f"{'='*60}")

    if total["error"] > 0 and total["ok"] == 0:
        sys.exit(1)   # fallo total → error en CI


if __name__ == "__main__":
    main()
