"""
src/loader.py
-------------
Carga y normalización de fondos mutuos + activos internacionales.
El SP500 se integra como un activo más en la matriz de retornos.

Author: Proyecto Fondos Mutuos Chile
"""

import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Tipo de cambio USD/CLP aproximado por año ────────────────────────────────
USD_CLP_MENSUAL = {
    "2017": 660, "2018": 641, "2019": 703, "2020": 792,
    "2021": 779, "2022": 873, "2023": 878, "2024": 945,
    "2025": 960, "2026": 970,
}

# ── Metadata de fondos ───────────────────────────────────────────────────────
FONDOS_SANTANDER = {
    "0P0000KBXJ": {"nombre": "Santander Renta Mediano Plazo APV",          "perfil": "conservador", "moneda": "CLP"},
    "0P0000KBZ5": {"nombre": "Santander Renta Selecta Chile APV",           "perfil": "conservador", "moneda": "CLP"},
    "0P0000KBZ6": {"nombre": "Santander Renta Selecta Chile GLOBA",         "perfil": "conservador", "moneda": "CLP"},
    "0P0000KBZQ": {"nombre": "Santander Acciones Global Emergente APV",     "perfil": "agresivo",    "moneda": "CLP"},
    "0P0000KBZR": {"nombre": "Santander Acciones Global Emergente EJECU",   "perfil": "agresivo",    "moneda": "CLP"},
    "0P0000KBZS": {"nombre": "Santander Acciones Global Emergente INVER",   "perfil": "agresivo",    "moneda": "CLP"},
    "0P0000KC0A": {"nombre": "Santander GO Acciones Globales ESG APV",      "perfil": "agresivo",    "moneda": "CLP"},
    "0P0000KC0E": {"nombre": "Santander Renta Largo Plazo UF APV",          "perfil": "conservador", "moneda": "CLP"},
    "0P0000KC1P": {"nombre": "Santander GO Acciones Asia Emergente APV",    "perfil": "agresivo",    "moneda": "CLP"},
    "0P0000KC1Q": {"nombre": "Santander GO Acciones Asia Emergente EJECU",  "perfil": "agresivo",    "moneda": "CLP"},
    "0P0000KC2J": {"nombre": "Santander B APV",                             "perfil": "moderado",    "moneda": "CLP"},
    "0P0000KC2T": {"nombre": "Santander Bonos Nacionales APV",              "perfil": "conservador", "moneda": "CLP"},
    "0P0000KC3F": {"nombre": "Santander Acciones Chilenas APV",             "perfil": "agresivo",    "moneda": "CLP"},
    "0P0000KC3M": {"nombre": "Santander Acciones Selectas Chile APV",       "perfil": "agresivo",    "moneda": "CLP"},
    "0P0000KC3N": {"nombre": "Santander Acciones Selectas Chile PATRI",     "perfil": "agresivo",    "moneda": "CLP"},
    "0P0000KNOT": {"nombre": "Santander Renta Extra Largo Plazo UF APV",    "perfil": "conservador", "moneda": "CLP"},
    "0P0000N9A0": {"nombre": "Santander Renta Extra Largo Plazo UF INVER",  "perfil": "conservador", "moneda": "CLP"},
    "0P0000V0AL": {"nombre": "Santander Private Banking Agresivo APV",      "perfil": "agresivo",    "moneda": "CLP"},
    "0P0000V0AN": {"nombre": "Santander Private Banking Equilibrio APV",    "perfil": "moderado",    "moneda": "CLP"},
    "0P000102JP": {"nombre": "Santander Renta Selecta Chile PATRI",         "perfil": "conservador", "moneda": "CLP"},
}

FONDOS_LARRAIN_VIAL = {
    "0P0000K9XA": {"nombre": "LV Cuenta Activa Defensiva Dolar P",   "perfil": "conservador", "moneda": "USD"},
    "0P0000K9XC": {"nombre": "LV Cuenta Activa Defensiva Dolar A",   "perfil": "conservador", "moneda": "USD"},
    "0P0000K9XE": {"nombre": "LV Cuenta Activa Defensiva Dolar APV", "perfil": "conservador", "moneda": "USD"},
    "0P0000KCCW": {"nombre": "LV Portfolio Lider F",                 "perfil": "moderado",    "moneda": "CLP"},
    "0P0000KCD2": {"nombre": "LV Ahorro Corporativo APV",            "perfil": "conservador", "moneda": "CLP"},
    "0P0000KCDG": {"nombre": "LV Latinoamericano APV",               "perfil": "agresivo",    "moneda": "CLP"},
    "0P0000KCDI": {"nombre": "LV Latinoamericano F",                 "perfil": "agresivo",    "moneda": "CLP"},
    "0P0000KCE0": {"nombre": "LV Bonos Latam A",                     "perfil": "moderado",    "moneda": "CLP"},
    "0P0000KCE1": {"nombre": "LV Bonos Latam APV",                   "perfil": "moderado",    "moneda": "CLP"},
    "0P0000KCE3": {"nombre": "LV Asia APV",                          "perfil": "agresivo",    "moneda": "CLP"},
    "0P0000KCE5": {"nombre": "LV Asia F",                            "perfil": "agresivo",    "moneda": "CLP"},
    "0P0000KCEF": {"nombre": "LV Enfoque A",                         "perfil": "moderado",    "moneda": "CLP"},
    "0P0000KCEK": {"nombre": "LV Enfoque APV",                       "perfil": "moderado",    "moneda": "CLP"},
    "0P0000KCHS": {"nombre": "LV Acciones Nacionales APV",           "perfil": "agresivo",    "moneda": "CLP"},
    "0P0000KCIT": {"nombre": "LV Ahorro Capital APV",                "perfil": "conservador", "moneda": "CLP"},
    "0P0000TFHA": {"nombre": "LV Acciones Nacionales I",             "perfil": "agresivo",    "moneda": "CLP"},
    "0P0000TFVX": {"nombre": "LV Ahorro Estrategico I",              "perfil": "conservador", "moneda": "CLP"},
    "0P0000TFW3": {"nombre": "LV Ahorro Capital I",                  "perfil": "conservador", "moneda": "CLP"},
    "0P0000TG5U": {"nombre": "LV Bonos Latam F",                     "perfil": "moderado",    "moneda": "CLP"},
    "0P0000TG5W": {"nombre": "LV Bonos Latam P",                     "perfil": "moderado",    "moneda": "CLP"},
    "0P0000TG61": {"nombre": "LV Enfoque F",                         "perfil": "moderado",    "moneda": "CLP"},
    "0P0000TG62": {"nombre": "LV Enfoque I",                         "perfil": "moderado",    "moneda": "CLP"},
}

# SP500 como activo — integrado desde el inicio
FONDOS_EXTERNOS = {
    "SP500": {"nombre": "S&P 500 (CLP)", "perfil": "sp500", "moneda": "USD"},
}

ALL_FONDOS = {**FONDOS_SANTANDER, **FONDOS_LARRAIN_VIAL, **FONDOS_EXTERNOS}

MESES = {
    "Ene":"01","Feb":"02","Mar":"03","Abr":"04","May":"05","Jun":"06",
    "Jul":"07","Ago":"08","Sep":"09","Oct":"10","Nov":"11","Dic":"12",
}

SP500_COLOR = "#00b4d8"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _num(s):
    if pd.isna(s) or str(s).strip() in ("", "-"):
        return np.nan
    s = str(s).strip().replace("%", "").replace(" ", "")
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _fecha_mes_anio(s):
    """'Mar 2026' → '2026-03-01'"""
    try:
        p = str(s).strip().split()
        return "{}-{}-01".format(p[1], MESES[p[0]])
    except Exception:
        return None


def _fecha_ddmmyyyy(s):
    """'01.03.2026' → '2026-03-01'"""
    try:
        p = str(s).strip().split(".")
        return "{}-{}-01".format(p[2], p[1])
    except Exception:
        return None


def _usd_clp(fecha_str):
    year = str(fecha_str)[:4]
    return USD_CLP_MENSUAL.get(year, 900)


def _col_precio(columnas):
    for c in columnas:
        if "ltim" in c:
            return c
    raise ValueError("No se encontro columna de precio. Columnas: {}".format(columnas))


# ── Carga de fondos chilenos ─────────────────────────────────────────────────

def cargar_csv(path, fondo_id, corredora):
    info      = ALL_FONDOS.get(fondo_id, {"nombre": fondo_id, "perfil": "moderado", "moneda": "CLP"})
    moneda    = info.get("moneda", "CLP")

    df        = pd.read_csv(str(path), encoding="utf-8-sig")
    col_p     = _col_precio(df.columns.tolist())

    df["fecha"]       = df["Fecha"].apply(_fecha_mes_anio)
    df["valor_orig"]  = df[col_p].apply(_num)
    df["retorno_pct"] = df["% var."].apply(_num)
    df["retorno"]     = df["retorno_pct"] / 100
    df["fondo_id"]    = fondo_id
    df["nombre"]      = info["nombre"]
    df["perfil"]      = info["perfil"]
    df["corredora"]   = corredora
    df["moneda_orig"] = moneda

    if moneda == "USD":
        df["tc"]          = df["fecha"].apply(lambda x: _usd_clp(x) if x else np.nan)
        df["valor_cuota"] = df["valor_orig"] * df["tc"]
    else:
        df["valor_cuota"] = df["valor_orig"]

    cols = ["fecha","fondo_id","nombre","perfil","corredora",
            "moneda_orig","valor_cuota","retorno_pct","retorno"]
    df = df[cols].dropna(subset=["fecha","valor_cuota"]).copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df.sort_values("fecha").reset_index(drop=True)


# ── Carga del SP500 ──────────────────────────────────────────────────────────

def cargar_sp500_como_fondo(path="data/raw/sp500.csv"):
    """
    Carga el SP500 en el mismo formato que los fondos chilenos.
    Convierte precios USD → CLP y calcula retornos mensuales.
    """
    df    = pd.read_csv(str(path), encoding="utf-8-sig")
    col_p = _col_precio(df.columns.tolist())

    df["fecha"]      = df["Fecha"].apply(_fecha_ddmmyyyy)
    df["precio_usd"] = df[col_p].apply(_num)
    df["retorno_pct"]= df["% var."].apply(_num)
    df["retorno"]    = df["retorno_pct"] / 100

    df = df.dropna(subset=["fecha","precio_usd"]).copy()
    df["fecha"]      = pd.to_datetime(df["fecha"])
    df               = df.sort_values("fecha").reset_index(drop=True)

    df["tc"]          = df["fecha"].apply(lambda x: _usd_clp(str(x)[:10]))
    df["valor_cuota"] = df["precio_usd"] * df["tc"]
    df["fondo_id"]    = "SP500"
    df["nombre"]      = "S&P 500 (CLP)"
    df["perfil"]      = "sp500"
    df["corredora"]   = "externo"
    df["moneda_orig"] = "USD"

    cols = ["fecha","fondo_id","nombre","perfil","corredora",
            "moneda_orig","valor_cuota","retorno_pct","retorno"]
    return df[cols].dropna(subset=["fecha","valor_cuota"])


# ── Carga completa ───────────────────────────────────────────────────────────

def cargar_corredora(carpeta, corredora, prefijo="Datos_historicos_"):
    carpeta  = Path(carpeta)
    archivos = sorted([f for f in os.listdir(str(carpeta)) if f.endswith(".csv")])
    dfs = []
    for nombre in archivos:
        fondo_id = nombre.replace(prefijo, "").replace(".csv", "")
        try:
            dfs.append(cargar_csv(carpeta / nombre, fondo_id, corredora))
        except Exception as e:
            print("  Error en {}: {}".format(nombre, e))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def cargar_todos(base_dir="data", fecha_inicio="2020-01-01",
                  fecha_fin=None, incluir_sp500=True):
    """
    Carga todas las corredoras + SP500 como activo adicional.

    Parámetros
    ----------
    incluir_sp500 : bool — agrega SP500 al universo de activos

    Retorna
    -------
    df_long  : DataFrame long
    retornos : DataFrame wide fecha × fondo_id
    precios  : DataFrame wide fecha × fondo_id
    meta     : DataFrame con metadata por fondo_id
    """
    base_dir = Path(base_dir)
    dfs = []

    # Fondos chilenos
    for key, prefijo in [("santander", "Datos_historicos_"),
                          ("larrain_vial", "Datos_historicos_")]:
        ruta = base_dir / key
        if ruta.exists():
            df = cargar_corredora(ruta, key, prefijo)
            if not df.empty:
                dfs.append(df)
                print("  {}: {} fondos, {} registros".format(
                    key, df["fondo_id"].nunique(), len(df)))

    # SP500
    if incluir_sp500:
        sp500_path = base_dir / "raw" / "sp500.csv"
        if sp500_path.exists():
            df_sp = cargar_sp500_como_fondo(str(sp500_path))
            dfs.append(df_sp)
            print("  SP500: 1 activo, {} registros".format(len(df_sp)))
        else:
            print("  SP500: archivo no encontrado en {}".format(sp500_path))

    if not dfs:
        raise ValueError("No se encontraron datos.")

    df_long = pd.concat(dfs, ignore_index=True)

    # Filtrar fechas
    if fecha_inicio:
        df_long = df_long[df_long["fecha"] >= pd.to_datetime(fecha_inicio)]
    if fecha_fin:
        df_long = df_long[df_long["fecha"] <= pd.to_datetime(fecha_fin)]

    retornos = df_long.pivot_table(index="fecha", columns="fondo_id", values="retorno")
    precios  = df_long.pivot_table(index="fecha", columns="fondo_id", values="valor_cuota")
    meta     = (df_long[["fondo_id","nombre","perfil","corredora","moneda_orig"]]
                .drop_duplicates("fondo_id")
                .set_index("fondo_id"))

    n_fondos_chilenos = len(meta[meta["corredora"] != "externo"])
    print("\nTotal: {} fondos ({} chilenos + SP500) | {} meses | {} → {}".format(
        len(meta), n_fondos_chilenos, len(retornos),
        retornos.index.min().date(), retornos.index.max().date()))

    return df_long, retornos, precios, meta
