"""
src/benchmarks.py
-----------------
Carga y análisis de benchmarks externos: SP500 y IPSA.

El SP500 se carga desde CSV local (Investing.com) y se convierte a CLP.
El IPSA se aproxima desde los fondos de acciones nacionales disponibles
(proxy si no hay CSV directo).

Funcionalidades:
- Cargar SP500 en USD y CLP
- Construir proxy IPSA desde fondos de acciones chilenas
- Calcular alpha vs benchmark (Jensen's alpha)
- Comparativa de retorno acumulado portafolio vs benchmarks
- Tabla de métricas comparativa
"""

import numpy as np
import pandas as pd
from pathlib import Path
from src.loader import _num, USD_CLP_MENSUAL
from src.metrics import sharpe_ratio, max_drawdown, TPM_ANUAL

MESES = {
    "Ene":"01","Feb":"02","Mar":"03","Abr":"04","May":"05","Jun":"06",
    "Jul":"07","Ago":"08","Sep":"09","Oct":"10","Nov":"11","Dic":"12",
}


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


# ── SP500 ────────────────────────────────────────────────────────────────────

def cargar_sp500(path="data/raw/sp500.csv", fecha_inicio="2020-01-01"):
    """
    Carga el SP500 desde CSV de Investing.com.
    Convierte a CLP usando tipo de cambio histórico aproximado.

    Retorna DataFrame con columnas: precio_usd, precio_clp, retorno_usd, retorno_clp
    """
    df = pd.read_csv(str(path), encoding="utf-8-sig")
    col_p = [c for c in df.columns if "ltim" in c][0]

    df["fecha"]       = df["Fecha"].apply(_fecha_ddmmyyyy)
    df["precio_usd"]  = df[col_p].apply(_num)
    df["retorno_usd"] = df["% var."].apply(_num) / 100

    df = df.dropna(subset=["fecha", "precio_usd"]).copy()
    df["fecha"]      = pd.to_datetime(df["fecha"])
    df               = df.sort_values("fecha").reset_index(drop=True)

    # Convertir a CLP
    df["tc"]          = df["fecha"].apply(lambda x: _usd_clp(str(x)[:10]))
    df["precio_clp"]  = df["precio_usd"] * df["tc"]
    df["retorno_clp"] = df["precio_clp"].pct_change()

    if fecha_inicio:
        df = df[df["fecha"] >= pd.to_datetime(fecha_inicio)]

    df = df.set_index("fecha")
    return df


def get_retornos_sp500(path="data/raw/sp500.csv", fecha_inicio="2020-01-01",
                        moneda="clp"):
    """
    Retorna Serie de retornos mensuales del SP500.
    moneda: 'clp' | 'usd'
    """
    df = cargar_sp500(path, fecha_inicio)
    col = "retorno_clp" if moneda == "clp" else "retorno_usd"
    return df[col].dropna().rename("SP500")


# ── IPSA (proxy) ─────────────────────────────────────────────────────────────

def construir_proxy_ipsa(retornos, meta, fecha_inicio="2020-01-01"):
    """
    Construye un proxy del IPSA promediando los fondos de acciones nacionales.
    Fondos incluidos: perfil agresivo + nombre contiene 'Chile' o 'Nacional' o 'Chilenas'.
    """
    keywords = ["chile", "nacional", "chilenas", "selectas chile", "renta selecta chile"]
    fondos_ipsa = [
        f for f in retornos.columns
        if f in meta.index
        and meta.loc[f, "perfil"] == "agresivo"
        and any(kw in meta.loc[f, "nombre"].lower() for kw in keywords)
    ]
    if not fondos_ipsa:
        # Fallback: todos los fondos agresivos
        fondos_ipsa = [f for f in retornos.columns
                       if f in meta.index and meta.loc[f, "perfil"] == "agresivo"]

    ret_proxy = retornos[fondos_ipsa].mean(axis=1).rename("IPSA_proxy")
    return ret_proxy, fondos_ipsa


# ── Alpha de Jensen ──────────────────────────────────────────────────────────

def calcular_alpha(retornos_portafolio, retornos_benchmark,
                    tpm_mensual=TPM_ANUAL/12):
    """
    Calcula el Alpha de Jensen:
    α = E[r_p] - [r_f + β·(E[r_m] - r_f)]

    donde β = cov(r_p, r_m) / var(r_m)

    Un alpha positivo indica que el portafolio supera al benchmark
    ajustado por riesgo sistémico.

    Retorna dict con alpha, beta, r_squared, p_value
    """
    from scipy import stats as scipy_stats

    # Alinear series
    df = pd.DataFrame({
        "port": retornos_portafolio,
        "bench": retornos_benchmark,
    }).dropna()

    if len(df) < 12:
        return {"alpha": np.nan, "beta": np.nan, "r2": np.nan, "p_value": np.nan}

    # Exceso de retorno
    exceso_p = df["port"]  - tpm_mensual
    exceso_m = df["bench"] - tpm_mensual

    # Regresion: exceso_p = alpha + beta * exceso_m + epsilon
    slope, intercept, r, p_val, _ = scipy_stats.linregress(exceso_m, exceso_p)

    return {
        "alpha":       float(intercept * 12),   # anualizado
        "beta":        float(slope),
        "r2":          float(r**2),
        "p_value":     float(p_val),
        "tracking_error": float((df["port"] - df["bench"]).std() * np.sqrt(12)),
        "information_ratio": float((df["port"] - df["bench"]).mean() /
                                    (df["port"] - df["bench"]).std() * np.sqrt(12))
                              if (df["port"] - df["bench"]).std() > 0 else np.nan,
        "n_meses": len(df),
    }


def tabla_alpha(portafolios, retornos, meta,
                sp500_path="data/raw/sp500.csv",
                fecha_inicio="2020-01-01"):
    """
    Tabla comparativa de alpha de cada portafolio vs SP500 e IPSA.
    """
    from src.metrics import simular_historico

    ret_sp500 = get_retornos_sp500(sp500_path, fecha_inicio)
    ret_ipsa, _ = construir_proxy_ipsa(retornos, meta, fecha_inicio)

    rows = []
    for p_id, res in portafolios.items():
        fondos = [f for f in res["composicion"] if f in retornos.columns]
        if not fondos:
            continue
        w = np.array([res["composicion"][f] for f in fondos])
        w = w / w.sum()
        R = retornos[fondos].dropna()
        ret_port = pd.Series(R.values @ w, index=R.index, name=p_id)

        # Alpha vs SP500
        alpha_sp = calcular_alpha(ret_port, ret_sp500)
        # Alpha vs IPSA
        alpha_ip = calcular_alpha(ret_port, ret_ipsa)

        rows.append({
            "portafolio":     res["label"],
            "ret_anual":      float(ret_port.mean() * 12),
            "sharpe":         res["sharpe"],
            "alpha_vs_sp500": alpha_sp["alpha"],
            "beta_vs_sp500":  alpha_sp["beta"],
            "r2_vs_sp500":    alpha_sp["r2"],
            "ir_vs_sp500":    alpha_sp["information_ratio"],
            "alpha_vs_ipsa":  alpha_ip["alpha"],
            "beta_vs_ipsa":   alpha_ip["beta"],
            "tracking_err_sp500": alpha_sp["tracking_error"],
        })

    return pd.DataFrame(rows)


# ── Stress Testing ────────────────────────────────────────────────────────────

STRESS_SCENARIOS = {
    "COVID Crash (Mar 2020)": {
        "inicio": "2020-02-01",
        "fin":    "2020-04-01",
        "descripcion": "Caída mercados globales por pandemia COVID-19",
    },
    "Recuperacion COVID (May-Dic 2020)": {
        "inicio": "2020-05-01",
        "fin":    "2020-12-01",
        "descripcion": "Rally post-COVID, mercados globales al alza",
    },
    "Inicio subida TPM (Sep-Dic 2021)": {
        "inicio": "2021-09-01",
        "fin":    "2021-12-01",
        "descripcion": "Banco Central comienza a subir la TPM en Chile",
    },
    "Crisis inflacion (2022)": {
        "inicio": "2022-01-01",
        "fin":    "2022-12-01",
        "descripcion": "Inflación máxima histórica, TPM en alza agresiva",
    },
    "TPM en peak (2023)": {
        "inicio": "2023-01-01",
        "fin":    "2023-12-01",
        "descripcion": "TPM al 11.25%, mercados locales bajo presión",
    },
    "Normalizacion (2024)": {
        "inicio": "2024-01-01",
        "fin":    "2024-12-01",
        "descripcion": "Bajada gradual de TPM, recuperación moderada",
    },
}


def stress_test(portafolios, retornos, meta,
                sp500_path="data/raw/sp500.csv"):
    """
    Evalúa el comportamiento de cada portafolio en escenarios históricos de estrés.

    Retorna DataFrame con retorno acumulado de cada portafolio en cada escenario.
    """
    ret_sp500 = get_retornos_sp500(sp500_path, fecha_inicio="2019-01-01")
    ret_ipsa, _ = construir_proxy_ipsa(retornos, meta)

    # Portafolios clave
    port_sel = {k: v for k, v in portafolios.items()
                if k in ["conservador", "moderado", "agresivo", "optimo"]}

    rows = []
    for escenario, cfg in STRESS_SCENARIOS.items():
        row = {"Escenario": escenario, "Descripcion": cfg["descripcion"]}

        for p_id, res in port_sel.items():
            fondos = [f for f in res["composicion"] if f in retornos.columns]
            if not fondos:
                continue
            w = np.array([res["composicion"][f] for f in fondos])
            w = w / w.sum()
            R = retornos.loc[cfg["inicio"]:cfg["fin"], fondos].dropna()
            if R.empty:
                row[res["label"]] = None
                continue
            ret_port = R.values @ w
            row[res["label"]] = float(np.prod(1 + ret_port) - 1)

        # SP500 en el escenario
        sp_esc = ret_sp500.loc[cfg["inicio"]:cfg["fin"]].dropna()
        row["S&P 500 (CLP)"] = float(np.prod(1 + sp_esc) - 1) if not sp_esc.empty else None

        # IPSA proxy
        ip_esc = ret_ipsa.loc[cfg["inicio"]:cfg["fin"]].dropna()
        row["IPSA (proxy)"] = float(np.prod(1 + ip_esc) - 1) if not ip_esc.empty else None

        rows.append(row)

    return pd.DataFrame(rows)


# ── Atribución de retorno ────────────────────────────────────────────────────

def atribucion_retorno(portafolio, retornos, meta):
    """
    Análisis de atribución de retorno (Brinson-Hood-Beebower simplificado).

    Descompone el retorno total del portafolio en:
    - Contribución por fondo: peso × retorno del fondo
    - Contribución por corredora
    - Contribución por perfil

    Retorna tres DataFrames: por_fondo, por_corredora, por_perfil
    """
    fondos = [f for f in portafolio["composicion"] if f in retornos.columns]
    if not fondos:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    pesos = {f: portafolio["composicion"][f] for f in fondos}
    total_peso = sum(pesos.values())
    pesos = {f: p / total_peso for f, p in pesos.items()}

    R = retornos[fondos].dropna()

    # Retorno y contribucion por fondo
    rows_fondo = []
    for f in fondos:
        ret_f = float(R[f].mean() * 12)
        peso  = pesos[f]
        rows_fondo.append({
            "fondo_id":    f,
            "nombre":      meta.loc[f, "nombre"] if f in meta.index else f,
            "corredora":   meta.loc[f, "corredora"] if f in meta.index else "?",
            "perfil":      meta.loc[f, "perfil"] if f in meta.index else "?",
            "peso":        peso,
            "ret_anual":   ret_f,
            "contribucion":peso * ret_f,   # contribucion al retorno total
        })

    df_fondo = (pd.DataFrame(rows_fondo)
                .sort_values("contribucion", ascending=False)
                .reset_index(drop=True))

    # Por corredora
    df_corr = (df_fondo.groupby("corredora")
               .agg(peso_total=("peso","sum"),
                    contribucion_total=("contribucion","sum"),
                    n_fondos=("fondo_id","count"))
               .reset_index()
               .sort_values("contribucion_total", ascending=False))

    # Por perfil
    df_perfil = (df_fondo.groupby("perfil")
                 .agg(peso_total=("peso","sum"),
                      contribucion_total=("contribucion","sum"),
                      n_fondos=("fondo_id","count"))
                 .reset_index()
                 .sort_values("contribucion_total", ascending=False))

    return df_fondo, df_corr, df_perfil
