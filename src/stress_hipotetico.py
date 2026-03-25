"""
src/stress_hipotetico.py
------------------------
Stress testing hipotético: escenarios que NO ocurrieron pero PODRÍAN ocurrir.

A diferencia del stress test histórico (que usa períodos reales del pasado),
el stress test hipotético define shocks específicos y simula su impacto
en el portafolio actual.

Tipos de shocks implementados:
  1. Shock de mercado: caída del SP500 de X%
  2. Shock de tasa: subida/bajada de la TPM de X puntos base
  3. Shock de tipo de cambio: depreciación/apreciación del CLP
  4. Shock combinado: múltiples factores simultáneos (crisis sistémica)
  5. Shock de correlación: todas las correlaciones suben a 0.9 (flight to quality)

Metodología:
  Para un shock en el factor k de magnitud delta_k:
  delta_r_portafolio = sum_i (w_i * beta_ik * delta_k)

  donde beta_ik es la sensibilidad del fondo i al factor k,
  estimada por regresión histórica.

Referencias:
  - Basel Committee (2009): "Principles for sound stress testing"
  - FSOC (2012): Annual Report stress testing framework
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from src.metrics import TPM_ANUAL


# ── Sensibilidades (betas) de cada fondo ─────────────────────────────────────

def calcular_sensibilidades(retornos, meta):
    """
    Estima la sensibilidad de cada fondo a:
    1. Retorno del mercado (beta vs SP500)
    2. Nivel de tasas (proxy: diferencia de retornos renta fija)
    3. Tipo de cambio (para fondos con exposición USD)

    Retorna DataFrame con sensibilidades por fondo.
    """
    sensibilidades = {}

    # Factor 1: mercado (SP500 si disponible, sino promedio agresivo)
    if "SP500" in retornos.columns:
        factor_mercado = retornos["SP500"].dropna()
    else:
        fondos_agr = [f for f in retornos.columns
                      if f in meta.index and meta.loc[f, "perfil"] == "agresivo"]
        factor_mercado = retornos[fondos_agr].mean(axis=1).dropna() if fondos_agr else None

    # Factor 2: tasa (proxy: retorno promedio fondos conservadores)
    fondos_cons = [f for f in retornos.columns
                   if f in meta.index and meta.loc[f, "perfil"] == "conservador"]
    factor_tasa = retornos[fondos_cons].mean(axis=1).dropna() if fondos_cons else None

    for fid in retornos.columns:
        r = retornos[fid].dropna()
        sens = {"fondo_id": fid}

        if factor_mercado is not None:
            df = pd.DataFrame({"r": r, "m": factor_mercado}).dropna()
            if len(df) >= 12:
                slope, _, _, _, _ = scipy_stats.linregress(df["m"], df["r"])
                sens["beta_mercado"] = float(slope)
            else:
                sens["beta_mercado"] = 0.0

        if factor_tasa is not None:
            df = pd.DataFrame({"r": r, "t": factor_tasa}).dropna()
            if len(df) >= 12:
                slope, _, _, _, _ = scipy_stats.linregress(df["t"], df["r"])
                sens["beta_tasa"] = float(slope)
            else:
                sens["beta_tasa"] = 0.0

        # USD exposure: fondos con moneda_orig = USD tienen beta_usd ≈ 1
        if fid in meta.index:
            sens["beta_usd"] = 1.0 if meta.loc[fid, "moneda_orig"] == "USD" else 0.0
        else:
            sens["beta_usd"] = 0.0

        sensibilidades[fid] = sens

    return pd.DataFrame(sensibilidades).T.set_index("fondo_id").astype(float)


# ── Escenarios hipotéticos ────────────────────────────────────────────────────

ESCENARIOS_HIPOTETICOS = {
    "Crash SP500 -20%": {
        "descripcion": "Caída abrupta del 20% en SP500 (similar a COVID pero más severo)",
        "shock_mercado": -0.20,
        "shock_tasa":     0.00,
        "shock_usd":      0.10,  # USD se aprecia en crisis
    },
    "Crash SP500 -40%": {
        "descripcion": "Crash severo tipo 2008-2009",
        "shock_mercado": -0.40,
        "shock_tasa":     0.00,
        "shock_usd":      0.15,
    },
    "Subida TPM +300 bps": {
        "descripcion": "Banco Central sube TPM 3pp de golpe (shock inflacionario extremo)",
        "shock_mercado": -0.05,
        "shock_tasa":    -0.08,   # renta fija cae con subida de tasas
        "shock_usd":      0.05,
    },
    "Depreciacion CLP -20%": {
        "descripcion": "CLP se deprecia 20% (crisis política o externa)",
        "shock_mercado": -0.10,
        "shock_tasa":     0.02,
        "shock_usd":      0.20,   # fondos USD se benefician
    },
    "Crisis sistemica": {
        "descripcion": "Combinación: SP500 -30%, TPM +200bps, CLP -15%",
        "shock_mercado": -0.30,
        "shock_tasa":    -0.06,
        "shock_usd":      0.15,
    },
    "Rally SP500 +30%": {
        "descripcion": "Escenario optimista: SP500 sube 30% en un año",
        "shock_mercado":  0.30,
        "shock_tasa":     0.01,
        "shock_usd":     -0.05,
    },
    "Shock correlacion": {
        "descripcion": "Todas las correlaciones suben a 0.9 (flight to quality)",
        "shock_mercado": -0.15,
        "shock_tasa":    -0.03,
        "shock_usd":      0.08,
        "shock_correlacion": True,
    },
}


def stress_hipotetico(portafolios, retornos, meta, monto=10_000_000):
    """
    Aplica escenarios hipotéticos a cada portafolio.

    Para cada escenario, el impacto en el portafolio es:
    P&L = sum_i (w_i * beta_i * shock_factor)

    Retorna DataFrame con impacto en % y en CLP por escenario y portafolio.
    """
    sensibilidades = calcular_sensibilidades(retornos, meta)

    port_sel = {k: v for k, v in portafolios.items()
                if k in ["conservador", "moderado", "agresivo", "sp500", "optimo"]
                and k in portafolios}

    rows = []
    for escenario, cfg in ESCENARIOS_HIPOTETICOS.items():
        row = {"Escenario": escenario, "Descripcion": cfg["descripcion"]}

        for p_id, res in port_sel.items():
            fondos = [f for f in res["composicion"] if f in retornos.columns]
            if not fondos:
                continue
            pesos  = np.array([res["composicion"][f] for f in fondos])
            pesos  = pesos / pesos.sum()

            # Impacto ponderado por sensibilidades
            impacto_total = 0.0
            for i, fid in enumerate(fondos):
                if fid not in sensibilidades.index:
                    continue
                sens = sensibilidades.loc[fid]
                impacto_fondo = (
                    sens.get("beta_mercado", 0) * cfg.get("shock_mercado", 0) +
                    sens.get("beta_tasa",    0) * cfg.get("shock_tasa",    0) +
                    sens.get("beta_usd",     0) * cfg.get("shock_usd",     0)
                )
                impacto_total += pesos[i] * impacto_fondo

            row[res["label"]] = impacto_total

        rows.append(row)

    df = pd.DataFrame(rows)

    # Agregar impacto en CLP
    for col in df.columns:
        if col in ["Escenario", "Descripcion"]:
            continue
        df[col + " ($CLP)"] = df[col] * monto

    return df


def tabla_stress_comparativo(df_historico, df_hipotetico):
    """
    Combina stress test histórico e hipotético en una tabla unificada.
    """
    cols_port = [c for c in df_historico.columns
                 if c not in ["Escenario", "Descripcion"]]

    df_hist = df_historico[["Escenario"] + cols_port].copy()
    df_hist["Tipo"] = "Histórico"

    df_hip = df_hipotetico[["Escenario"] + [c for c in cols_port
                                              if c in df_hipotetico.columns]].copy()
    df_hip["Tipo"] = "Hipotético"

    return pd.concat([df_hist, df_hip], ignore_index=True)
