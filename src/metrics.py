"""
src/metrics.py
--------------
Métricas financieras: Sharpe Ratio, VaR, CVaR, drawdown,
simulación histórica y estadísticas descriptivas.

Referencias:
- Sharpe (1966): "Mutual Fund Performance"
- Rockafellar & Uryasev (2000): CVaR optimization
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Tasa libre de riesgo: TPM promedio Chile 2020-2026
TPM_ANUAL   = 0.056
TPM_MENSUAL = TPM_ANUAL / 12
FECHA_INICIO = "2020-01-01"


# ── Métricas individuales ────────────────────────────────────────────────────

def sharpe_ratio(retornos, tpm_anual=TPM_ANUAL):
    """Sharpe Ratio anualizado. Retorna nan si volatilidad = 0."""
    r = np.asarray(retornos)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return np.nan
    exceso = r - tpm_anual / 12
    vol = exceso.std(ddof=1)
    return float(exceso.mean() / vol * np.sqrt(12)) if vol > 0 else np.nan


def sortino_ratio(retornos, tpm_anual=TPM_ANUAL):
    """
    Sortino Ratio: penaliza solo la volatilidad negativa.
    Más apropiado que Sharpe para distribuciones asimétricas.
    """
    r = np.asarray(retornos)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return np.nan
    tpm_m   = tpm_anual / 12
    exceso  = r - tpm_m
    ret_neg = exceso[exceso < 0]
    downside_vol = np.sqrt(np.mean(ret_neg**2)) * np.sqrt(12) if len(ret_neg) > 0 else np.nan
    ret_anual    = exceso.mean() * 12
    return float(ret_anual / downside_vol) if downside_vol and downside_vol > 0 else np.nan


def var_historico(retornos, confianza=0.95):
    """
    Value at Risk histórico al nivel de confianza dado.
    Retorna pérdida máxima esperada (número negativo).
    Ej: VaR 95% = -0.05 significa que el 95% del tiempo la pérdida <= 5%.
    """
    r = np.asarray(retornos)
    r = r[~np.isnan(r)]
    return float(np.percentile(r, (1 - confianza) * 100))


def cvar_historico(retornos, confianza=0.95):
    """
    Conditional Value at Risk (Expected Shortfall).
    Media de las pérdidas que superan el VaR — más conservador que VaR.
    """
    r = np.asarray(retornos)
    r = r[~np.isnan(r)]
    var = var_historico(r, confianza)
    return float(r[r <= var].mean()) if (r <= var).sum() > 0 else np.nan


def max_drawdown(retornos):
    """Máxima caída desde el pico histórico (en retorno acumulado)."""
    r   = np.asarray(retornos)
    r   = r[~np.isnan(r)]
    cum = np.cumprod(1 + r)
    rolling_max = np.maximum.accumulate(cum)
    dd  = (cum - rolling_max) / rolling_max
    return float(dd.min())


def calmar_ratio(retornos, tpm_anual=TPM_ANUAL):
    """Calmar Ratio = Retorno anual / |Max Drawdown|. Mide retorno por unidad de drawdown."""
    r = np.asarray(retornos)
    r = r[~np.isnan(r)]
    ret_anual = r.mean() * 12
    mdd = abs(max_drawdown(r))
    return float(ret_anual / mdd) if mdd > 0 else np.nan


def omega_ratio(retornos, threshold=0.0):
    """
    Omega Ratio: razón entre ganancias y pérdidas relativas al threshold.
    Omega > 1 significa más masa en ganancias que en pérdidas.
    """
    r = np.asarray(retornos)
    r = r[~np.isnan(r)]
    ganancias = (r[r > threshold] - threshold).sum()
    perdidas  = (threshold - r[r < threshold]).sum()
    return float(ganancias / perdidas) if perdidas > 0 else np.nan


# ── Tabla de estadísticas ────────────────────────────────────────────────────

def tabla_stats(retornos, meta, tpm_anual=TPM_ANUAL):
    """
    Calcula métricas completas para todos los fondos.
    Incluye Sharpe, Sortino, VaR, CVaR, Calmar y Omega ratios.
    """
    rows = []
    for fid in retornos.columns:
        r = retornos[fid].dropna().values
        if len(r) < 12:
            continue
        ret_a = float(r.mean() * 12)
        vol_a = float(r.std(ddof=1) * np.sqrt(12))
        rows.append({
            "fondo_id":          fid,
            "nombre":            meta.loc[fid, "nombre"] if fid in meta.index else fid,
            "perfil":            meta.loc[fid, "perfil"] if fid in meta.index else "?",
            "corredora":         meta.loc[fid, "corredora"] if fid in meta.index else "?",
            "moneda_orig":       meta.loc[fid, "moneda_orig"] if fid in meta.index else "CLP",
            "retorno_anual":     ret_a,
            "volatilidad_anual": vol_a,
            "sharpe":            sharpe_ratio(r, tpm_anual),
            "sortino":           sortino_ratio(r, tpm_anual),
            "var_95":            var_historico(r, 0.95),
            "cvar_95":           cvar_historico(r, 0.95),
            "max_drawdown":      max_drawdown(r),
            "calmar":            calmar_ratio(r, tpm_anual),
            "omega":             omega_ratio(r),
            "retorno_acumulado": float(np.prod(1 + r) - 1),
            "skewness":          float(scipy_stats.skew(r)),
            "kurtosis":          float(scipy_stats.kurtosis(r)),
            "meses_positivos":   int((r > 0).sum()),
            "n_meses":           len(r),
        })
    return (pd.DataFrame(rows)
            .sort_values("sharpe", ascending=False)
            .reset_index(drop=True))


# ── Drawdown serie ───────────────────────────────────────────────────────────

def drawdown_serie(serie):
    """Retorna Serie de drawdown como % desde el máximo histórico."""
    rolling_max = serie.expanding().max()
    return (serie - rolling_max) / rolling_max * 100


# ── Simulación histórica ─────────────────────────────────────────────────────

def simular_historico(retornos, pesos, monto_inicial, aporte_mensual=0):
    """
    Simula el crecimiento histórico de un portafolio con aportes periódicos.

    Parámetros
    ----------
    retornos       : DataFrame wide fecha × fondo_id
    pesos          : dict {fondo_id: peso} (deben sumar 1)
    monto_inicial  : monto inicial en CLP
    aporte_mensual : aporte mensual adicional en CLP (default 0)

    Retorna Serie con valor de la cartera en cada fecha.
    """
    fondos   = list(pesos.keys())
    w        = np.array([pesos[f] for f in fondos])
    R        = retornos[fondos].dropna()
    cartera  = monto_inicial
    valores  = []
    for _, row in R.iterrows():
        ret_port = float(row.values @ w)
        cartera  = cartera * (1 + ret_port) + aporte_mensual
        valores.append(cartera)
    return pd.Series(valores, index=R.index, name="cartera")


# ── Estacionalidad ───────────────────────────────────────────────────────────

def estacionalidad_por_mes(retornos, meta):
    """Retorno promedio mensual por perfil y mes del año."""
    meses_nombres = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",
                     7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}
    resultados = {}
    for perfil in ["conservador", "moderado", "agresivo"]:
        fondos_p = [f for f in retornos.columns
                    if f in meta.index and meta.loc[f, "perfil"] == perfil]
        if not fondos_p:
            continue
        ret_p = retornos[fondos_p].mean(axis=1)
        por_mes = {}
        for m in range(1, 13):
            vals = ret_p[ret_p.index.month == m].dropna()
            por_mes[meses_nombres[m]] = float(vals.mean() * 100) if len(vals) > 0 else 0.0
        resultados[perfil] = por_mes
    return pd.DataFrame(resultados)
