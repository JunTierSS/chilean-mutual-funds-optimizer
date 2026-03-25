"""
src/bootstrap.py
----------------
Simulación de Monte Carlo por Bootstrap de Bloques (Block Bootstrap).

Problema con Monte Carlo clásico (distribución normal multivariada):
- Asume que los retornos son i.i.d. (independientes e idénticamente distribuidos)
- Ignora la autocorrelación temporal de los retornos
- Subestima eventos extremos (colas pesadas)

Solución — Block Bootstrap:
En vez de simular desde una distribución paramétrica, remuestrea bloques
consecutivos del historial real. Esto preserva:
- La autocorrelación temporal dentro de cada bloque
- Los eventos extremos reales (COVID, crisis 2022)
- La correlación entre fondos en cada período

Dos variantes:
1. Fixed Block Bootstrap (Kunsch 1989): bloques de longitud fija
2. Stationary Bootstrap (Politis & Romano 1994): longitud de bloque aleatoria

Referencias:
- Kunsch (1989): "The jackknife and the bootstrap for general stationary observations"
- Politis & Romano (1994): "The stationary bootstrap"
- Efron & Tibshirani (1993): "An Introduction to the Bootstrap"
"""

import numpy as np
import pandas as pd


def _block_bootstrap_indices(n_obs, block_size, n_sim_obs, rng):
    """
    Genera índices para block bootstrap.
    Muestrea bloques de longitud block_size hasta completar n_sim_obs.
    """
    indices = []
    while len(indices) < n_sim_obs:
        start = rng.integers(0, max(1, n_obs - block_size + 1))
        bloque = list(range(start, min(start + block_size, n_obs)))
        indices.extend(bloque)
    return indices[:n_sim_obs]


def proyectar_bootstrap(retornos, pesos, monto_inicial, n_meses,
                         n_sim=1000, block_size=6, seed=42,
                         aporte_mensual=0, crecimiento_anual=0.0):
    """
    Monte Carlo por Bootstrap de Bloques.

    Parámetros
    ----------
    retornos         : DataFrame wide fecha × fondo_id
    pesos            : dict {fondo_id: peso}
    monto_inicial    : CLP
    n_meses          : horizonte de proyección
    n_sim            : número de simulaciones
    block_size       : longitud del bloque en meses (default 6 = semestral)
                       Regla práctica: sqrt(n_obs) para datos mensuales
    aporte_mensual   : aporte adicional mensual en t=0 (default 0)
    crecimiento_anual: tasa de crecimiento anual del aporte (default 0.0)
                       Ej: 0.03 = el aporte crece 3% cada año

    Retorna
    -------
    dict con fechas, percentiles y estadísticas finales
    """
    rng    = np.random.default_rng(seed)
    fondos = [f for f in pesos if f in retornos.columns]
    w      = np.array([pesos[f] for f in fondos])
    R      = retornos[fondos].dropna()
    n_obs  = len(R)

    # Tasa de crecimiento mensual equivalente
    tasa_m = (1 + crecimiento_anual) ** (1 / 12) - 1

    sims = np.zeros((n_sim, n_meses))

    for i in range(n_sim):
        # Remuestrear bloques del historial
        indices = _block_bootstrap_indices(n_obs, block_size, n_meses, rng)
        R_boot  = R.iloc[indices].values  # shape: (n_meses, n_fondos)

        # Retorno del portafolio en cada período simulado
        ret_port = R_boot @ w

        # Acumular valor de la cartera (aporte crece con el tiempo)
        cartera = monto_inicial
        for t in range(n_meses):
            aporte_t = aporte_mensual * (1 + tasa_m) ** t
            cartera  = cartera * (1 + ret_port[t]) + aporte_t
            sims[i, t] = cartera

    fechas = pd.date_range(
        start=R.index[-1] + pd.DateOffset(months=1),
        periods=n_meses, freq="MS"
    )
    finales = sims[:, -1]

    return {
        "metodo":       "block_bootstrap",
        "block_size":   block_size,
        "n_sim":        n_sim,
        "fechas":       fechas,
        "simulaciones": sims,
        "p5":    np.percentile(sims, 5,  axis=0),
        "p25":   np.percentile(sims, 25, axis=0),
        "p50":   np.percentile(sims, 50, axis=0),
        "p75":   np.percentile(sims, 75, axis=0),
        "p95":   np.percentile(sims, 95, axis=0),
        "minimo":   float(finales.min()),
        "maximo":   float(finales.max()),
        "promedio": float(finales.mean()),
        "mediana":  float(np.median(finales)),
        "std":      float(finales.std()),
    }


def proyectar_montecarlo_normal(retornos, pesos, monto_inicial, n_meses,
                                 n_sim=1000, seed=42, aporte_mensual=0,
                                 crecimiento_anual=0.0):
    """
    Monte Carlo clásico con distribución normal multivariada.
    Incluido para comparación directa con el bootstrap.
    """
    rng    = np.random.default_rng(seed)
    fondos = [f for f in pesos if f in retornos.columns]
    w      = np.array([pesos[f] for f in fondos])
    R      = retornos[fondos].dropna()
    mu     = R.mean().values
    sigma  = R.cov().values

    tasa_m = (1 + crecimiento_anual) ** (1 / 12) - 1

    sims = np.zeros((n_sim, n_meses))
    for i in range(n_sim):
        r_sim   = rng.multivariate_normal(mu, sigma, n_meses)
        cartera = monto_inicial
        for t in range(n_meses):
            aporte_t = aporte_mensual * (1 + tasa_m) ** t
            cartera  = cartera * (1 + float(r_sim[t] @ w)) + aporte_t
            sims[i, t] = cartera

    fechas  = pd.date_range(
        start=R.index[-1] + pd.DateOffset(months=1),
        periods=n_meses, freq="MS"
    )
    finales = sims[:, -1]

    return {
        "metodo":       "montecarlo_normal",
        "block_size":   None,
        "n_sim":        n_sim,
        "fechas":       fechas,
        "simulaciones": sims,
        "p5":    np.percentile(sims, 5,  axis=0),
        "p25":   np.percentile(sims, 25, axis=0),
        "p50":   np.percentile(sims, 50, axis=0),
        "p75":   np.percentile(sims, 75, axis=0),
        "p95":   np.percentile(sims, 95, axis=0),
        "minimo":   float(finales.min()),
        "maximo":   float(finales.max()),
        "promedio": float(finales.mean()),
        "mediana":  float(np.median(finales)),
        "std":      float(finales.std()),
    }


def comparar_metodos(retornos, pesos, monto_inicial, n_meses,
                      n_sim=1000, block_size=6):
    """
    Compara Bootstrap vs Monte Carlo Normal.
    Retorna DataFrame con métricas de ambos métodos.
    """
    bs   = proyectar_bootstrap(retornos, pesos, monto_inicial, n_meses,
                                n_sim=n_sim, block_size=block_size)
    mc   = proyectar_montecarlo_normal(retornos, pesos, monto_inicial, n_meses,
                                        n_sim=n_sim)
    rows = []
    for nombre, res in [("Bootstrap (bloques)", bs), ("Monte Carlo Normal", mc)]:
        rows.append({
            "Método":   nombre,
            "Mínimo":   "${:,.0f}".format(res["minimo"]),
            "P5":       "${:,.0f}".format(res["p5"][-1]),
            "Mediana":  "${:,.0f}".format(res["mediana"]),
            "P95":      "${:,.0f}".format(res["p95"][-1]),
            "Máximo":   "${:,.0f}".format(res["maximo"]),
            "Std":      "${:,.0f}".format(res["std"]),
        })
    return pd.DataFrame(rows), bs, mc
