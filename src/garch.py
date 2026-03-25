"""
src/garch.py
------------
Modelamiento de volatilidad condicional con GARCH(1,1).

Motivación:
  La volatilidad de los retornos financieros NO es constante en el tiempo.
  En 2022 (crisis inflación + TPM al alza), la volatilidad de los fondos
  chilenos fue 3-4x mayor que en 2024 (normalización). Ignorar esto
  subestima el riesgo en períodos de alta volatilidad.

Modelo GARCH(1,1) — Bollerslev (1986):
  r_t = mu + epsilon_t
  epsilon_t = sigma_t * z_t,  z_t ~ N(0,1)
  sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2

  donde:
    omega > 0 : varianza incondicional base
    alpha >= 0 : reacción a shocks pasados (ARCH effect)
    beta  >= 0 : persistencia de la volatilidad
    alpha + beta < 1 : condición de estacionariedad

Interpretación de parámetros:
  - alpha alto: volatilidad reacciona rápido a noticias (ej: COVID crash)
  - beta alto:  volatilidad persiste mucho tiempo (ej: crisis 2022)
  - alpha + beta cercano a 1: volatilidad muy persistente (mercados emergentes)

Referencias:
  - Bollerslev (1986): "Generalized Autoregressive Conditional Heteroskedasticity"
  - Engle (1982): "Autoregressive Conditional Heteroscedasticity" (Nobel 2003)
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False


# ── GARCH(1,1) con arch library ──────────────────────────────────────────────

def ajustar_garch(retornos_serie, p=1, q=1, dist="normal"):
    """
    Ajusta un modelo GARCH(p,q) a una serie de retornos.

    Parámetros
    ----------
    retornos_serie : Series de retornos mensuales
    p : orden ARCH (reacción a shocks)
    q : orden GARCH (persistencia)
    dist : 'normal' | 't' | 'skewt'

    Retorna
    -------
    dict con parámetros, volatilidad condicional y diagnósticos
    """
    if not ARCH_AVAILABLE:
        return _garch_manual(retornos_serie)

    r = retornos_serie.dropna() * 100  # escalar a % para estabilidad numérica

    modelo = arch_model(r, vol="Garch", p=p, q=q, dist=dist, rescale=False)
    try:
        resultado = modelo.fit(disp="off", show_warning=False)
    except Exception:
        return _garch_manual(retornos_serie)

    # Volatilidad condicional anualizada
    vol_cond = resultado.conditional_volatility / 100 * np.sqrt(12)

    params = resultado.params
    return {
        "omega":       float(params.get("omega", 0)),
        "alpha":       float(params.get("alpha[1]", 0)),
        "beta":        float(params.get("beta[1]", 0)),
        "persistencia":float(params.get("alpha[1]", 0) + params.get("beta[1]", 0)),
        "vol_incondicional": float(np.sqrt(params.get("omega", 0) /
                               max(1 - params.get("alpha[1]",0) - params.get("beta[1]",0), 1e-8))
                               / 100 * np.sqrt(12)),
        "vol_condicional":   pd.Series(vol_cond.values,
                                        index=retornos_serie.dropna().index,
                                        name="vol_garch"),
        "aic":               float(resultado.aic),
        "bic":               float(resultado.bic),
        "log_likelihood":    float(resultado.loglikelihood),
        "modelo":            resultado,
    }


def _garch_manual(retornos_serie):
    """
    Implementación manual de GARCH(1,1) por máxima verosimilitud
    cuando arch no está disponible.
    """
    from scipy.optimize import minimize

    r = retornos_serie.dropna().values
    n = len(r)
    r_demeaned = r - r.mean()

    def neg_log_likelihood(params):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(r_demeaned)
        for t in range(1, n):
            sigma2[t] = omega + alpha * r_demeaned[t-1]**2 + beta * sigma2[t-1]
        if np.any(sigma2 <= 0):
            return 1e10
        ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + r_demeaned**2 / sigma2)
        return -ll

    # Punto inicial
    var_init = np.var(r_demeaned)
    x0 = [var_init * 0.1, 0.1, 0.8]
    bounds = [(1e-8, None), (0, 0.99), (0, 0.99)]

    try:
        res = minimize(neg_log_likelihood, x0, method="L-BFGS-B", bounds=bounds)
        omega, alpha, beta = res.x
    except Exception:
        omega, alpha, beta = var_init * 0.1, 0.1, 0.8

    # Calcular volatilidad condicional
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(r_demeaned)
    for t in range(1, n):
        sigma2[t] = omega + alpha * r_demeaned[t-1]**2 + beta * sigma2[t-1]

    vol_cond = np.sqrt(sigma2) * np.sqrt(12)  # anualizada

    return {
        "omega":       float(omega),
        "alpha":       float(alpha),
        "beta":        float(beta),
        "persistencia":float(alpha + beta),
        "vol_incondicional": float(np.sqrt(omega / max(1 - alpha - beta, 1e-8)) * np.sqrt(12)),
        "vol_condicional":   pd.Series(vol_cond,
                                        index=retornos_serie.dropna().index,
                                        name="vol_garch"),
        "aic":               None,
        "bic":               None,
        "log_likelihood":    None,
        "modelo":            None,
    }


def garch_todos_perfiles(retornos, meta):
    """
    Ajusta GARCH(1,1) al retorno promedio de cada perfil.
    Compara volatilidad constante (histórica) vs condicional (GARCH).

    Retorna dict {perfil: resultado_garch}
    """
    resultados = {}
    for perfil in ["conservador", "moderado", "agresivo", "sp500"]:
        fondos = [f for f in retornos.columns
                  if f in meta.index and meta.loc[f, "perfil"] == perfil]
        if not fondos:
            continue
        ret_p = retornos[fondos].mean(axis=1).dropna()
        res   = ajustar_garch(ret_p)
        resultados[perfil] = res
        print("  {:12} alpha={:.3f} beta={:.3f} persist={:.3f} vol_incond={:.2%}".format(
            perfil,
            res["alpha"], res["beta"], res["persistencia"],
            res["vol_incondicional"]))
    return resultados


def vol_condicional_portafolio(retornos, pesos):
    """
    Calcula la volatilidad condicional GARCH del portafolio completo.
    """
    fondos   = [f for f in pesos if f in retornos.columns]
    w        = np.array([pesos[f] for f in fondos])
    w        = w / w.sum()
    R        = retornos[fondos].dropna()
    ret_port = pd.Series(R.values @ w, index=R.index)
    return ajustar_garch(ret_port)


def var_garch(retornos_serie, confianza=0.95, horizonte=1):
    """
    VaR dinámico usando volatilidad GARCH.
    Más preciso que el VaR histórico en períodos de alta volatilidad.

    VaR_t = mu + sigma_t * z_{1-alpha}
    donde z_{1-alpha} es el cuantil de la distribución normal.
    """
    from scipy.stats import norm

    res = ajustar_garch(retornos_serie)
    vol_t = res["vol_condicional"] / np.sqrt(12)  # de anual a mensual

    z = norm.ppf(1 - confianza)
    mu = retornos_serie.dropna().mean()
    var_dinamico = mu + vol_t * z  # negativo = pérdida

    return pd.Series(var_dinamico.values,
                      index=vol_t.index,
                      name="var_garch_{:.0f}%".format(confianza*100))
