"""
src/optimizer.py
----------------
Optimización de portafolios con 4 perfiles:
  conservador | moderado | agresivo | sp500 (puro) + optimo global

El SP500 compite directamente con los fondos chilenos en los perfiles
moderado, agresivo y optimo. El perfil 'sp500' es 100% SP500.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.covariance import get_cov_matrix
from src.metrics import TPM_ANUAL

PERFILES = {
    "conservador": {
        "label": "Conservador", "color": "#2ecc71",
        "vol_max": 0.06,
        "tipos":   ["conservador"],           # excluye SP500 (alta vol)
    },
    "moderado": {
        "label": "Moderado", "color": "#f1c40f",
        "vol_max": 0.12,
        "tipos":   ["conservador", "moderado"],  # excluye SP500 (restriccion vol)
    },
    "agresivo": {
        "label": "Agresivo", "color": "#e74c3c",
        "vol_max": 0.99,
        "tipos":   ["moderado", "agresivo", "sp500"],  # SP500 compite aqui
    },
    "sp500": {
        "label": "S&P 500", "color": "#00b4d8",
        "vol_max": 0.99,
        "tipos":   ["sp500"],                 # solo SP500 puro
    },
}


def _filtrar_fondos(retornos, meta, tipos, min_meses=24):
    return [f for f in retornos.columns
            if f in meta.index
            and meta.loc[f, "perfil"] in tipos
            and retornos[f].count() >= min_meses]


def _metricas(w, mu, Sigma, tpm_anual=TPM_ANUAL):
    ret = float(w @ mu * 12)
    vol = float(np.sqrt(w @ Sigma @ w * 12))
    sharpe = (ret - tpm_anual) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def _construir_resultado(fondos, w, mu, Sigma, perfil, tpm_anual,
                          metodo, shrinkage, peso_min):
    ret_p, vol_p, sharpe_p = _metricas(w, mu, Sigma, tpm_anual)
    cfg = PERFILES.get(perfil, {"label": perfil, "color": "#888"})
    composicion = {fondos[i]: float(w[i])
                   for i in range(len(fondos)) if w[i] > peso_min * 0.5}
    return {
        "perfil":      perfil,
        "label":       cfg["label"],
        "color":       cfg["color"],
        "fondos":      fondos,
        "pesos":       w.tolist(),
        "composicion": composicion,
        "ret_anual":   ret_p,
        "vol_anual":   vol_p,
        "sharpe":      sharpe_p,
        "n_activos":   len(composicion),
        "metodo":      metodo,
        "shrinkage":   shrinkage,
    }


def portafolio_sp500_puro(retornos, meta, tpm_anual=TPM_ANUAL):
    """Portafolio 100% SP500. Calcula sus métricas directamente."""
    if "SP500" not in retornos.columns:
        return None
    r   = retornos["SP500"].dropna()
    ret = float(r.mean() * 12)
    vol = float(r.std() * np.sqrt(12))
    cfg = PERFILES["sp500"]
    return {
        "perfil":      "sp500",
        "label":       cfg["label"],
        "color":       cfg["color"],
        "fondos":      ["SP500"],
        "pesos":       [1.0],
        "composicion": {"SP500": 1.0},
        "ret_anual":   ret,
        "vol_anual":   vol,
        "sharpe":      (ret - tpm_anual) / vol if vol > 0 else 0.0,
        "n_activos":   1,
        "metodo":      "puro",
        "shrinkage":   None,
    }


def optimizar(retornos, meta, perfil, peso_min=0.02, peso_max=0.40,
              n_intentos=50, tpm_anual=TPM_ANUAL, cov_metodo="ledoit_wolf"):
    """Portafolio de máximo Sharpe con Ledoit-Wolf para un perfil dado."""

    # SP500 puro — no necesita optimización
    if perfil == "sp500":
        return portafolio_sp500_puro(retornos, meta, tpm_anual)

    cfg    = PERFILES[perfil]
    fondos = _filtrar_fondos(retornos, meta, cfg["tipos"])
    if len(fondos) < 2:
        return None

    R      = retornos[fondos].dropna()
    mu     = R.mean().values
    Sigma, shrinkage = get_cov_matrix(R, metodo=cov_metodo)
    n      = len(fondos)

    def neg_sharpe(w):
        _, _, sh = _metricas(w, mu, Sigma, tpm_anual)
        return -sh

    def vol_anual(w):
        return float(np.sqrt(w @ Sigma @ w * 12))

    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0}]
    if cfg["vol_max"] < 0.5:
        constraints.append({"type": "ineq",
                             "fun": lambda w, v=cfg["vol_max"]: v - vol_anual(w)})

    bounds = [(peso_min, peso_max)] * n
    rng    = np.random.default_rng(42)
    mejor  = None

    for _ in range(n_intentos):
        w0  = rng.dirichlet(np.ones(n))
        res = minimize(neg_sharpe, w0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"maxiter": 2000, "ftol": 1e-12})
        if res.success and (mejor is None or res.fun < mejor.fun):
            mejor = res

    if mejor is None:
        return None

    return _construir_resultado(
        fondos, mejor.x, mu, Sigma, perfil,
        tpm_anual, "markowitz_lw", shrinkage, peso_min)


def optimizar_robusto(retornos, meta, perfil, peso_min=0.02, peso_max=0.40,
                       n_intentos=50, tpm_anual=TPM_ANUAL,
                       epsilon=0.1, cov_metodo="ledoit_wolf"):
    """Optimización robusta min-max Sharpe."""
    if perfil == "sp500":
        return None  # SP500 puro no necesita optimizacion robusta

    cfg    = PERFILES[perfil]
    fondos = _filtrar_fondos(retornos, meta, cfg["tipos"])
    if len(fondos) < 2:
        return None

    R      = retornos[fondos].dropna()
    mu     = R.mean().values
    Sigma, shrinkage = get_cov_matrix(R, metodo=cov_metodo)
    Sigma_worst = Sigma + epsilon * np.diag(np.diag(Sigma))
    n      = len(fondos)

    def neg_sharpe_robusto(w):
        ret = float(w @ mu * 12)
        vol = float(np.sqrt(w @ Sigma_worst @ w * 12))
        return -(ret - tpm_anual) / vol if vol > 0 else 0.0

    def vol_worst(w):
        return float(np.sqrt(w @ Sigma_worst @ w * 12))

    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0}]
    if cfg["vol_max"] < 0.5:
        constraints.append({"type": "ineq",
                             "fun": lambda w, v=cfg["vol_max"]: v - vol_worst(w)})

    bounds = [(peso_min, peso_max)] * n
    rng    = np.random.default_rng(42)
    mejor  = None

    for _ in range(n_intentos):
        w0  = rng.dirichlet(np.ones(n))
        res = minimize(neg_sharpe_robusto, w0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"maxiter": 2000, "ftol": 1e-12})
        if res.success and (mejor is None or res.fun < mejor.fun):
            mejor = res

    if mejor is None:
        return None

    res_dict = _construir_resultado(
        fondos, mejor.x, mu, Sigma, perfil,
        tpm_anual, "robusto", shrinkage, peso_min)
    res_dict["label"] = cfg["label"] + " (Robusto)"
    res_dict["epsilon"] = epsilon
    return res_dict


def optimizar_global(retornos, meta, peso_min=0.0, peso_max=0.49,
                      min_fondos=3, n_intentos=80, tpm_anual=TPM_ANUAL,
                      cov_metodo="ledoit_wolf"):
    """
    Máximo Sharpe sin restricción de perfil.
    El SP500 compite con pesos 0–49% junto a los fondos chilenos.
    """
    # Todos los fondos con suficientes datos (incluyendo SP500)
    fondos = [f for f in retornos.columns if retornos[f].count() >= 24]
    R      = retornos[fondos].dropna()
    mu     = R.mean().values
    Sigma, shrinkage = get_cov_matrix(R, metodo=cov_metodo)
    n      = len(fondos)

    def neg_sharpe(w):
        _, _, sh = _metricas(w, mu, Sigma, tpm_anual)
        return -sh

    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0}]
    bounds = [(peso_min, peso_max)] * n
    rng    = np.random.default_rng(42)
    mejor  = None

    for _ in range(n_intentos):
        w0  = rng.dirichlet(np.ones(n))
        res = minimize(neg_sharpe, w0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"maxiter": 2000, "ftol": 1e-12})
        if res.success and (mejor is None or res.fun < mejor.fun):
            if (res.x > 0.01).sum() >= min_fondos:
                mejor = res

    if mejor is None:
        return None

    return {
        "perfil":      "optimo",
        "label":       "Optimo Global",
        "color":       "#FFD700",
        "fondos":      fondos,
        "pesos":       mejor.x.tolist(),
        "composicion": {fondos[i]: float(mejor.x[i])
                        for i in range(n) if mejor.x[i] > 0.01},
        "ret_anual":   float(mejor.x @ mu * 12),
        "vol_anual":   float(np.sqrt(mejor.x @ Sigma @ mejor.x * 12)),
        "sharpe":      float(-mejor.fun),
        "n_activos":   int((mejor.x > 0.01).sum()),
        "metodo":      "global_lw",
        "shrinkage":   shrinkage,
    }


def optimizar_todos(retornos, meta, robusto=True, **kwargs):
    """
    Optimiza los 4 perfiles + óptimo global.
    Perfiles: conservador, moderado, agresivo, sp500, optimo.
    """
    resultados = {}
    for perfil in ["conservador", "moderado", "agresivo", "sp500"]:
        label = PERFILES[perfil]["label"]
        print("  Optimizando {}...".format(label))
        res = optimizar(retornos, meta, perfil, **kwargs)
        if res:
            resultados[perfil] = res
            print("    -> Sharpe {:.3f} | Ret {:.2%} | Vol {:.2%}".format(
                res["sharpe"], res["ret_anual"], res["vol_anual"]))

        if robusto and perfil != "sp500":
            res_r = optimizar_robusto(retornos, meta, perfil, **kwargs)
            if res_r:
                resultados[perfil + "_robusto"] = res_r

    print("  Optimizando global (con SP500)...")
    opt = optimizar_global(retornos, meta, **kwargs)
    if opt:
        resultados["optimo"] = opt
        sp500_peso = opt["composicion"].get("SP500", 0)
        print("    -> Sharpe {:.3f} | Ret {:.2%} | Vol {:.2%} | SP500: {:.1%}".format(
            opt["sharpe"], opt["ret_anual"], opt["vol_anual"], sp500_peso))

    return resultados


def frontera_eficiente(retornos, fondos=None, n_puntos=60,
                        cov_metodo="ledoit_wolf"):
    if fondos is None:
        fondos = list(retornos.columns)
    R      = retornos[fondos].dropna()
    mu     = R.mean().values
    Sigma, _ = get_cov_matrix(R, metodo=cov_metodo)
    n      = len(fondos)
    puntos = []

    for target in np.linspace(mu.min() * 12, mu.max() * 12, n_puntos):
        cons = [
            {"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: float(w @ mu * 12) - t},
        ]
        res = minimize(lambda w: float(w @ Sigma @ w),
                       np.ones(n) / n, method="SLSQP",
                       bounds=[(0, 1)] * n, constraints=cons,
                       options={"maxiter": 500})
        if res.success:
            vol = float(np.sqrt(res.fun * 12))
            puntos.append({
                "ret_anual": target, "vol_anual": vol,
                "sharpe":    (target - TPM_ANUAL) / vol if vol > 0 else np.nan,
            })
    return pd.DataFrame(puntos)
