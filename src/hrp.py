"""
src/hrp.py
----------
Hierarchical Risk Parity (HRP) — López de Prado (2016).

Publicado en: "Building Diversified Portfolios that Outperform Out-of-Sample"
Journal of Portfolio Management, 2016.

Motivación:
  Markowitz invierte la matriz de covarianza, amplificando errores de estimación.
  HRP construye el portafolio usando solo varianzas (no inversión matricial),
  apoyándose en el clustering jerárquico para estructurar la diversificación.

Algoritmo en 3 pasos:
  1. Clustering jerárquico sobre matriz de correlaciones (distancia Mantegna)
  2. Quasi-diagonalización: reordenar activos según el dendrograma
  3. Bisección recursiva: asignar capital inversamente proporcional a varianza

Ventajas sobre Markowitz:
  - No requiere invertir la matriz de covarianza
  - Naturalmente diversificado (no concentra en pocos activos)
  - Robusto a errores de estimación de retornos esperados
  - Funciona bien con universos grandes y pocos datos

Referencias:
  - López de Prado (2016): doi.org/10.3905/jpm.2016.42.4.059
  - Implementación basada en: mlfinlab (Hudson & Thames)
"""

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from src.metrics import sharpe_ratio, max_drawdown, TPM_ANUAL


# ── Paso 1: Matriz de distancia ──────────────────────────────────────────────

def _correlacion_a_distancia(corr):
    """
    Convierte correlación a distancia de Mantegna:
    d(i,j) = sqrt(0.5 * (1 - rho_ij))

    Satisface desigualdad triangular — válida para clustering jerárquico.
    """
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr.values), 0, 1))
    np.fill_diagonal(dist, 0)
    return pd.DataFrame(dist, index=corr.index, columns=corr.columns)


# ── Paso 2: Quasi-diagonalización ────────────────────────────────────────────

def _get_quasi_diag(linkage):
    """
    Extrae el orden de los activos según el dendrograma (quasi-diagonalización).
    Los activos similares quedan adyacentes — la covarianza resultante es
    aproximadamente diagonal por bloques.
    """
    linkage = linkage.astype(int)
    sort_ix = pd.Series([linkage[-1, 0], linkage[-1, 1]])
    num_items = linkage[-1, 3]

    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i   = df0.index
        j   = df0.values - num_items
        sort_ix[i] = linkage[j, 0]
        df0 = pd.Series(linkage[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])

    return sort_ix.tolist()


# ── Paso 3: Bisección recursiva ──────────────────────────────────────────────

def _cluster_var(cov, cluster_items):
    """
    Varianza del sub-portafolio de mínima varianza dentro de un cluster.
    Usa pesos inversamente proporcionales a la varianza individual.
    """
    cov_slice = cov.loc[cluster_items, cluster_items]
    w = _ivp_weights(cov_slice)
    var = float(w @ cov_slice.values @ w)
    return var


def _ivp_weights(cov):
    """
    Inverse-Variance Portfolio weights dentro de un cluster.
    w_i = (1/sigma_i^2) / sum(1/sigma_j^2)
    """
    ivp = 1.0 / np.diag(cov.values)
    ivp /= ivp.sum()
    return ivp


def _hrp_recursive(sort_ix, cov):
    """
    Bisección recursiva: divide el dendrograma en dos mitades,
    asigna capital proporcionalmente al inverso de la varianza de cada mitad.
    """
    w = pd.Series(1.0, index=sort_ix)
    cluster_items = [sort_ix]

    while len(cluster_items) > 0:
        cluster_items = [
            i[j:k]
            for i in cluster_items
            for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
            if len(i) > 1
        ]
        for i in range(0, len(cluster_items), 2):
            cluster_left  = cluster_items[i]
            cluster_right = cluster_items[i + 1]

            var_left  = _cluster_var(cov, cluster_left)
            var_right = _cluster_var(cov, cluster_right)

            alpha = 1.0 - var_left / (var_left + var_right)

            w[cluster_left]  *= alpha
            w[cluster_right] *= 1.0 - alpha

    return w


# ── Interfaz principal ───────────────────────────────────────────────────────

def hrp_portfolio(retornos, meta=None, perfil_filtro=None,
                   metodo_linkage="ward", min_meses=24):
    """
    Construye el portafolio HRP completo.

    Parámetros
    ----------
    retornos       : DataFrame wide fecha × fondo_id
    meta           : DataFrame metadata (opcional, para filtrar por perfil)
    perfil_filtro  : lista de perfiles a incluir (ej. ['agresivo','sp500'])
    metodo_linkage : 'ward' | 'complete' | 'average'

    Retorna
    -------
    dict con pesos, métricas y estructura jerárquica
    """
    # Filtrar fondos
    fondos = [f for f in retornos.columns if retornos[f].count() >= min_meses]
    if meta is not None and perfil_filtro:
        fondos = [f for f in fondos
                  if f in meta.index and meta.loc[f, "perfil"] in perfil_filtro]
    if len(fondos) < 2:
        return None

    R    = retornos[fondos].dropna()
    corr = R.corr()
    cov  = R.cov()

    # Paso 1: distancia y clustering
    dist         = _correlacion_a_distancia(corr)
    dist_cond    = squareform(dist.values)
    linkage_mat  = hierarchy.linkage(dist_cond, method=metodo_linkage)

    # Paso 2: quasi-diagonalización
    sort_ix = _get_quasi_diag(linkage_mat)
    sort_ix = corr.index[sort_ix].tolist()

    # Paso 3: bisección recursiva
    w = _hrp_recursive(sort_ix, cov.loc[sort_ix, sort_ix])
    w = w.reindex(fondos).fillna(0)
    w = w / w.sum()  # normalizar

    # Calcular métricas del portafolio
    ret_port = R.values @ w.values
    ret_a    = float(ret_port.mean() * 12)
    vol_a    = float(ret_port.std() * np.sqrt(12))
    sharpe   = (ret_a - TPM_ANUAL) / vol_a if vol_a > 0 else 0.0
    mdd      = max_drawdown(ret_port)

    composicion = {f: float(w[f]) for f in fondos if w[f] > 0.001}

    return {
        "perfil":        "hrp",
        "label":         "HRP",
        "color":         "#9b59b6",
        "fondos":        fondos,
        "pesos":         w.values.tolist(),
        "composicion":   composicion,
        "ret_anual":     ret_a,
        "vol_anual":     vol_a,
        "sharpe":        sharpe,
        "max_drawdown":  mdd,
        "n_activos":     int((w > 0.001).sum()),
        "linkage":       linkage_mat,
        "sort_ix":       sort_ix,
        "metodo":        "hrp_{}".format(metodo_linkage),
        "shrinkage":     None,
    }


def comparar_hrp_markowitz(hrp_res, markowitz_res, retornos):
    """
    Tabla comparativa HRP vs Markowitz por perfil.
    """
    rows = []
    for label, res in [("HRP", hrp_res), ("Markowitz (LW)", markowitz_res)]:
        if res is None:
            continue
        fondos = [f for f in res["composicion"] if f in retornos.columns]
        w      = np.array([res["composicion"][f] for f in fondos])
        w      = w / w.sum()
        R      = retornos[fondos].dropna()
        ret_p  = R.values @ w

        rows.append({
            "Metodo":          label,
            "Sharpe":          res["sharpe"],
            "Retorno anual":   res["ret_anual"],
            "Volatilidad":     res["vol_anual"],
            "Max Drawdown":    max_drawdown(ret_p),
            "N activos":       res["n_activos"],
            "Concentracion":   float(np.sum(w**2)),  # HHI: 1=concentrado, 1/n=diversificado
        })

    return pd.DataFrame(rows)
