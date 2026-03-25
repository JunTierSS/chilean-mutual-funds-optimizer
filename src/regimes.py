"""
src/regimes.py
--------------
Detección de regímenes de mercado con Hidden Markov Models (HMM).

Un HMM asume que el mercado alterna entre estados ocultos (regímenes) con
distintas distribuciones de retornos. El modelo aprende:
- Cuántos regímenes existen (selección automática por BIC)
- La distribución de retornos en cada régimen
- La probabilidad de transición entre regímenes

Estados típicos en mercados financieros:
- Régimen 1 (Bull): retornos altos, volatilidad moderada
- Régimen 2 (Bear): retornos negativos, volatilidad alta
- Régimen 3 (Lateral): retornos bajos, volatilidad baja

Referencias:
- Hamilton (1989): "A New Approach to the Economic Analysis of Nonstationary
  Time Series and the Business Cycle"
- hmmlearn: Gaussian HMM implementation
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

try:
    from hmmlearn import hmm as hmmlearn_hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False


def _retorno_mercado(retornos, meta, perfil="agresivo"):
    """
    Construye una serie de retorno de mercado promediando los fondos de un perfil.
    Usada como señal de entrada al HMM.
    """
    fondos = [f for f in retornos.columns
              if f in meta.index and meta.loc[f, "perfil"] == perfil]
    if not fondos:
        fondos = list(retornos.columns)
    return retornos[fondos].mean(axis=1).dropna()


def seleccionar_n_regimenes(retornos, meta, max_regimenes=5, n_iter=200):
    """
    Selecciona el número óptimo de regímenes usando BIC (Bayesian Information Criterion).
    BIC penaliza la complejidad del modelo — menor BIC = mejor balance ajuste/parsimonia.

    Retorna dict {n_regimenes: bic_score} y el n óptimo.
    """
    if not HMM_AVAILABLE:
        return {2: 0, 3: 0}, 3

    serie = _retorno_mercado(retornos, meta)
    X = serie.values.reshape(-1, 1)

    bic_scores = {}
    for n in range(2, max_regimenes + 1):
        try:
            modelo = hmmlearn_hmm.GaussianHMM(
                n_components=n, covariance_type="full",
                n_iter=n_iter, random_state=42
            )
            modelo.fit(X)
            log_likelihood = modelo.score(X)
            # BIC = -2 * log_L + k * log(n_obs)
            # k = parámetros del modelo
            k = n * n + 2 * n  # transiciones + medias + varianzas
            bic = -2 * log_likelihood * len(X) + k * np.log(len(X))
            bic_scores[n] = float(bic)
        except Exception:
            continue

    n_optimo = min(bic_scores, key=bic_scores.get) if bic_scores else 3
    return bic_scores, n_optimo


def ajustar_hmm(retornos, meta, n_regimenes=None, n_iter=500):
    """
    Ajusta un Gaussian HMM a los retornos del mercado.

    Si n_regimenes=None, selecciona automáticamente por BIC.

    Retorna
    -------
    modelo       : modelo HMM ajustado
    estados      : Serie con estado (régimen) asignado a cada fecha
    n_regimenes  : número de regímenes usado
    params       : dict con media/volatilidad por régimen
    bic_scores   : dict {n: bic} para graficar selección de modelo
    """
    if not HMM_AVAILABLE:
        return _hmm_fallback(retornos, meta)

    serie = _retorno_mercado(retornos, meta)
    X = serie.values.reshape(-1, 1)

    # Selección automática si no se especifica
    bic_scores, n_opt = seleccionar_n_regimenes(retornos, meta)
    if n_regimenes is None:
        n_regimenes = n_opt

    modelo = hmmlearn_hmm.GaussianHMM(
        n_components=n_regimenes,
        covariance_type="full",
        n_iter=n_iter,
        random_state=42,
    )
    modelo.fit(X)
    estados_raw = modelo.predict(X)

    # Ordenar regímenes por retorno medio (0 = peor, n-1 = mejor)
    medias = [modelo.means_[i][0] for i in range(n_regimenes)]
    orden  = np.argsort(medias)
    mapa   = {old: new for new, old in enumerate(orden)}
    estados_ordenados = np.array([mapa[e] for e in estados_raw])

    estados = pd.Series(estados_ordenados, index=serie.index, name="regimen")

    # Parámetros por régimen
    params = {}
    etiquetas = {0: "Bajista", 1: "Lateral", 2: "Alcista"}
    if n_regimenes == 2:
        etiquetas = {0: "Bajista", 1: "Alcista"}

    for i in range(n_regimenes):
        idx_orig = orden[i]
        params[i] = {
            "etiqueta":   etiquetas.get(i, "Régimen {}".format(i+1)),
            "media_mens": float(modelo.means_[idx_orig][0]),
            "vol_mens":   float(np.sqrt(modelo.covars_[idx_orig][0][0])),
            "media_anual":float(modelo.means_[idx_orig][0] * 12),
            "vol_anual":  float(np.sqrt(modelo.covars_[idx_orig][0][0]) * np.sqrt(12)),
            "frecuencia": float((estados_ordenados == i).mean()),
        }

    return modelo, estados, n_regimenes, params, bic_scores


def _hmm_fallback(retornos, meta):
    """
    Fallback si hmmlearn no está disponible.
    Usa umbral simple de volatilidad para asignar regímenes.
    """
    serie = _retorno_mercado(retornos, meta)
    vol_rolling = serie.rolling(3).std()
    q33 = vol_rolling.quantile(0.33)
    q66 = vol_rolling.quantile(0.66)

    estados = pd.Series(index=serie.index, dtype=int, name="regimen")
    estados[vol_rolling <= q33] = 2   # Alcista (baja vol)
    estados[(vol_rolling > q33) & (vol_rolling <= q66)] = 1  # Lateral
    estados[vol_rolling > q66] = 0   # Bajista (alta vol)
    estados = estados.fillna(1)

    params = {
        0: {"etiqueta": "Bajista",  "media_anual": -0.05, "vol_anual": 0.20, "frecuencia": 0.33},
        1: {"etiqueta": "Lateral",  "media_anual":  0.04, "vol_anual": 0.10, "frecuencia": 0.34},
        2: {"etiqueta": "Alcista",  "media_anual":  0.12, "vol_anual": 0.08, "frecuencia": 0.33},
    }
    return None, estados, 3, params, {}


def regimen_actual(estados):
    """Retorna el régimen más reciente."""
    return int(estados.iloc[-1])


def probabilidad_transicion(modelo, n_regimenes):
    """
    Retorna la matriz de transición entre regímenes como DataFrame.
    transmat[i,j] = P(régimen j | régimen i)
    """
    if modelo is None:
        return pd.DataFrame()
    df = pd.DataFrame(
        modelo.transmat_,
        index=["Desde régimen {}".format(i) for i in range(n_regimenes)],
        columns=["A régimen {}".format(j) for j in range(n_regimenes)],
    )
    return df


def retornos_por_regimen(retornos, estados, meta):
    """
    Calcula estadísticas de retorno por régimen para cada perfil.
    Útil para entender cómo se comporta cada tipo de fondo en cada régimen.
    """
    rows = []
    regimenes = sorted(estados.unique())
    for perfil in ["conservador", "moderado", "agresivo"]:
        fondos = [f for f in retornos.columns
                  if f in meta.index and meta.loc[f, "perfil"] == perfil]
        if not fondos:
            continue
        ret_p = retornos[fondos].mean(axis=1)
        for reg in regimenes:
            fechas_reg = estados[estados == reg].index
            r_reg = ret_p[ret_p.index.isin(fechas_reg)].dropna()
            if len(r_reg) > 0:
                rows.append({
                    "perfil":   perfil,
                    "regimen":  reg,
                    "ret_med":  float(r_reg.mean() * 12),
                    "vol_med":  float(r_reg.std() * np.sqrt(12)),
                    "n_meses":  len(r_reg),
                })
    return pd.DataFrame(rows)
