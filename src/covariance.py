"""
src/covariance.py
-----------------
Estimadores robustos de la matriz de covarianza para optimización de portafolios.

El estimador clásico de covarianza (muestral) es inestable cuando el número
de activos p es grande relativo al número de observaciones n. Con 42 fondos
y ~75 meses, p/n ≈ 0.56 — zona de inestabilidad.

Solución: Ledoit-Wolf shrinkage (2004), que combina la covarianza muestral
con un estimador más estable (target diagonal) minimizando el error cuadrático
medio de Frobenius.

Referencias:
- Ledoit & Wolf (2004): "A well-conditioned estimator for large-dimensional
  covariance matrices", Journal of Multivariate Analysis
- scikit-learn: sklearn.covariance.LedoitWolf
"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, OAS


def covarianza_muestral(retornos):
    """
    Covarianza muestral estándar. Sesgada cuando n/p es pequeño.
    Usada como baseline de comparación.
    """
    return retornos.cov().values, None


def covarianza_ledoit_wolf(retornos):
    """
    Estimador Ledoit-Wolf shrinkage.

    Combina la covarianza muestral Σ_sample con un target diagonal T:
        Σ_lw = (1 - α) * Σ_sample + α * T

    donde α (shrinkage_intensity) se calcula analíticamente para
    minimizar el error cuadrático esperado.

    Parámetros
    ----------
    retornos : DataFrame wide (fechas × fondos)

    Retorna
    -------
    cov_matrix    : np.ndarray de covarianza shrinkage
    shrinkage_coef: coeficiente α aplicado (entre 0 y 1)
                    0 = sin shrinkage, 1 = solo target diagonal
    """
    R = retornos.dropna().values
    lw = LedoitWolf(assume_centered=False)
    lw.fit(R)
    return lw.covariance_, lw.shrinkage_


def covarianza_oas(retornos):
    """
    Oracle Approximating Shrinkage (OAS) — alternativa a Ledoit-Wolf.
    Generalmente más precisa cuando la distribución es aproximadamente gaussiana.
    """
    R = retornos.dropna().values
    oas = OAS(assume_centered=False)
    oas.fit(R)
    return oas.covariance_, oas.shrinkage_


def comparar_estimadores(retornos):
    """
    Compara condition number de los distintos estimadores.
    Un condition number alto indica matriz inestable (problemas de inversión).

    Retorna DataFrame comparativo con:
    - condition_number: razón entre eigenvalor máximo y mínimo
    - shrinkage: coeficiente aplicado
    - det_log: log del determinante (proxy de "información")
    """
    resultados = {}

    # Muestral
    cov_m, _ = covarianza_muestral(retornos)
    eigvals_m = np.linalg.eigvalsh(cov_m)
    eigvals_m = eigvals_m[eigvals_m > 1e-10]
    resultados["Muestral"] = {
        "condition_number": float(eigvals_m.max() / eigvals_m.min()),
        "shrinkage":        0.0,
        "min_eigenvalue":   float(eigvals_m.min()),
    }

    # Ledoit-Wolf
    cov_lw, alpha_lw = covarianza_ledoit_wolf(retornos)
    eigvals_lw = np.linalg.eigvalsh(cov_lw)
    eigvals_lw = eigvals_lw[eigvals_lw > 1e-10]
    resultados["Ledoit-Wolf"] = {
        "condition_number": float(eigvals_lw.max() / eigvals_lw.min()),
        "shrinkage":        float(alpha_lw),
        "min_eigenvalue":   float(eigvals_lw.min()),
    }

    # OAS
    cov_oas, alpha_oas = covarianza_oas(retornos)
    eigvals_oas = np.linalg.eigvalsh(cov_oas)
    eigvals_oas = eigvals_oas[eigvals_oas > 1e-10]
    resultados["OAS"] = {
        "condition_number": float(eigvals_oas.max() / eigvals_oas.min()),
        "shrinkage":        float(alpha_oas),
        "min_eigenvalue":   float(eigvals_oas.min()),
    }

    return pd.DataFrame(resultados).T


def get_cov_matrix(retornos, metodo="ledoit_wolf"):
    """
    Interfaz unificada para obtener la matriz de covarianza.

    metodo: 'muestral' | 'ledoit_wolf' | 'oas'
    Retorna (cov_matrix, shrinkage_coef)
    """
    if metodo == "ledoit_wolf":
        return covarianza_ledoit_wolf(retornos)
    elif metodo == "oas":
        return covarianza_oas(retornos)
    else:
        return covarianza_muestral(retornos)
