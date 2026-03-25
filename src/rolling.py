"""
src/rolling.py
--------------
Análisis de correlación dinámica y métricas rodantes (rolling).

Motivación:
  Las correlaciones entre activos no son estáticas. Durante el COVID crash
  (Mar 2020), todos los activos se correlacionaron hacia 1 simultáneamente.
  En 2022, la renta fija y variable se descorrelacionaron por la subida de TPM.
  Ignorar esto subestima el riesgo en períodos de estrés.

Contenido:
  1. Correlación rodante entre pares de activos
  2. Volatilidad rodante por perfil
  3. Sharpe Ratio rodante (¿cuándo fue bueno el portafolio?)
  4. Rolling beta vs SP500
  5. Heatmap de correlación por período
"""

import numpy as np
import pandas as pd
from src.metrics import TPM_ANUAL


# ── Correlación dinámica ──────────────────────────────────────────────────────

def correlacion_rodante(retornos, fondo_a, fondo_b, ventana=12):
    """
    Correlación rodante entre dos fondos en ventana de N meses.

    Ventana de 12 meses = 1 año de historia rodante.
    Ventana de 6 meses = más sensible a cambios recientes.
    """
    r_a = retornos[fondo_a].dropna()
    r_b = retornos[fondo_b].dropna()
    df  = pd.DataFrame({"a": r_a, "b": r_b}).dropna()
    return df["a"].rolling(ventana).corr(df["b"]).rename(
        "corr({},{})".format(fondo_a, fondo_b))


def correlacion_media_rodante(retornos, meta, perfil=None, ventana=12):
    """
    Correlación promedio entre todos los fondos (o de un perfil)
    en ventana rodante. Indica cuándo el mercado se mueve junto.

    Alta correlación media → menor diversificación real → mayor riesgo sistémico.
    """
    if perfil:
        fondos = [f for f in retornos.columns
                  if f in meta.index and meta.loc[f, "perfil"] == perfil]
    else:
        fondos = list(retornos.columns)

    if len(fondos) < 2:
        return pd.Series(dtype=float)

    R = retornos[fondos].dropna(how="all")

    # Correlación media de la matriz (promedio fuera de la diagonal)
    corr_media = []
    for t in range(ventana, len(R) + 1):
        bloque = R.iloc[t - ventana:t].dropna(axis=1)
        if bloque.shape[1] < 2:
            corr_media.append(np.nan)
            continue
        corr_m = bloque.corr().values
        # Promedio de la triangular superior (sin diagonal)
        idx_up = np.triu_indices_from(corr_m, k=1)
        corr_media.append(float(np.nanmean(corr_m[idx_up])))

    return pd.Series(corr_media, index=R.index[ventana-1:], name="corr_media")


# ── Volatilidad rodante ───────────────────────────────────────────────────────

def volatilidad_rodante(retornos, meta, ventana=12):
    """
    Volatilidad anualizada rodante por perfil.
    Muestra cuándo el mercado estuvo más o menos volátil.
    """
    resultados = {}
    for perfil in ["conservador", "moderado", "agresivo", "sp500"]:
        fondos = [f for f in retornos.columns
                  if f in meta.index and meta.loc[f, "perfil"] == perfil]
        if not fondos:
            continue
        ret_p = retornos[fondos].mean(axis=1).dropna()
        vol_r = ret_p.rolling(ventana).std() * np.sqrt(12)
        resultados[perfil] = vol_r.rename("vol_{}".format(perfil))
    return pd.DataFrame(resultados)


# ── Sharpe Ratio rodante ──────────────────────────────────────────────────────

def sharpe_rodante(retornos, pesos, ventana=24, tpm_anual=TPM_ANUAL):
    """
    Sharpe Ratio rodante del portafolio.
    Ventana de 24 meses = 2 años de historia.

    Permite identificar:
    - En qué períodos el portafolio fue eficiente
    - Si el Sharpe se degradó recientemente (señal de alerta)
    """
    fondos   = [f for f in pesos if f in retornos.columns]
    w        = np.array([pesos[f] for f in fondos])
    w        = w / w.sum()
    R        = retornos[fondos].dropna()
    ret_port = pd.Series(R.values @ w, index=R.index)

    tpm_m   = tpm_anual / 12
    exceso  = ret_port - tpm_m
    sharpe  = exceso.rolling(ventana).mean() / exceso.rolling(ventana).std()
    sharpe  = sharpe * np.sqrt(12)  # anualizar

    return sharpe.rename("sharpe_rodante")


# ── Rolling beta vs SP500 ─────────────────────────────────────────────────────

def beta_rodante(retornos_portafolio, retornos_benchmark, ventana=12):
    """
    Beta rodante del portafolio vs un benchmark.

    Beta > 1 : más sensible que el mercado (amplifica movimientos)
    Beta < 1 : menos sensible (amortigua movimientos)
    Beta < 0 : se mueve en contra del mercado (cobertura natural)

    Útil para ver si el portafolio cambió su exposición sistémica en el tiempo.
    """
    df = pd.DataFrame({
        "port":  retornos_portafolio,
        "bench": retornos_benchmark,
    }).dropna()

    def beta_ventana(x):
        if len(x) < 3:
            return np.nan
        cov  = np.cov(x[:, 0], x[:, 1])
        var_b = cov[1, 1]
        return cov[0, 1] / var_b if var_b > 0 else np.nan

    betas = []
    for t in range(ventana, len(df) + 1):
        bloque = df.iloc[t - ventana:t].values
        betas.append(beta_ventana(bloque))

    return pd.Series(betas, index=df.index[ventana-1:], name="beta_rodante")


# ── Heatmap de correlación por período ───────────────────────────────────────

def correlacion_por_periodo(retornos, fondos_sel=None, n_periodos=4):
    """
    Divide el historial en N períodos iguales y calcula la correlación
    media de cada período. Muestra cómo evolucionó la estructura de
    correlaciones del mercado.
    """
    if fondos_sel is None:
        fondos_sel = list(retornos.columns)[:15]  # máximo 15 para legibilidad

    R = retornos[fondos_sel].dropna(how="all")
    n = len(R)
    step = n // n_periodos

    resultados = {}
    for i in range(n_periodos):
        inicio = i * step
        fin    = (i + 1) * step if i < n_periodos - 1 else n
        bloque = R.iloc[inicio:fin].dropna(axis=1)
        label  = "{} → {}".format(
            R.index[inicio].strftime("%Y-%m"),
            R.index[min(fin-1, n-1)].strftime("%Y-%m"))
        resultados[label] = bloque.corr()

    return resultados


# ── Resumen de métricas rodantes ─────────────────────────────────────────────

def resumen_rolling(retornos, portafolios, meta, ventana=12):
    """
    Calcula un resumen de métricas rodantes para todos los portafolios.
    Retorna DataFrame con estadísticas de la distribución del Sharpe rodante.
    """
    rows = []
    for p_id, res in portafolios.items():
        if "_robusto" in p_id:
            continue
        sh_rolling = sharpe_rodante(retornos, res["composicion"], ventana=ventana)
        sh_rolling = sh_rolling.dropna()
        if sh_rolling.empty:
            continue
        rows.append({
            "portafolio":    res["label"],
            "sharpe_medio":  float(sh_rolling.mean()),
            "sharpe_min":    float(sh_rolling.min()),
            "sharpe_max":    float(sh_rolling.max()),
            "sharpe_actual": float(sh_rolling.iloc[-1]),
            "pct_positivo":  float((sh_rolling > 0).mean()),
            "estabilidad":   float(1 / sh_rolling.std()) if sh_rolling.std() > 0 else np.nan,
        })
    return pd.DataFrame(rows)
