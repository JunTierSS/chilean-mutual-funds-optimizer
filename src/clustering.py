"""
src/clustering.py
-----------------
Clustering de fondos mutuos por comportamiento histórico de retornos.

Motivación: la clasificación oficial (conservador/moderado/agresivo) de la CMF
no siempre refleja el comportamiento real de los fondos. Dos fondos "agresivos"
pueden tener correlación baja y comportarse muy distinto en mercados bajistas.

Métodos implementados:
1. K-Means sobre retornos normalizados
2. Clustering jerárquico (Ward) sobre matriz de correlaciones
3. Selección automática del número óptimo de clusters (silhouette + elbow)

Referencias:
- Hierarchical clustering en finanzas: Mantegna (1999)
- Silhouette analysis: Rousseeuw (1987)
"""

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def correlacion_a_distancia(corr_matrix):
    """
    Convierte matriz de correlación a matriz de distancia usando:
    d(i,j) = sqrt(0.5 * (1 - ρ(i,j)))

    Esta métrica satisface la desigualdad triangular y es estándar
    en análisis jerárquico de mercados financieros (Mantegna 1999).
    """
    dist = np.sqrt(0.5 * (1 - corr_matrix.values))
    np.fill_diagonal(dist, 0)
    return pd.DataFrame(dist, index=corr_matrix.index, columns=corr_matrix.columns)


def clustering_jerarquico(retornos, metodo_linkage="ward"):
    """
    Clustering jerárquico sobre matriz de correlaciones de retornos.

    Parámetros
    ----------
    retornos        : DataFrame wide fecha × fondo_id
    metodo_linkage  : 'ward' | 'complete' | 'average' | 'single'

    Retorna
    -------
    linkage_matrix : para graficar dendrograma
    corr_matrix    : matriz de correlación
    dist_matrix    : matriz de distancia
    """
    R = retornos.dropna(how="all", axis=1)
    corr = R.corr()
    dist = correlacion_a_distancia(corr)

    # Condensar matriz de distancia (triangular superior)
    dist_condensada = squareform(dist.values)
    linkage_matrix  = hierarchy.linkage(dist_condensada, method=metodo_linkage)

    return linkage_matrix, corr, dist


def asignar_clusters_jerarquico(retornos, n_clusters, metodo_linkage="ward"):
    """
    Asigna cluster a cada fondo usando clustering jerárquico.

    Retorna Serie con {fondo_id: cluster_id}.
    """
    R = retornos.dropna(how="all", axis=1)
    corr = R.corr()
    dist = correlacion_a_distancia(corr)
    dist_condensada = squareform(dist.values)
    linkage_matrix  = hierarchy.linkage(dist_condensada, method=metodo_linkage)
    labels = hierarchy.fcluster(linkage_matrix, n_clusters, criterion="maxclust")
    return pd.Series(labels, index=R.columns, name="cluster")


def kmeans_fondos(retornos, n_clusters=None, max_k=8, random_state=42):
    """
    K-Means sobre features estadísticos de cada fondo:
    [retorno_anual, volatilidad_anual, skewness, max_drawdown, autocorr_lag1]

    Selección automática de k si n_clusters=None usando silhouette score.

    Retorna
    -------
    labels      : Serie {fondo_id: cluster}
    k_optimo    : número de clusters elegido
    silhouettes : dict {k: silhouette_score} para el elbow plot
    """
    R = retornos.dropna(how="all", axis=1).dropna()

    # Construir features por fondo
    features = {}
    for fid in R.columns:
        r = R[fid].values
        features[fid] = {
            "retorno_anual":     r.mean() * 12,
            "volatilidad_anual": r.std() * np.sqrt(12),
            "skewness":          float(pd.Series(r).skew()),
            "max_drawdown":      float(np.min(np.cumprod(1+r)/np.maximum.accumulate(np.cumprod(1+r))-1)),
            "autocorr_lag1":     float(pd.Series(r).autocorr(lag=1)) if len(r) > 1 else 0,
        }

    X = pd.DataFrame(features).T
    X_scaled = StandardScaler().fit_transform(X)

    # Selección automática de k
    silhouettes = {}
    k_range = range(2, min(max_k + 1, len(R.columns)))

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels_k = km.fit_predict(X_scaled)
        if len(np.unique(labels_k)) > 1:
            silhouettes[k] = silhouette_score(X_scaled, labels_k)

    if n_clusters is None:
        k_optimo = max(silhouettes, key=silhouettes.get) if silhouettes else 3
    else:
        k_optimo = n_clusters

    # Ajuste final con k óptimo
    km_final = KMeans(n_clusters=k_optimo, random_state=random_state, n_init=10)
    labels_final = km_final.fit_predict(X_scaled)

    labels_serie = pd.Series(labels_final + 1, index=X.index, name="cluster")
    return labels_serie, k_optimo, silhouettes, X


def resumen_clusters(labels, retornos, meta):
    """
    Genera resumen estadístico por cluster:
    retorno medio, volatilidad media, fondos por perfil, etc.
    """
    R = retornos.dropna(how="all", axis=1)
    rows = []
    for cluster_id in sorted(labels.unique()):
        fondos_c = labels[labels == cluster_id].index.tolist()
        r_c = R[fondos_c].mean(axis=1).values
        perfiles = [meta.loc[f, "perfil"] if f in meta.index else "?" for f in fondos_c]
        corredoras = [meta.loc[f, "corredora"] if f in meta.index else "?" for f in fondos_c]
        rows.append({
            "cluster":           cluster_id,
            "n_fondos":          len(fondos_c),
            "retorno_anual":     float(r_c.mean() * 12),
            "volatilidad_anual": float(r_c.std() * np.sqrt(12)),
            "perfil_dominante":  max(set(perfiles), key=perfiles.count),
            "corredoras":        ", ".join(sorted(set(corredoras))),
            "fondos":            ", ".join([meta.loc[f,"nombre"][:20] if f in meta.index else f
                                            for f in fondos_c[:3]]) + ("..." if len(fondos_c) > 3 else ""),
        })
    return pd.DataFrame(rows)
