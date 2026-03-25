"""
src/backtesting.py
------------------
Walk-forward validation para evaluar la robustez del portafolio óptimo.

Metodología:
1. Se divide el historial en ventanas deslizantes
2. En cada ventana: calibrar en 70%, evaluar en 30% (out-of-sample)
3. Métricas out-of-sample son las realmente válidas para reportar

Esto evita el sesgo de look-ahead: no se puede usar información futura
para tomar decisiones pasadas.

Esquema walk-forward:
|--- train_1 ---|-- test_1 --|
        |--- train_2 ---|-- test_2 --|
                |--- train_3 ---|-- test_3 --|

Referencias:
- Pardo (2008): "The Evaluation and Optimization of Trading Strategies"
- Lopez de Prado (2018): "Advances in Financial Machine Learning"
"""

import numpy as np
import pandas as pd
from src.optimizer import optimizar, optimizar_global, PERFILES
from src.metrics import sharpe_ratio, max_drawdown, var_historico, TPM_ANUAL


def walk_forward_validation(retornos, meta, perfil="optimo",
                             train_size=0.70, n_splits=5,
                             peso_min=0.02, peso_max=0.40,
                             tpm_anual=TPM_ANUAL):
    """
    Walk-forward validation con ventana expandible.

    Parámetros
    ----------
    train_size : fracción del total para entrenamiento en cada split
    n_splits   : número de splits (ventanas)

    Retorna
    -------
    resultados : lista de dicts con métricas in-sample y out-of-sample
    df_resumen : DataFrame con métricas promedio por split
    """
    n_total = len(retornos)
    min_train = int(n_total * train_size * 0.5)  # mínimo absoluto para entrenar

    # Generar splits
    splits = []
    step = max(1, (n_total - min_train) // n_splits)

    for i in range(n_splits):
        train_end = min_train + i * step
        test_start = train_end
        test_end   = min(train_end + step, n_total)

        if test_start >= n_total or train_end - 0 < 24:
            continue

        splits.append({
            "train": retornos.iloc[:train_end],
            "test":  retornos.iloc[test_start:test_end],
            "fecha_train_fin":  retornos.index[train_end - 1],
            "fecha_test_inicio": retornos.index[test_start],
            "fecha_test_fin":   retornos.index[test_end - 1],
        })

    resultados = []

    for i, split in enumerate(splits):
        ret_train = split["train"]
        ret_test  = split["test"]

        # Optimizar en train
        if perfil == "optimo":
            res = optimizar_global(ret_train, meta,
                                    peso_min=0.0, peso_max=0.49,
                                    tpm_anual=tpm_anual)
        else:
            res = optimizar(ret_train, meta, perfil,
                            peso_min=peso_min, peso_max=peso_max,
                            tpm_anual=tpm_anual)

        if res is None:
            continue

        # Evaluar pesos en test (out-of-sample)
        fondos_disp = [f for f in res["composicion"] if f in ret_test.columns]
        if not fondos_disp:
            continue

        pesos_norm = {f: res["composicion"][f] for f in fondos_disp}
        total_peso = sum(pesos_norm.values())
        pesos_norm = {f: p / total_peso for f, p in pesos_norm.items()}

        w_test = np.array([pesos_norm[f] for f in fondos_disp])
        R_test = ret_test[fondos_disp].dropna()
        if R_test.empty:
            continue
        ret_port_test = R_test.values @ w_test

        # Métricas in-sample
        w_train = np.array([res["composicion"].get(f, 0) for f in res["fondos"]])
        R_train = ret_train[res["fondos"]].dropna()
        if not R_train.empty:
            ret_port_train = R_train.values @ np.array([
                res["composicion"].get(f, 0) for f in res["fondos"]
            ])
            sh_train = sharpe_ratio(ret_port_train, tpm_anual)
        else:
            sh_train = np.nan

        sh_test  = sharpe_ratio(ret_port_test, tpm_anual)
        mdd_test = max_drawdown(ret_port_test)

        resultados.append({
            "split":              i + 1,
            "fecha_train_fin":    split["fecha_train_fin"].date(),
            "fecha_test_inicio":  split["fecha_test_inicio"].date(),
            "fecha_test_fin":     split["fecha_test_fin"].date(),
            "n_meses_train":      len(ret_train),
            "n_meses_test":       len(R_test),
            "sharpe_insample":    float(sh_train) if not np.isnan(sh_train) else None,
            "sharpe_outsample":   float(sh_test)  if not np.isnan(sh_test)  else None,
            "ret_anual_test":     float(ret_port_test.mean() * 12),
            "vol_anual_test":     float(ret_port_test.std() * np.sqrt(12)),
            "max_drawdown_test":  float(mdd_test),
            "var_95_test":        float(var_historico(ret_port_test, 0.95)),
            "n_activos":          res["n_activos"],
            "composicion":        res["composicion"],
        })

    df_resumen = pd.DataFrame([{
        "split":             r["split"],
        "periodo_test":      "{} → {}".format(r["fecha_test_inicio"], r["fecha_test_fin"]),
        "sharpe_insample":   r["sharpe_insample"],
        "sharpe_outsample":  r["sharpe_outsample"],
        "ret_anual_test":    r["ret_anual_test"],
        "vol_anual_test":    r["vol_anual_test"],
        "max_drawdown_test": r["max_drawdown_test"],
        "n_activos":         r["n_activos"],
    } for r in resultados])

    return resultados, df_resumen


def degradacion_sharpe(df_resumen):
    """
    Calcula la degradación promedio del Sharpe entre in-sample y out-of-sample.
    Un valor negativo grande indica sobreajuste (overfitting).
    """
    df = df_resumen.dropna(subset=["sharpe_insample", "sharpe_outsample"])
    if df.empty:
        return {}
    degradacion = df["sharpe_outsample"] - df["sharpe_insample"]
    return {
        "degradacion_promedio":  float(degradacion.mean()),
        "degradacion_mediana":   float(degradacion.median()),
        "pct_splits_positivos":  float((df["sharpe_outsample"] > 0).mean()),
        "sharpe_is_promedio":    float(df["sharpe_insample"].mean()),
        "sharpe_oos_promedio":   float(df["sharpe_outsample"].mean()),
    }
