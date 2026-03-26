"""
app.py
------
Chilean Mutual Funds Portfolio Optimizer — Streamlit MVP
Portafolio: Ing. Civil Industrial + MSc Data Science

Run:  streamlit run app.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import io
import plotly.figure_factory as ff

from src.backtesting import degradacion_sharpe, walk_forward_validation
from src.bootstrap import proyectar_bootstrap, proyectar_montecarlo_normal
from src.clustering import kmeans_fondos, clustering_jerarquico, resumen_clusters
from src.garch import garch_todos_perfiles, vol_condicional_portafolio
from src.hrp import hrp_portfolio
from src.loader import cargar_todos
from src.metrics import (
    cvar_historico,
    max_drawdown,
    sharpe_ratio,
    simular_historico,
    sortino_ratio,
    tabla_stats,
    var_historico,
)
from src.optimizer import PERFILES, frontera_eficiente, optimizar, optimizar_global
from src.regimes import ajustar_hmm
from src.rolling import (
    beta_rodante,
    correlacion_media_rodante,
    sharpe_rodante,
    volatilidad_rodante,
)
from src.stress_hipotetico import stress_hipotetico, ESCENARIOS_HIPOTETICOS

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chilean Funds Optimizer",
    page_icon="🇨🇱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stMetricValue"]  { font-size: 1.25rem; font-weight: 700; }
    [data-testid="stMetricLabel"]  { font-size: 0.78rem; color: #aaa; }
    .stTabs [data-baseweb="tab"]   { font-size: 0.85rem; font-weight: 600; }
    .block-container               { padding-top: 1.2rem; padding-bottom: 2rem; }
    div[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)

# Paleta de colores consistente
COLORES = {
    "conservador": "#2ecc71",
    "moderado":    "#f1c40f",
    "agresivo":    "#e74c3c",
    "optimo":      "#FFD700",
    "sp500":       "#00b4d8",
    "hrp":         "#9b59b6",
}

LAYOUT_DARK = dict(
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    font_color="white",
    margin=dict(t=30, b=20),
)

# ─────────────────────────────────────────────────────────────────────────────
# CARGA Y CACHÉ DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

def _data_fingerprint():
    """Hash basado en la fecha de modificación más reciente de los CSV de datos."""
    import os
    from pathlib import Path as _P
    mtimes = []
    for p in _P("data").rglob("*.csv"):
        try:
            mtimes.append(int(os.path.getmtime(p)))
        except OSError:
            pass
    return max(mtimes) if mtimes else 0


@st.cache_data(show_spinner="Cargando fondos históricos…")
def load_data(_fingerprint=0):
    return cargar_todos(base_dir="data", fecha_inicio="2020-01-01")


@st.cache_data(show_spinner="Calculando métricas…")
def get_stats(retornos, meta):
    return tabla_stats(retornos, meta)


@st.cache_data(show_spinner="Optimizando portafolio…")
def run_optimization(retornos, meta, perfil, n_intentos=40):
    if perfil == "optimo":
        return optimizar_global(retornos, meta, n_intentos=n_intentos,
                                peso_min=0.0, peso_max=0.49)
    return optimizar(retornos, meta, perfil, n_intentos=n_intentos)


@st.cache_data(show_spinner="Calculando frontera eficiente…")
def run_frontera(retornos):
    return frontera_eficiente(retornos, n_puntos=50)


@st.cache_data(show_spinner="Detectando regímenes de mercado…")
def run_hmm(retornos, meta):
    modelo, estados, n_reg, params, bic_scores = ajustar_hmm(retornos, meta)
    return estados, n_reg, params, bic_scores   # descartamos modelo (no serializable)


@st.cache_data(show_spinner="Simulando proyecciones…")
def run_bootstrap(retornos, pesos_tuple, monto, n_meses,
                  aporte_mensual=0, crecimiento_anual=0.0, n_sim=600):
    pesos = dict(pesos_tuple)
    bs = proyectar_bootstrap(retornos, pesos, monto, n_meses,
                             n_sim=n_sim, aporte_mensual=aporte_mensual,
                             crecimiento_anual=crecimiento_anual)
    mc = proyectar_montecarlo_normal(retornos, pesos, monto, n_meses,
                                     n_sim=n_sim, aporte_mensual=aporte_mensual,
                                     crecimiento_anual=crecimiento_anual)
    return bs, mc


@st.cache_data(show_spinner="Ejecutando walk-forward…")
def run_backtest(retornos, meta, perfil):
    _, df = walk_forward_validation(retornos, meta, perfil=perfil, n_splits=4,
                                    peso_min=0.0, peso_max=0.49)
    return df


@st.cache_data(show_spinner="Calculando HRP…")
def run_hrp(retornos, meta):
    return hrp_portfolio(retornos, meta)


@st.cache_data(show_spinner="Ajustando GARCH por perfil…")
def run_garch(retornos, meta):
    return garch_todos_perfiles(retornos, meta)


@st.cache_data(show_spinner="Calculando métricas rodantes…")
def run_rolling(retornos, meta):
    return volatilidad_rodante(retornos, meta, ventana=12)


@st.cache_data(show_spinner="Ejecutando stress test…")
def run_stress(retornos, meta, monto):
    portafolios = {}
    for p in ["conservador", "moderado", "agresivo", "optimo"]:
        res = run_optimization(retornos, meta, p)
        if res:
            portafolios[p] = res
    if "SP500" in retornos.columns:
        from src.optimizer import portafolio_sp500_puro
        sp = portafolio_sp500_puro(retornos, meta)
        if sp:
            portafolios["sp500"] = sp
    return stress_hipotetico(portafolios, retornos, meta, monto=monto)


@st.cache_data(show_spinner="Calculando clustering…")
def run_clustering(retornos, meta):
    labels, k_opt, silhouettes, features = kmeans_fondos(retornos)
    linkage, corr, dist = clustering_jerarquico(retornos)
    resumen = resumen_clusters(labels, retornos, meta)
    return labels, k_opt, silhouettes, features, linkage, corr, resumen


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def fmt_clp(v):
    return f"CLP {v:,.0f}"

def fmt_pct(v):
    return f"{v:.2%}" if pd.notna(v) else "—"

def fmt_n(v, decimales=3):
    return f"{v:.{decimales}f}" if pd.notna(v) else "—"

def _hex_rgba(hex_color, alpha):
    """Convierte color hex a rgba string para Plotly."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _capital_acum(monto, aporte, crecimiento_anual, n_meses):
    """
    Curva de capital aportado acumulado (dotted line en fan charts).
    Con crecimiento: suma geométrica del aporte creciente + monto inicial.
    Sin crecimiento (tasa≈0): monto + aporte * t (lineal).
    """
    tasa_m = (1 + crecimiento_anual) ** (1 / 12) - 1
    if tasa_m < 1e-10:
        return [monto + aporte * (t + 1) for t in range(n_meses)]
    return [monto + aporte * ((1 + tasa_m) ** (t + 1) - 1) / tasa_m
            for t in range(n_meses)]


def filtrar_universo(retornos, meta, brokers):
    fondos = [f for f in retornos.columns
              if f in meta.index and meta.loc[f, "corredora"] in brokers]
    if not fondos:
        st.error("⚠️ Selecciona al menos un broker en el sidebar.")
        st.stop()
    return retornos[fondos], meta.loc[fondos]


def _regime_spans(estados):
    """Agrupa fechas consecutivas del mismo régimen en intervalos."""
    spans = []
    if estados.empty:
        return spans
    cur_reg = int(estados.iloc[0])
    start   = estados.index[0]
    for date, reg in estados.items():
        reg = int(reg)
        if reg != cur_reg:
            spans.append((cur_reg, start, date))
            cur_reg = reg
            start   = date
    spans.append((cur_reg, start, estados.index[-1]))
    return spans


def _port_returns(retornos, composicion):
    """Calcula retornos del portafolio dado composición {fondo: peso}."""
    fondos = [f for f in composicion if f in retornos.columns]
    w      = np.array([composicion[f] for f in fondos])
    w      = w / w.sum()
    R      = retornos[fondos].dropna()
    return R.values @ w, fondos, w


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🇨🇱 Chilean Funds")
        st.caption("Optimizador de Fondos Mutuos")
        st.divider()

        st.markdown("**Universo de inversión**")
        use_sant = st.checkbox("Santander AM",   value=True)
        use_lv   = st.checkbox("LarrainVial AM", value=True)
        use_sp   = st.checkbox("S&P 500 (CLP)",  value=True)

        # Nuevas corredoras — disponibles solo si se han descargado los datos
        import os as _os
        _has_bci  = _os.path.isdir("data/bci")        and bool(list(__import__("pathlib").Path("data/bci").glob("*.csv")))
        _has_bice = _os.path.isdir("data/bice")       and bool(list(__import__("pathlib").Path("data/bice").glob("*.csv")))
        _has_be   = _os.path.isdir("data/bancochile") and bool(list(__import__("pathlib").Path("data/bancochile").glob("*.csv")))
        _has_sec  = _os.path.isdir("data/security")   and bool(list(__import__("pathlib").Path("data/security").glob("*.csv")))

        if any([_has_bci, _has_bice, _has_be, _has_sec]):
            st.caption("Corredoras adicionales")
        use_bci  = st.checkbox("BCI",          value=_has_bci,  disabled=not _has_bci)
        use_bice = st.checkbox("Bice Inv.",    value=_has_bice, disabled=not _has_bice)
        use_be   = st.checkbox("BancoEstado",  value=_has_be,   disabled=not _has_be)
        use_sec  = st.checkbox("Security",     value=_has_sec,  disabled=not _has_sec)

        st.divider()
        st.markdown("**Perfil de riesgo**")
        perfil = st.radio(
            "perfil",
            ["conservador", "moderado", "agresivo", "optimo"],
            format_func=lambda x: {
                "conservador": "🟢 Conservador",
                "moderado":    "🟡 Moderado",
                "agresivo":    "🔴 Agresivo",
                "optimo":      "⭐ Óptimo Global",
            }[x],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("**Monto inicial (CLP)**")
        monto = st.number_input(
            "monto",
            min_value=1_000_000,
            max_value=2_000_000_000,
            value=10_000_000,
            step=1_000_000,
            label_visibility="collapsed",
        )
        st.caption(f"CLP {monto:,.0f}")

        st.markdown("**Horizonte de proyección**")
        horizonte = st.select_slider(
            "horizonte",
            options=[3, 5, 10],
            value=5,
            format_func=lambda x: f"{x} años",
            label_visibility="collapsed",
        )

        st.markdown("**Aporte mensual (CLP)**")
        aporte = st.number_input(
            "aporte",
            min_value=0,
            max_value=50_000_000,
            value=0,
            step=100_000,
            help="Aporte adicional que se deposita cada mes",
            label_visibility="collapsed",
        )
        if aporte > 0:
            st.caption(f"CLP {aporte:,.0f}/mes")

        st.markdown("**Crecimiento anual del aporte**")
        crecimiento_pct = st.slider(
            "crecimiento_aporte",
            min_value=0, max_value=10, value=3, step=1,
            format="%d%%",
            help="El aporte crece esta tasa cada año (ej: ajuste por inflación). "
                 "Se ignora si el aporte mensual es 0.",
            label_visibility="collapsed",
            disabled=(aporte == 0),
        )
        crecimiento_aporte = crecimiento_pct / 100 if aporte > 0 else 0.0

        st.divider()
        st.caption("Fuente: Investing.com · CMF · 2020–2026")

    brokers = []
    if use_sant: brokers.append("santander")
    if use_lv:   brokers.append("larrain_vial")
    if use_sp:   brokers.append("externo")
    if use_bci:  brokers.append("bci")
    if use_bice: brokers.append("bice")
    if use_be:   brokers.append("bancochile")
    if use_sec:  brokers.append("security")
    return perfil, monto, horizonte, aporte, crecimiento_aporte, brokers


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — MERCADO
# ─────────────────────────────────────────────────────────────────────────────

def tab_mercado(retornos, meta, brokers):
    st.header("📊 Panorama del mercado")

    # ── KPIs ──
    n_ch  = len(meta[meta["corredora"] != "externo"])
    fecha_ini = retornos.index.min().strftime("%b %Y")
    fecha_fin = retornos.index.max().strftime("%b %Y")

    # Corredoras activas dinámicas
    nombres_broker = {
        "santander":   "Santander AM",
        "larrain_vial":"LarrainVial AM",
        "externo":     "S&P 500 (CLP)",
        "bci":         "BCI",
        "bice":        "Bice Inv.",
        "bancochile":  "BancoEstado",
        "security":    "Security",
    }
    corredoras_str = " · ".join(nombres_broker[b] for b in brokers if b in nombres_broker)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fondos seleccionados", str(len(meta)))
    c2.metric("Período",              f"{fecha_ini} → {fecha_fin}")
    c3.metric("Meses de historia",    str(len(retornos)))
    c4.metric("Fuentes activas",      corredoras_str)

    st.divider()

    # ── Evolución normalizada ──
    st.subheader("Evolución de valor cuota — base 100 (Ene 2020)")

    fig = go.Figure()
    for p in ["conservador", "moderado", "agresivo"]:
        fondos_p = [f for f in retornos.columns
                    if f in meta.index and meta.loc[f, "perfil"] == p]
        if not fondos_p:
            continue
        cum = (1 + retornos[fondos_p].mean(axis=1).dropna()).cumprod() * 100
        fig.add_trace(go.Scatter(x=cum.index, y=cum.values,
                                 name=p.capitalize(),
                                 line=dict(color=COLORES[p], width=2)))

    if "SP500" in retornos.columns:
        cum_sp = (1 + retornos["SP500"].dropna()).cumprod() * 100
        fig.add_trace(go.Scatter(x=cum_sp.index, y=cum_sp.values,
                                 name="S&P 500 (CLP)",
                                 line=dict(color=COLORES["sp500"], width=2, dash="dash")))

    fig.update_layout(height=420, hovermode="x unified",
                      yaxis_title="Índice (base 100)", **LAYOUT_DARK,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Correlación ──
    col_l, col_r = st.columns([3, 1])
    with col_l:
        st.subheader("Matriz de correlación")
        R    = retornos.dropna(how="all", axis=1)
        corr = R.corr()
        # Truncar nombres y deduplicar añadiendo sufijo numérico
        seen = {}
        short_names = {}
        for f in corr.columns:
            base = meta.loc[f, "nombre"][:18] if f in meta.index else f
            if base in seen:
                seen[base] += 1
                short_names[f] = f"{base} ({seen[base]})"
            else:
                seen[base] = 1
                short_names[f] = base
        corr_plot = corr.rename(index=short_names, columns=short_names)
        fig_c = px.imshow(corr_plot, color_continuous_scale="RdBu_r",
                          zmin=-1, zmax=1, height=480)
        fig_c.update_layout(**LAYOUT_DARK)
        st.plotly_chart(fig_c, use_container_width=True)

    with col_r:
        st.subheader("Estadísticas")
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        vals = corr.where(mask).stack().values
        st.metric("Correlación media",  f"{vals.mean():.3f}")
        st.metric("Correlación máxima", f"{vals.max():.3f}")
        st.metric("Correlación mínima", f"{vals.min():.3f}")
        st.caption("Alta correlación media → menor diversificación real → mayor riesgo sistémico")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — FONDOS
# ─────────────────────────────────────────────────────────────────────────────

def tab_fondos(retornos, meta, stats):
    st.header("🔍 Explorador de fondos")

    # ── Filtros ──
    c1, c2, c3 = st.columns(3)
    perfiles_disp  = sorted(stats["perfil"].unique())
    corr_disp      = sorted(stats["corredora"].unique())

    perfil_f = c1.multiselect("Perfil",    perfiles_disp, default=perfiles_disp)
    corr_f   = c2.multiselect("Corredora", corr_disp,     default=corr_disp)
    sort_col = c3.selectbox("Ordenar por",
                            ["sharpe", "retorno_anual", "volatilidad_anual",
                             "sortino", "max_drawdown"])

    df_f = stats[
        stats["perfil"].isin(perfil_f) &
        stats["corredora"].isin(corr_f)
    ].sort_values(sort_col, ascending=(sort_col in ["max_drawdown", "volatilidad_anual"]))

    # ── Tabla ──
    show_cols = ["nombre", "perfil", "corredora", "retorno_anual",
                 "volatilidad_anual", "sharpe", "sortino", "var_95",
                 "cvar_95", "max_drawdown", "n_meses"]
    df_d = df_f[show_cols].copy()
    df_d.columns = ["Fondo", "Perfil", "Corredora", "Ret. Anual", "Vol. Anual",
                    "Sharpe", "Sortino", "VaR 95%", "CVaR 95%", "Max DD", "Meses"]
    for col in ["Ret. Anual", "Vol. Anual", "VaR 95%", "CVaR 95%", "Max DD"]:
        df_d[col] = df_d[col].apply(lambda x: fmt_pct(x))
    for col in ["Sharpe", "Sortino"]:
        df_d[col] = df_d[col].apply(lambda x: fmt_n(x))

    st.dataframe(df_d, use_container_width=True, height=420)
    st.caption(f"{len(df_f)} fondos · haz clic en una fila para explorar")

    st.divider()

    # ── Scatter riesgo-retorno ──
    st.subheader("Mapa riesgo-retorno")
    fig_s = px.scatter(
        df_f.dropna(subset=["volatilidad_anual", "retorno_anual", "sharpe"]),
        x="volatilidad_anual", y="retorno_anual",
        color="perfil", hover_name="nombre",
        size=df_f.dropna(subset=["volatilidad_anual", "retorno_anual", "sharpe"])["sharpe"]
            .clip(lower=0.01),
        color_discrete_map=COLORES,
        labels={"volatilidad_anual": "Volatilidad Anual",
                "retorno_anual":     "Retorno Anual",
                "perfil":            "Perfil"},
        height=500,
    )
    fig_s.update_traces(marker=dict(opacity=0.85, line=dict(width=0.5, color="white")))
    fig_s.update_layout(xaxis_tickformat=".1%", yaxis_tickformat=".1%", **LAYOUT_DARK)
    st.plotly_chart(fig_s, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — PORTAFOLIO
# ─────────────────────────────────────────────────────────────────────────────

def tab_portafolio(retornos, meta, perfil, monto):
    st.header("🎯 Portafolio óptimo")

    res = run_optimization(retornos, meta, perfil)
    if res is None:
        st.warning("No se pudo optimizar. Activa más brokers o cambia el perfil.")
        return

    ret_p, fondos_c, w = _port_returns(retornos, res["composicion"])
    label = res["label"]

    # ── KPIs ──
    st.subheader(f"Portafolio **{label}** — Markowitz + Ledoit-Wolf")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Sharpe",     fmt_n(res["sharpe"]))
    c2.metric("Sortino",    fmt_n(sortino_ratio(ret_p)))
    c3.metric("Retorno",    fmt_pct(res["ret_anual"]))
    c4.metric("Volatilidad",fmt_pct(res["vol_anual"]))
    c5.metric("VaR 95%",    fmt_pct(var_historico(ret_p, 0.95)))
    c6.metric("Max DD",     fmt_pct(max_drawdown(ret_p)))

    if res.get("shrinkage") is not None:
        st.caption(f"Ledoit-Wolf shrinkage α = {res['shrinkage']:.3f} · "
                   f"{res['n_activos']} activos con peso significativo")
    st.divider()

    # ── Composición + retorno histórico ──
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Composición del portafolio")
        comp   = res["composicion"]
        labels = [meta.loc[f, "nombre"][:28] if f in meta.index else f for f in comp]
        values = list(comp.values())
        fig_p  = go.Figure(go.Pie(
            labels=labels, values=values, hole=0.42,
            textposition="inside", textinfo="percent",
            hovertemplate="%{label}<br>%{percent}<extra></extra>",
        ))
        fig_p.update_layout(height=400, showlegend=True,
                            legend=dict(font=dict(size=10)),
                            **LAYOUT_DARK)
        st.plotly_chart(fig_p, use_container_width=True)

    with col_r:
        st.subheader("Retorno histórico (in-sample)")
        hist = simular_historico(retornos, {f: comp[f] for f in fondos_c}, monto)
        ret_pct = (hist.iloc[-1] / monto - 1)
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(
            x=hist.index, y=hist.values,
            fill="tozeroy",
            fillcolor=f"rgba(255,215,0,0.08)",
            line=dict(color=res["color"], width=2),
            name=label,
            hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>",
        ))
        fig_h.add_hline(y=monto, line_dash="dot", line_color="#555",
                        annotation_text="Capital inicial")
        fig_h.update_layout(height=400, yaxis_title="Valor (CLP)",
                            yaxis_tickprefix="$", yaxis_tickformat=",",
                            hovermode="x unified", **LAYOUT_DARK)
        st.plotly_chart(fig_h, use_container_width=True)
        st.metric("Retorno acumulado total", fmt_pct(ret_pct))

    # ── Tabla de pesos ──
    with st.expander("Ver pesos detallados"):
        df_pesos = pd.DataFrame([{
            "Fondo":     meta.loc[f, "nombre"] if f in meta.index else f,
            "Perfil":    meta.loc[f, "perfil"] if f in meta.index else "—",
            "Corredora": meta.loc[f, "corredora"] if f in meta.index else "—",
            "Peso":      fmt_pct(w_i),
        } for f, w_i in sorted(comp.items(), key=lambda x: -x[1])])
        st.dataframe(df_pesos, use_container_width=True)

    # ── Export Excel ──
    st.divider()
    st.subheader("⬇️ Exportar portafolio")
    df_bt = run_backtest(retornos, meta, perfil)
    stats_all = get_stats(retornos, meta)
    excel_bytes = export_excel(res, stats_all, df_bt, meta)
    st.download_button(
        label="📥 Descargar Excel (portafolio + métricas + backtesting)",
        data=excel_bytes,
        file_name=f"portafolio_{res['label'].lower().replace(' ','_')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.divider()

    # ── Frontera eficiente ──
    st.subheader("Frontera eficiente")
    fe = run_frontera(retornos)
    if fe is not None and not fe.empty:
        fig_fe = go.Figure()
        fig_fe.add_trace(go.Scatter(
            x=fe["vol_anual"], y=fe["ret_anual"],
            mode="lines", line=dict(color="white", width=2),
            name="Frontera eficiente",
        ))
        for p_name, p_cfg in PERFILES.items():
            r_p = run_optimization(retornos, meta, p_name)
            if r_p:
                fig_fe.add_trace(go.Scatter(
                    x=[r_p["vol_anual"]], y=[r_p["ret_anual"]],
                    mode="markers+text",
                    marker=dict(size=14, color=p_cfg["color"],
                                symbol="circle", line=dict(color="white", width=1)),
                    text=[p_cfg["label"]], textposition="top center",
                    name=p_cfg["label"],
                ))
        fig_fe.update_layout(height=450,
                             xaxis_title="Volatilidad Anual", yaxis_title="Retorno Anual",
                             xaxis_tickformat=".1%", yaxis_tickformat=".1%",
                             hovermode="closest", **LAYOUT_DARK)
        st.plotly_chart(fig_fe, use_container_width=True)

    st.divider()

    # ── HRP vs Markowitz ──
    st.subheader("Comparación: Markowitz vs HRP (Hierarchical Risk Parity)")
    st.caption("HRP (López de Prado 2016) no invierte la matriz de covarianza — más robusto con pocos datos")

    hrp = run_hrp(retornos, meta)
    if hrp:
        col1, col2 = st.columns(2)
        metricas = ["Sharpe", "Retorno Anual", "Volatilidad", "Max Drawdown", "N° activos"]
        mk_vals = [fmt_n(res["sharpe"]), fmt_pct(res["ret_anual"]),
                   fmt_pct(res["vol_anual"]), fmt_pct(max_drawdown(ret_p)), res["n_activos"]]
        hrp_ret_p, _, _ = _port_returns(retornos, hrp["composicion"])
        hrp_vals = [fmt_n(hrp["sharpe"]), fmt_pct(hrp["ret_anual"]),
                    fmt_pct(hrp["vol_anual"]), fmt_pct(max_drawdown(hrp_ret_p)), hrp["n_activos"]]

        with col1:
            st.markdown(f"**{label} (Markowitz)**")
            for m, v in zip(metricas, mk_vals):
                st.metric(m, v)
        with col2:
            st.markdown("**HRP**")
            for m, v, v_mk in zip(metricas[:-1], hrp_vals[:-1], mk_vals[:-1]):
                try:
                    delta = float(hrp_vals[metricas.index(m)].replace("%","").replace("—","0")) \
                          - float(mk_vals[metricas.index(m)].replace("%","").replace("—","0"))
                except Exception:
                    delta = None
                st.metric(m, v)
            st.metric(metricas[-1], hrp_vals[-1])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — REGÍMENES
# ─────────────────────────────────────────────────────────────────────────────

def tab_regimenes(retornos, meta):
    st.header("🌊 Regímenes de mercado")
    st.caption("Hidden Markov Model (HMM) · Selección automática de estados por BIC")

    estados, n_reg, params, bic_scores = run_hmm(retornos, meta)

    # ── Régimen actual ──
    reg_actual  = int(estados.iloc[-1])
    p_actual    = params[reg_actual]
    COLORES_REG = {0: "#e74c3c", 1: "#f1c40f", 2: "#2ecc71", 3: "#00b4d8"}

    st.subheader(f"Régimen actual: **{p_actual['etiqueta']}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Retorno esperado (anual)",    fmt_pct(p_actual["media_anual"]))
    c2.metric("Volatilidad esperada (anual)", fmt_pct(p_actual["vol_anual"]))
    c3.metric("Frecuencia histórica",         fmt_pct(p_actual["frecuencia"]))

    st.divider()

    # ── Timeline ──
    st.subheader("Evolución temporal de regímenes")

    fondos_ag = [f for f in retornos.columns
                 if f in meta.index and meta.loc[f, "perfil"] == "agresivo"]
    if not fondos_ag:
        fondos_ag = list(retornos.columns[:5])

    ret_mkt = retornos[fondos_ag].mean(axis=1).dropna()
    cum_mkt = (1 + ret_mkt).cumprod() * 100

    fig = go.Figure()
    for reg, start, end in _regime_spans(estados):
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=COLORES_REG.get(reg, "#888"),
            opacity=0.18, line_width=0,
        )

    fig.add_trace(go.Scatter(
        x=cum_mkt.index, y=cum_mkt.values,
        line=dict(color="white", width=2),
        name="Mercado (fondos agresivos)",
        hovertemplate="%{x|%b %Y}<br>Índice: %{y:.1f}<extra></extra>",
    ))
    for reg, p in params.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color=COLORES_REG.get(reg, "#888"), symbol="square"),
            name=p["etiqueta"],
        ))

    fig.update_layout(height=420, yaxis_title="Índice (base 100)",
                      hovermode="x unified", **LAYOUT_DARK,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    col_l, col_r = st.columns(2)

    # ── Tabla de parámetros ──
    with col_l:
        st.subheader("Parámetros por régimen")
        df_reg = pd.DataFrame([{
            "Régimen":         p["etiqueta"],
            "Ret. Anual":      fmt_pct(p["media_anual"]),
            "Vol. Anual":      fmt_pct(p["vol_anual"]),
            "Frecuencia":      fmt_pct(p["frecuencia"]),
        } for p in params.values()])
        st.dataframe(df_reg, use_container_width=True)

    # ── BIC ──
    with col_r:
        if bic_scores:
            st.subheader("Selección de regímenes (BIC)")
            fig_b = px.bar(
                x=list(bic_scores.keys()),
                y=list(bic_scores.values()),
                labels={"x": "N° de regímenes", "y": "BIC"},
                color=list(bic_scores.values()),
                color_continuous_scale="RdYlGn_r",
                height=280,
            )
            fig_b.update_layout(showlegend=False, coloraxis_showscale=False, **LAYOUT_DARK)
            st.plotly_chart(fig_b, use_container_width=True)
            st.caption("Menor BIC = mejor balance entre ajuste y parsimonia del modelo")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — PROYECCIÓN
# ─────────────────────────────────────────────────────────────────────────────

def tab_proyeccion(retornos, meta, perfil, monto, horizonte, aporte, crecimiento_aporte=0.0):
    st.header("🚀 Proyección de cartera")

    res = run_optimization(retornos, meta, perfil)
    if res is None:
        st.warning("Primero optimiza un portafolio válido.")
        return

    fondos_c = [f for f in res["composicion"] if f in retornos.columns]
    pesos    = {f: res["composicion"][f] for f in fondos_c}
    total    = sum(pesos.values())
    pesos    = {f: v / total for f, v in pesos.items()}

    n_meses     = horizonte * 12
    pesos_tuple = tuple(sorted(pesos.items()))

    bs, mc = run_bootstrap(retornos, pesos_tuple, monto, n_meses,
                           aporte_mensual=aporte, crecimiento_anual=crecimiento_aporte)

    # ── Título (sin $ en subheader — Streamlit lo interpreta como LaTeX) ──
    capital_acum_arr = _capital_acum(monto, aporte, crecimiento_aporte, n_meses)
    total_aportado   = capital_acum_arr[-1]
    aporte_str = f" · Aporte mensual: CLP {aporte:,.0f}" if aporte > 0 else ""
    crec_str   = f" · Crece {crecimiento_aporte:.0%}/año" if aporte > 0 and crecimiento_aporte > 0 else ""
    st.subheader(f"Proyección a {horizonte} años · {res['label']}")
    st.caption(
        f"Capital inicial: CLP {monto:,.0f}{aporte_str}{crec_str} · "
        f"Total aportado al final: CLP {total_aportado:,.0f}"
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Capital inicial",     fmt_clp(monto))
    c2.metric("Total aportado",      fmt_clp(total_aportado),
              help=f"Capital inicial + aportes crecientes ({crecimiento_aporte:.0%}/año)"
                   if crecimiento_aporte > 0 else
                   f"Capital inicial + {n_meses} aportes de {fmt_clp(aporte)}")
    c3.metric("Bootstrap — P5",      fmt_clp(bs["p5"][-1]),
              delta=fmt_clp(bs["p5"][-1] - total_aportado),
              help="Escenario pesimista: 5% de trayectorias terminan por debajo")
    c4.metric("Bootstrap — Mediana", fmt_clp(bs["mediana"]),
              delta=fmt_clp(bs["mediana"] - total_aportado))
    c5.metric("Bootstrap — P95",     fmt_clp(bs["p95"][-1]),
              delta=fmt_clp(bs["p95"][-1] - total_aportado),
              help="Escenario optimista: 95% de trayectorias terminan por debajo")
    c6.metric("MC Normal — Mediana", fmt_clp(mc["mediana"]),
              delta=fmt_clp(mc["mediana"] - total_aportado))

    st.divider()

    # ── Fan chart Bootstrap vs MC ──
    st.subheader("Bootstrap de bloques vs Monte Carlo Normal")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(bs["fechas"]) + list(bs["fechas"])[::-1],
        y=list(bs["p95"]) + list(bs["p5"])[::-1],
        fill="toself", fillcolor="rgba(255,215,0,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Bootstrap P5–P95",
    ))
    fig.add_trace(go.Scatter(
        x=bs["fechas"], y=bs["p50"],
        line=dict(color="#FFD700", width=2.5),
        name="Bootstrap (mediana)",
        hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=list(mc["fechas"]) + list(mc["fechas"])[::-1],
        y=list(mc["p95"]) + list(mc["p5"])[::-1],
        fill="toself", fillcolor="rgba(0,180,216,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="MC Normal P5–P95",
    ))
    fig.add_trace(go.Scatter(
        x=mc["fechas"], y=mc["p50"],
        line=dict(color="#00b4d8", width=2, dash="dash"),
        name="MC Normal (mediana)",
        hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>",
    ))
    # Línea de capital aportado acumulado: crece con la tasa definida
    fig.add_trace(go.Scatter(
        x=bs["fechas"], y=capital_acum_arr,
        line=dict(color="#aaaaaa", width=1.5, dash="dot"),
        name="Capital aportado acumulado",
        hovertemplate="%{x|%b %Y}<br>Aportado: $%{y:,.0f}<extra></extra>",
    ))

    fig.update_layout(height=500, hovermode="x unified",
                      yaxis_title="Valor cartera (CLP)",
                      yaxis_tickprefix="$", yaxis_tickformat=",",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02),
                      **LAYOUT_DARK)
    st.plotly_chart(fig, use_container_width=True)

    # ── Tabla Bootstrap vs MC ──
    df_metodos = pd.DataFrame([
        {"Método": "Bootstrap (bloques)",
         "P5": fmt_clp(bs["p5"][-1]), "P25": fmt_clp(bs["p25"][-1]),
         "Mediana": fmt_clp(bs["mediana"]),
         "P75": fmt_clp(bs["p75"][-1]), "P95": fmt_clp(bs["p95"][-1]),
         "Std": fmt_clp(bs["std"])},
        {"Método": "Monte Carlo Normal",
         "P5": fmt_clp(mc["p5"][-1]), "P25": fmt_clp(mc["p25"][-1]),
         "Mediana": fmt_clp(mc["mediana"]),
         "P75": fmt_clp(mc["p75"][-1]), "P95": fmt_clp(mc["p95"][-1]),
         "Std": fmt_clp(mc["std"])},
    ])
    st.dataframe(df_metodos, use_container_width=True, hide_index=True)
    st.caption(
        "Bootstrap preserva autocorrelación y eventos extremos del historial real. "
        "Monte Carlo Normal asume retornos i.i.d. gaussianos — subestima colas pesadas."
    )

    st.divider()

    # ── Comparación de perfiles ──
    st.subheader("Comparación de escenarios por perfil (Bootstrap — medianas)")
    st.caption("Mismos parámetros: capital inicial, aporte mensual y horizonte. Método: Bootstrap.")

    PERFILES_COMP = [
        ("conservador", "#2ecc71"),
        ("moderado",    "#f1c40f"),
        ("agresivo",    "#e74c3c"),
        ("optimo",      "#FFD700"),
        ("sp500",       "#00b4d8"),
    ]

    fig2    = go.Figure()
    rows_tb = []

    for p_name, color in PERFILES_COMP:
        res_p = run_optimization(retornos, meta, p_name)
        if res_p is None:
            continue
        f_p = [f for f in res_p["composicion"] if f in retornos.columns]
        if not f_p:
            continue
        pw  = {f: res_p["composicion"][f] for f in f_p}
        pw  = {f: v / sum(pw.values()) for f, v in pw.items()}
        pt  = tuple(sorted(pw.items()))
        bs_p, _ = run_bootstrap(retornos, pt, monto, n_meses,
                                aporte_mensual=aporte, crecimiento_anual=crecimiento_aporte)

        label_p = res_p["label"]

        # Banda P5-P95
        fig2.add_trace(go.Scatter(
            x=list(bs_p["fechas"]) + list(bs_p["fechas"])[::-1],
            y=list(bs_p["p95"]) + list(bs_p["p5"])[::-1],
            fill="toself",
            fillcolor=_hex_rgba(color, 0.10),
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        ))
        # Mediana
        dash = "dash" if p_name == "sp500" else "solid"
        fig2.add_trace(go.Scatter(
            x=bs_p["fechas"], y=bs_p["p50"],
            line=dict(color=color, width=2, dash=dash),
            name=label_p,
            hovertemplate=f"{label_p}<br>%{{x|%b %Y}}<br>$%{{y:,.0f}}<extra></extra>",
        ))

        rows_tb.append({
            "Perfil":       label_p,
            "Sharpe":       fmt_n(res_p["sharpe"]),
            "Ret. anual":   fmt_pct(res_p["ret_anual"]),
            "Vol. anual":   fmt_pct(res_p["vol_anual"]),
            "P5 final":     fmt_clp(bs_p["p5"][-1]),
            "Mediana":      fmt_clp(bs_p["mediana"]),
            "P95 final":    fmt_clp(bs_p["p95"][-1]),
            "Ganancia P50": fmt_clp(bs_p["mediana"] - total_aportado),
        })

    fig2.add_trace(go.Scatter(
        x=bs["fechas"], y=capital_acum_arr,
        line=dict(color="#aaaaaa", width=1.5, dash="dot"),
        name="Capital aportado acumulado",
        hovertemplate="%{x|%b %Y}<br>Aportado: $%{y:,.0f}<extra></extra>",
        showlegend=True,
    ))
    fig2.update_layout(height=520, hovermode="x unified",
                       yaxis_title="Valor cartera (CLP)",
                       yaxis_tickprefix="$", yaxis_tickformat=",",
                       legend=dict(orientation="h", yanchor="bottom", y=1.02),
                       **LAYOUT_DARK)
    st.plotly_chart(fig2, use_container_width=True)

    if rows_tb:
        st.dataframe(pd.DataFrame(rows_tb), use_container_width=True, hide_index=True)

    st.divider()

    # ── Estacionalidad ──
    st.subheader("Estacionalidad — Retorno promedio mensual por perfil")
    st.caption("¿Hay meses sistemáticamente mejores o peores? Promedio histórico 2020–2026.")

    from src.metrics import estacionalidad_por_mes
    df_est = estacionalidad_por_mes(retornos, meta)

    if not df_est.empty:
        # Heatmap perfiles × meses
        fig_est = px.imshow(
            df_est.T,
            color_continuous_scale="RdYlGn",
            zmin=-2, zmax=2,
            text_auto=".2f",
            labels={"color": "Ret. medio (%)"},
            aspect="auto",
            height=280,
        )
        fig_est.update_layout(**LAYOUT_DARK,
                              coloraxis_colorbar=dict(title="Ret. %"))
        fig_est.update_traces(textfont_size=10)
        st.plotly_chart(fig_est, use_container_width=True)

        # Gráfico de líneas por perfil
        fig_est2 = go.Figure()
        meses_ord = ["Ene","Feb","Mar","Abr","May","Jun",
                     "Jul","Ago","Sep","Oct","Nov","Dic"]
        for col in df_est.columns:
            serie = df_est[col].reindex(meses_ord)
            fig_est2.add_trace(go.Bar(
                x=meses_ord,
                y=serie.values,
                name=col.capitalize(),
                marker_color=[COLORES.get(col, "#888")] * 12,
                opacity=0.85,
            ))
        fig_est2.update_layout(
            height=340, barmode="group",
            yaxis_title="Retorno promedio (%)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            **LAYOUT_DARK,
        )
        st.plotly_chart(fig_est2, use_container_width=True)
        st.caption(
            "Retornos en %. Los meses con retornos consistentemente negativos "
            "pueden indicar patrones de liquidaciones estacionales."
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — BACKTESTING
# ─────────────────────────────────────────────────────────────────────────────

def tab_backtesting(retornos, meta, perfil):
    st.header("🔄 Validación Walk-Forward")
    st.caption(
        "Walk-forward con ventana deslizante: optimiza en 70% (in-sample), "
        "evalúa en 30% (out-of-sample). Detecta overfitting."
    )

    df = run_backtest(retornos, meta, perfil)

    if df is None or df.empty:
        st.warning("Datos insuficientes para backtesting.")
        return

    deg = degradacion_sharpe(df)

    # ── KPIs ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sharpe IS promedio",   fmt_n(deg.get("sharpe_is_promedio", 0)))
    c2.metric("Sharpe OOS promedio",  fmt_n(deg.get("sharpe_oos_promedio", 0)))
    degrad = deg.get("degradacion_promedio", 0)
    c3.metric("Degradación OOS−IS",   f"{degrad:+.3f}",
              help="Positivo = OOS > IS = sin overfitting")
    c4.metric("Splits OOS > 0",       fmt_pct(deg.get("pct_splits_positivos", 0)))

    if degrad >= 0:
        st.success("✅ Sin overfitting: el portafolio generaliza fuera de muestra.")
    else:
        st.warning("⚠️ Degradación negativa: posible sobreajuste en la muestra.")

    st.divider()

    # ── Gráfico IS vs OOS ──
    st.subheader("Sharpe In-Sample vs Out-of-Sample por split")

    df_p = df.dropna(subset=["sharpe_insample", "sharpe_outsample"])
    if not df_p.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_p["periodo_test"], y=df_p["sharpe_insample"],
                             name="In-Sample", marker_color="#3498db"))
        fig.add_trace(go.Bar(x=df_p["periodo_test"], y=df_p["sharpe_outsample"],
                             name="Out-of-Sample", marker_color="#2ecc71"))
        fig.add_hline(y=0, line_dash="dot", line_color="#e74c3c",
                      annotation_text="Sharpe = 0")
        fig.update_layout(height=400, barmode="group", yaxis_title="Sharpe Ratio",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02),
                          **LAYOUT_DARK)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tabla ──
    st.subheader("Resultados por split")
    df_show = df[["periodo_test", "sharpe_insample", "sharpe_outsample",
                  "ret_anual_test", "vol_anual_test",
                  "max_drawdown_test", "n_activos"]].copy()
    df_show.columns = ["Período", "Sharpe IS", "Sharpe OOS",
                       "Ret. Anual", "Vol. Anual", "Max DD", "N° activos"]
    for c in ["Ret. Anual", "Vol. Anual", "Max DD"]:
        df_show[c] = df_show[c].apply(lambda x: fmt_pct(x) if pd.notna(x) else "—")
    for c in ["Sharpe IS", "Sharpe OOS"]:
        df_show[c] = df_show[c].apply(lambda x: fmt_n(x) if pd.notna(x) else "—")
    st.dataframe(df_show, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 — ANÁLISIS DINÁMICO (GARCH + ROLLING)
# ─────────────────────────────────────────────────────────────────────────────

def tab_dinamico(retornos, meta, perfil, monto):
    st.header("📉 Análisis dinámico")
    st.caption(
        "Volatilidad condicional GARCH(1,1) · Métricas rodantes · "
        "Las correlaciones y el riesgo NO son constantes en el tiempo."
    )

    # ── Volatilidad rodante por perfil ──
    st.subheader("Volatilidad anualizada rodante (ventana 12 meses)")
    vol_df = run_rolling(retornos, meta)

    fig_v = go.Figure()
    for col in vol_df.columns:
        serie = vol_df[col].dropna()
        fig_v.add_trace(go.Scatter(
            x=serie.index, y=serie.values,
            name=col.capitalize(),
            line=dict(color=COLORES.get(col, "#888"), width=2),
            hovertemplate="%{x|%b %Y}<br>Vol: %{y:.2%}<extra></extra>",
        ))
    fig_v.update_layout(height=380, hovermode="x unified",
                        yaxis_tickformat=".1%", yaxis_title="Volatilidad anual",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        **LAYOUT_DARK)
    st.plotly_chart(fig_v, use_container_width=True)

    st.divider()

    # ── GARCH por perfil ──
    st.subheader("GARCH(1,1) — Volatilidad condicional vs histórica")
    st.caption("alpha = reacción a shocks · beta = persistencia · alpha+beta < 1 garantiza estacionariedad")

    garch_res = run_garch(retornos, meta)

    if garch_res:
        # Tabla de parámetros
        rows_g = []
        for p_name, g in garch_res.items():
            rows_g.append({
                "Perfil":            p_name.capitalize(),
                "α (ARCH)":          fmt_n(g["alpha"]),
                "β (GARCH)":         fmt_n(g["beta"]),
                "Persistencia α+β":  fmt_n(g["persistencia"]),
                "Vol. incondicional":fmt_pct(g["vol_incondicional"]),
            })
        st.dataframe(pd.DataFrame(rows_g), use_container_width=True, hide_index=True)
        st.caption("Persistencia > 0.95 → shocks de volatilidad duran meses (típico mercados emergentes)")

        st.divider()

        # Gráfico de volatilidad condicional GARCH
        fig_g = go.Figure()
        for p_name, g in garch_res.items():
            vc = g.get("vol_condicional")
            if vc is None or vc.empty:
                continue
            fig_g.add_trace(go.Scatter(
                x=vc.index, y=vc.values,
                name=p_name.capitalize(),
                line=dict(color=COLORES.get(p_name, "#888"), width=2),
                hovertemplate="%{x|%b %Y}<br>σ GARCH: %{y:.2%}<extra></extra>",
            ))
        fig_g.update_layout(height=380, hovermode="x unified",
                            yaxis_tickformat=".1%", yaxis_title="Volatilidad condicional GARCH (anual)",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02),
                            **LAYOUT_DARK)
        st.plotly_chart(fig_g, use_container_width=True)

    st.divider()

    # ── Sharpe rodante del portafolio seleccionado ──
    st.subheader(f"Sharpe Ratio rodante del portafolio — {perfil.capitalize()} (ventana 18 meses)")
    res = run_optimization(retornos, meta, perfil)
    if res:
        fondos_c = [f for f in res["composicion"] if f in retornos.columns]
        pesos    = {f: res["composicion"][f] for f in fondos_c}
        sh_roll  = sharpe_rodante(retornos, pesos, ventana=18)

        fig_sh = go.Figure()
        fig_sh.add_trace(go.Scatter(
            x=sh_roll.index, y=sh_roll.values,
            fill="tozeroy",
            fillcolor=_hex_rgba(COLORES.get(perfil, "#FFD700"), 0.10),
            line=dict(color=COLORES.get(perfil, "#FFD700"), width=2),
            hovertemplate="%{x|%b %Y}<br>Sharpe: %{y:.3f}<extra></extra>",
        ))
        fig_sh.add_hline(y=0, line_dash="dot", line_color="#e74c3c")
        fig_sh.add_hline(y=0.5, line_dash="dot", line_color="#2ecc71",
                         annotation_text="Sharpe = 0.5")
        fig_sh.update_layout(height=350, yaxis_title="Sharpe Ratio rodante",
                             hovermode="x unified", **LAYOUT_DARK)
        st.plotly_chart(fig_sh, use_container_width=True)

    st.divider()

    # ── Beta rodante vs SP500 ──
    if "SP500" in retornos.columns and res:
        st.subheader("Beta rodante del portafolio vs S&P 500 (ventana 12 meses)")
        ret_p   = pd.Series(
            retornos[fondos_c].dropna().values @ np.array(
                [pesos[f] / sum(pesos.values()) for f in fondos_c]),
            index=retornos[fondos_c].dropna().index,
        )
        ret_sp  = retornos["SP500"].dropna()
        beta_r  = beta_rodante(ret_p, ret_sp, ventana=12)

        fig_b = go.Figure()
        fig_b.add_trace(go.Scatter(
            x=beta_r.index, y=beta_r.values,
            line=dict(color="#00b4d8", width=2),
            fill="tozeroy", fillcolor="rgba(0,180,216,0.08)",
            hovertemplate="%{x|%b %Y}<br>Beta: %{y:.3f}<extra></extra>",
        ))
        fig_b.add_hline(y=1, line_dash="dot", line_color="#f1c40f",
                        annotation_text="Beta = 1 (mismo riesgo que SP500)")
        fig_b.add_hline(y=0, line_dash="dot", line_color="#e74c3c")
        fig_b.update_layout(height=350, yaxis_title="Beta vs S&P 500",
                            hovermode="x unified", **LAYOUT_DARK)
        st.plotly_chart(fig_b, use_container_width=True)
        st.caption(
            "Beta > 1: el portafolio amplifica los movimientos del SP500. "
            "Beta < 1: amortigua. Beta < 0: cobertura natural."
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 8 — STRESS TEST HIPOTÉTICO
# ─────────────────────────────────────────────────────────────────────────────

def tab_stress(retornos, meta, monto):
    st.header("⚡ Stress Test hipotético")
    st.caption(
        "Impacto estimado de shocks que NO ocurrieron pero PODRÍAN ocurrir. "
        "Metodología: sensibilidades (beta) históricas × magnitud del shock."
    )

    df_stress = run_stress(retornos, meta, monto)
    if df_stress is None or df_stress.empty:
        st.warning("No hay portafolios para estressar. Activa más brokers.")
        return

    # Columnas de impacto en % (sin las de CLP)
    cols_pct = [c for c in df_stress.columns
                if c not in ["Escenario", "Descripcion"] and "$CLP" not in c]

    # ── Heatmap de impacto ──
    st.subheader("Heatmap de impacto por escenario y portafolio (%)")

    df_heat = df_stress[["Escenario"] + cols_pct].set_index("Escenario")
    df_heat = df_heat.astype(float)

    fig_h = px.imshow(
        df_heat,
        color_continuous_scale="RdYlGn",
        zmin=-0.35, zmax=0.35,
        text_auto=".1%",
        aspect="auto",
        height=420,
        labels={"color": "Impacto"},
    )
    fig_h.update_layout(**LAYOUT_DARK, coloraxis_colorbar=dict(tickformat=".0%"))
    fig_h.update_traces(textfont_size=11)
    st.plotly_chart(fig_h, use_container_width=True)

    st.divider()

    # ── Barras por escenario ──
    st.subheader("Impacto en CLP por escenario")

    cols_clp = [c for c in df_stress.columns if "$CLP" in c]
    escenario_sel = st.selectbox("Escenario", df_stress["Escenario"].tolist())
    row = df_stress[df_stress["Escenario"] == escenario_sel].iloc[0]

    st.caption(row.get("Descripcion", ""))

    portafolios_sel = [c.replace(" ($CLP)", "") for c in cols_clp]
    vals_clp        = [float(row[c]) for c in cols_clp]
    colors_bar      = ["#2ecc71" if v >= 0 else "#e74c3c" for v in vals_clp]

    fig_bar = go.Figure(go.Bar(
        x=portafolios_sel,
        y=vals_clp,
        marker_color=colors_bar,
        text=[fmt_clp(v) for v in vals_clp],
        textposition="outside",
        hovertemplate="%{x}<br>%{y:,.0f} CLP<extra></extra>",
    ))
    fig_bar.add_hline(y=0, line_color="white", line_width=0.5)
    fig_bar.update_layout(height=380, yaxis_title="Impacto (CLP)",
                          yaxis_tickprefix="$", yaxis_tickformat=",",
                          **LAYOUT_DARK)
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Tabla completa ──
    with st.expander("Ver tabla completa de impactos (%)"):
        df_show = df_stress[["Escenario", "Descripcion"] + cols_pct].copy()
        for c in cols_pct:
            df_show[c] = df_show[c].apply(lambda x: fmt_pct(float(x)) if pd.notna(x) else "—")
        st.dataframe(df_show, use_container_width=True, hide_index=True)

    st.caption(
        "Sensibilidades estimadas por regresión histórica OLS sobre datos 2020–2026. "
        "Los shocks son hipotéticos — no predicen eventos futuros."
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 9 — CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def tab_clustering(retornos, meta):
    st.header("🔬 Clustering de fondos")
    st.caption(
        "La clasificación oficial CMF (conservador/moderado/agresivo) "
        "no refleja el comportamiento real. K-Means agrupa fondos por "
        "similitud estadística: retorno, volatilidad, skewness, drawdown y autocorrelación."
    )

    labels, k_opt, silhouettes, features, linkage, corr, resumen = run_clustering(retornos, meta)

    # ── K-Means scatter ──
    st.subheader(f"K-Means — {k_opt} clusters óptimos (silhouette score)")

    CLUSTER_COLORS = px.colors.qualitative.Set2

    # Añadir metadata al DataFrame de features
    feat_plot = features.copy()
    feat_plot["cluster"]   = labels.reindex(feat_plot.index).astype(str)
    feat_plot["nombre"]    = feat_plot.index.map(
        lambda f: meta.loc[f, "nombre"][:25] if f in meta.index else f)
    feat_plot["perfil_cmf"] = feat_plot.index.map(
        lambda f: meta.loc[f, "perfil"] if f in meta.index else "?")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Retorno vs Volatilidad — coloreado por cluster**")
        fig_km = px.scatter(
            feat_plot,
            x="volatilidad_anual", y="retorno_anual",
            color="cluster",
            hover_name="nombre",
            symbol="perfil_cmf",
            color_discrete_sequence=CLUSTER_COLORS,
            labels={"volatilidad_anual": "Volatilidad Anual",
                    "retorno_anual": "Retorno Anual",
                    "cluster": "Cluster K-Means",
                    "perfil_cmf": "Perfil CMF"},
            height=450,
        )
        fig_km.update_traces(marker=dict(size=10, opacity=0.85,
                                         line=dict(color="white", width=0.5)))
        fig_km.update_layout(xaxis_tickformat=".1%", yaxis_tickformat=".1%",
                              **LAYOUT_DARK)
        st.plotly_chart(fig_km, use_container_width=True)

    with col_r:
        st.markdown("**Silhouette score por k — selección automática**")
        if silhouettes:
            fig_sil = px.bar(
                x=list(silhouettes.keys()),
                y=list(silhouettes.values()),
                labels={"x": "Número de clusters k", "y": "Silhouette score"},
                color=list(silhouettes.values()),
                color_continuous_scale="YlGn",
                height=240,
            )
            fig_sil.update_layout(showlegend=False, coloraxis_showscale=False,
                                   **LAYOUT_DARK)
            st.plotly_chart(fig_sil, use_container_width=True)
            st.caption(f"k óptimo = **{k_opt}** (mayor silhouette = clusters más compactos y separados)")

        st.markdown("**Resumen por cluster**")
        df_res = resumen[["cluster", "n_fondos", "retorno_anual",
                           "volatilidad_anual", "perfil_dominante"]].copy()
        df_res["retorno_anual"]     = df_res["retorno_anual"].apply(fmt_pct)
        df_res["volatilidad_anual"] = df_res["volatilidad_anual"].apply(fmt_pct)
        df_res.columns = ["Cluster", "N°", "Ret. Anual", "Vol. Anual", "Perfil dominante"]
        st.dataframe(df_res, use_container_width=True, hide_index=True)

    st.divider()

    # ── Dendrograma jerárquico ──
    st.subheader("Clustering jerárquico — Distancia de Mantegna (1999)")
    st.caption(
        "d(i,j) = √(0.5·(1−ρᵢⱼ)) satisface la desigualdad triangular. "
        "Fondos cercanos en el dendrograma se comportan de forma similar."
    )

    fondos_dend = list(corr.columns)
    nombres_dend = [meta.loc[f, "nombre"][:20] if f in meta.index else f
                    for f in fondos_dend]

    try:
        fig_dend = ff.create_dendrogram(
            corr.values,
            orientation="bottom",
            labels=nombres_dend,
            colorscale=px.colors.qualitative.Set2,
            linkagefun=lambda x: linkage,
        )
        fig_dend.update_layout(height=500, xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
                                **LAYOUT_DARK)
        st.plotly_chart(fig_dend, use_container_width=True)
    except Exception:
        # Fallback: heatmap de correlación ordenado por clusters
        st.info("Mostrando heatmap de correlación ordenado por clusters (fallback del dendrograma).")
        orden = labels.sort_values().index.tolist()
        corr_ord = corr.loc[orden, orden]
        nombres_ord = [meta.loc[f, "nombre"][:15] if f in meta.index else f for f in orden]
        corr_ord.index   = nombres_ord
        corr_ord.columns = nombres_ord
        fig_corr = px.imshow(corr_ord, color_continuous_scale="RdBu_r",
                              zmin=-1, zmax=1, height=500)
        fig_corr.update_layout(**LAYOUT_DARK)
        st.plotly_chart(fig_corr, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIMULADORES — CACHED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Simulando estrategias de rebalanceo…")
def run_rebalanceo_hist(retornos, pesos_tuple, monto, estrategia, umbral=0.05):
    """Simula 4 estrategias de rebalanceo sobre datos históricos reales."""
    fondos   = [f for f, _ in pesos_tuple]
    w_target = np.array([w for _, w in pesos_tuple])
    w_target = w_target / w_target.sum()
    R        = retornos[fondos].dropna()
    valores  = monto * w_target
    historia = [monto]
    n_reb    = 0

    for t, (_, row) in enumerate(R.iterrows()):
        valores = valores * (1 + row.values)
        total   = valores.sum()
        historia.append(total)
        if estrategia == "mensual":
            valores = total * w_target; n_reb += 1
        elif estrategia == "anual" and (t + 1) % 12 == 0:
            valores = total * w_target; n_reb += 1
        elif estrategia == "umbral":
            if np.max(np.abs(valores / total - w_target)) > umbral:
                valores = total * w_target; n_reb += 1
        # buy_hold: sin acción

    serie   = pd.Series(historia[1:], index=R.index)
    ret_arr = serie.pct_change().dropna().values
    return {
        "historia":      serie,
        "n_rebalanceos": n_reb,
        "sharpe":        sharpe_ratio(ret_arr),
        "max_dd":        max_drawdown(ret_arr),
        "valor_final":   historia[-1],
        "ret_total":     historia[-1] / monto - 1,
    }


@st.cache_data(show_spinner="Simulando retiro periódico…")
def run_bootstrap_retiro(retornos, pesos_tuple, capital, retiro_mensual, n_meses,
                         crecimiento_anual=0.0, n_sim=600):
    """Bootstrap con retiro mensual (negativo). El retiro puede crecer con inflación."""
    pesos = dict(pesos_tuple)
    return proyectar_bootstrap(retornos, pesos, capital, n_meses,
                               n_sim=n_sim, aporte_mensual=-abs(retiro_mensual),
                               crecimiento_anual=crecimiento_anual)


# ─────────────────────────────────────────────────────────────────────────────
# SIMULADORES — FUNCIONES DE CADA SIM
# ─────────────────────────────────────────────────────────────────────────────

def _pesos_perfil(retornos, meta, perfil):
    """Devuelve (pesos_dict, pesos_tuple, fondos_validos) para un perfil."""
    res = run_optimization(retornos, meta, perfil)
    if res is None:
        return None, None, None
    fondos = [f for f in res["composicion"] if f in retornos.columns]
    pw     = {f: res["composicion"][f] for f in fondos}
    total  = sum(pw.values())
    pw     = {f: v / total for f, v in pw.items()}
    return pw, tuple(sorted(pw.items())), fondos


# ── Sim 1: Meta financiera ────────────────────────────────────────────────────

def sim_meta(retornos, meta, monto, aporte, crecimiento_aporte=0.0):
    st.subheader("🎯 Simulador de meta financiera")
    st.caption("¿Cuánto necesito ahorrar cada mes para llegar a mi objetivo?")

    col1, col2, col3 = st.columns(3)
    meta_clp  = col1.number_input("Meta financiera (CLP)", min_value=1_000_000,
                                   max_value=10_000_000_000, value=100_000_000,
                                   step=5_000_000)
    horizonte = col2.slider("Horizonte (años)", 1, 30, 10)
    perfil_m  = col3.radio("Perfil de inversión",
                            ["conservador", "moderado", "agresivo", "optimo"],
                            format_func=lambda x: {
                                "conservador":"🟢 Conservador","moderado":"🟡 Moderado",
                                "agresivo":"🔴 Agresivo","optimo":"⭐ Óptimo",}[x],
                            horizontal=True)

    res = run_optimization(retornos, meta, perfil_m)
    if res is None:
        st.warning("No se pudo optimizar el perfil seleccionado.")
        return

    r_m = res["ret_anual"] / 12   # retorno mensual estimado
    n   = horizonte * 12

    # Aporte necesario (fórmula de anualidad futura)
    if abs(r_m) > 1e-6:
        factor      = (1 + r_m) ** n
        aporte_req  = (meta_clp - monto * factor) * r_m / (factor - 1)
    else:
        aporte_req  = (meta_clp - monto) / n

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Capital inicial", f"CLP {monto:,.0f}")
    c2.metric("Meta", f"CLP {meta_clp:,.0f}")
    c3.metric("Aporte mensual necesario",
              f"CLP {max(aporte_req, 0):,.0f}",
              help="Basado en el retorno esperado promedio del portafolio")
    c4.metric("Retorno esperado anual", fmt_pct(res["ret_anual"]))

    if aporte_req <= 0:
        st.success(f"✅ Con el capital inicial de CLP {monto:,.0f} ya alcanzas la meta "
                   f"en {horizonte} años sin aportes adicionales.")

    st.divider()

    # Bootstrap: P(alcanzar meta) con el aporte actual del sidebar
    pesos, pt, _ = _pesos_perfil(retornos, meta, perfil_m)
    if pt is None:
        return

    bs_meta, _ = run_bootstrap(retornos, pt, monto, n,
                               aporte_mensual=aporte, crecimiento_anual=crecimiento_aporte)
    prob_meta   = float((bs_meta["simulaciones"][:, -1] >= meta_clp).mean())
    mediana_fin = bs_meta["mediana"]

    c1, c2, c3 = st.columns(3)
    crec_help = f" (crece {crecimiento_aporte:.0%}/año)" if crecimiento_aporte > 0 else ""
    c1.metric("P(alcanzar meta) con aporte actual",
              fmt_pct(prob_meta),
              help=f"Con aporte mensual actual de CLP {aporte:,.0f}{crec_help}")
    c2.metric("Mediana proyectada al final", f"CLP {mediana_fin:,.0f}")
    c3.metric("Brecha mediana vs meta",
              f"CLP {mediana_fin - meta_clp:,.0f}",
              delta=f"CLP {mediana_fin - meta_clp:,.0f}")

    # Fan chart con línea de meta
    capital_acum = _capital_acum(monto, aporte, crecimiento_aporte, n)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(bs_meta["fechas"]) + list(bs_meta["fechas"])[::-1],
        y=list(bs_meta["p95"]) + list(bs_meta["p5"])[::-1],
        fill="toself", fillcolor=_hex_rgba(COLORES.get(perfil_m, "#FFD700"), 0.12),
        line=dict(color="rgba(0,0,0,0)"), name="P5–P95", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=bs_meta["fechas"], y=bs_meta["p50"],
        line=dict(color=COLORES.get(perfil_m, "#FFD700"), width=2.5),
        name="Mediana proyectada",
        hovertemplate="%{x|%b %Y}<br>CLP %{y:,.0f}<extra></extra>",
    ))
    fig.add_hline(y=meta_clp, line_dash="solid", line_color="#2ecc71", line_width=2,
                  annotation_text=f"Meta: CLP {meta_clp:,.0f}",
                  annotation_font_color="#2ecc71")
    fig.add_trace(go.Scatter(
        x=bs_meta["fechas"], y=capital_acum,
        line=dict(color="#aaa", width=1.5, dash="dot"),
        name="Capital aportado acumulado",
    ))
    fig.update_layout(height=480, hovermode="x unified",
                      yaxis_title="Valor cartera (CLP)",
                      yaxis_tickformat=",",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02),
                      **LAYOUT_DARK)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Simulación Bootstrap de bloques · {bs_meta['n_sim']} trayectorias · "
        f"Perfil {perfil_m} · Retorno esperado {fmt_pct(res['ret_anual'])} anual"
    )


# ── Sim 2: Rebalanceo ─────────────────────────────────────────────────────────

def sim_rebalanceo(retornos, meta, monto):
    st.subheader("⚖️ Simulador de rebalanceo")
    st.caption(
        "Compara cuánto impacta en el resultado final la frecuencia con que "
        "reequilibras tu portafolio a los pesos objetivo."
    )

    col1, col2 = st.columns(2)
    perfil_r = col1.radio("Perfil", ["conservador","moderado","agresivo","optimo"],
                           format_func=lambda x: {"conservador":"🟢 Conservador",
                               "moderado":"🟡 Moderado","agresivo":"🔴 Agresivo",
                               "optimo":"⭐ Óptimo"}[x],
                           horizontal=True)
    umbral_pct = col2.slider("Umbral de desviación (%)", 2, 20, 5,
                              help="Para estrategia 'por umbral': rebalancea cuando "
                                   "algún activo se desvía más del X% de su peso objetivo")

    pesos, pt, _ = _pesos_perfil(retornos, meta, perfil_r)
    if pt is None:
        st.warning("No se pudo optimizar."); return

    ESTRATEGIAS = {
        "buy_hold": ("Buy & Hold",   "#e74c3c"),
        "anual":    ("Anual",        "#f1c40f"),
        "umbral":   (f"Umbral {umbral_pct}%", "#00b4d8"),
        "mensual":  ("Mensual",      "#2ecc71"),
    }

    resultados = {}
    for key in ESTRATEGIAS:
        resultados[key] = run_rebalanceo_hist(
            retornos, pt, monto, key, umbral=umbral_pct/100
        )

    # Gráfico histórico
    fig = go.Figure()
    for key, (label, color) in ESTRATEGIAS.items():
        r = resultados[key]
        dash = "dot" if key == "buy_hold" else "solid"
        fig.add_trace(go.Scatter(
            x=r["historia"].index, y=r["historia"].values,
            name=label, line=dict(color=color, width=2, dash=dash),
            hovertemplate=f"{label}<br>%{{x|%b %Y}}<br>CLP %{{y:,.0f}}<extra></extra>",
        ))
    fig.add_hline(y=monto, line_dash="dot", line_color="#555",
                  annotation_text="Capital inicial")
    fig.update_layout(height=440, hovermode="x unified",
                      yaxis_title="Valor cartera (CLP)", yaxis_tickformat=",",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02),
                      **LAYOUT_DARK)
    st.plotly_chart(fig, use_container_width=True)

    # Tabla comparativa
    rows = []
    for key, (label, _) in ESTRATEGIAS.items():
        r = resultados[key]
        rows.append({
            "Estrategia":     label,
            "Valor final":    f"CLP {r['valor_final']:,.0f}",
            "Retorno total":  fmt_pct(r["ret_total"]),
            "Sharpe":         fmt_n(r["sharpe"]),
            "Max Drawdown":   fmt_pct(r["max_dd"]),
            "N° rebalanceos": r["n_rebalanceos"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "Datos históricos reales 2020–2026. "
        "El rebalanceo frecuente mantiene el riesgo controlado pero puede incurrir en costos."
    )


# ── Sim 3: ¿Qué habría pasado? ────────────────────────────────────────────────

def sim_historico(retornos, meta, monto):
    st.subheader("📅 ¿Qué habría pasado si hubieras invertido en una fecha histórica?")
    st.caption("Usa retornos reales — sin bootstrap. Exactamente lo que habría ocurrido.")

    fechas_disp = [str(f.date()) for f in retornos.index]
    col1, col2 = st.columns(2)
    fecha_sel  = col1.selectbox("Fecha de entrada", fechas_disp,
                                 index=0,
                                 help="Elige la fecha en que habrías invertido")
    perfil_h   = col2.radio("Perfil", ["conservador","moderado","agresivo","optimo"],
                             format_func=lambda x: {"conservador":"🟢","moderado":"🟡",
                                 "agresivo":"🔴","optimo":"⭐"}[x] + f" {x.capitalize()}",
                             horizontal=True)

    fecha_dt  = pd.to_datetime(fecha_sel)
    ret_slice = retornos[retornos.index >= fecha_dt]

    if ret_slice.empty:
        st.warning("No hay datos desde esa fecha."); return

    pesos, _, fondos = _pesos_perfil(retornos, meta, perfil_h)
    if pesos is None:
        st.warning("No se pudo optimizar."); return

    # Simular cartera histórica real
    hist = simular_historico(ret_slice, pesos, monto)
    ret_arr  = hist.pct_change().dropna().values
    ret_acum = hist.iloc[-1] / monto - 1
    mdd      = max_drawdown(ret_arr)
    sh       = sharpe_ratio(ret_arr)

    # Calcular fecha de recuperación tras el peor drawdown
    rolling_max = hist.expanding().max()
    dd_serie    = (hist - rolling_max) / rolling_max
    idx_min_dd  = dd_serie.idxmin()
    post_min    = hist[hist.index > idx_min_dd]
    recuperado  = post_min[post_min >= rolling_max[idx_min_dd]]
    fecha_rec   = recuperado.index[0] if not recuperado.empty else None

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Capital inicial",  f"CLP {monto:,.0f}")
    c2.metric("Valor actual",     f"CLP {hist.iloc[-1]:,.0f}")
    c3.metric("Retorno acumulado",fmt_pct(ret_acum))
    c4.metric("Max Drawdown",     fmt_pct(mdd))
    c5.metric("Sharpe",           fmt_n(sh))

    if fecha_rec:
        st.info(f"📍 Peor caída: {idx_min_dd.strftime('%b %Y')} ({fmt_pct(dd_serie.min())}) "
                f"— Recuperación: {fecha_rec.strftime('%b %Y')} "
                f"({int((fecha_rec - idx_min_dd).days / 30)} meses)")
    else:
        st.warning(f"⚠️ La cartera aún no ha recuperado su máximo tras la caída de "
                   f"{idx_min_dd.strftime('%b %Y')}")

    # Gráfico cartera + drawdown
    col_l, col_r = st.columns([3, 1])
    with col_l:
        fig = go.Figure()
        # SP500 como referencia
        if "SP500" in ret_slice.columns:
            hist_sp = simular_historico(ret_slice, {"SP500": 1.0}, monto)
            fig.add_trace(go.Scatter(
                x=hist_sp.index, y=hist_sp.values,
                line=dict(color=COLORES["sp500"], width=1.5, dash="dash"),
                name="S&P 500 (referencia)",
                hovertemplate="%{x|%b %Y}<br>CLP %{y:,.0f}<extra></extra>",
            ))
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist.values,
            fill="tozeroy", fillcolor=_hex_rgba(COLORES.get(perfil_h, "#FFD700"), 0.08),
            line=dict(color=COLORES.get(perfil_h, "#FFD700"), width=2.5),
            name=f"Portafolio {perfil_h}",
            hovertemplate="%{x|%b %Y}<br>CLP %{y:,.0f}<extra></extra>",
        ))
        fig.add_hline(y=monto, line_dash="dot", line_color="#555",
                      annotation_text="Capital inicial")
        fig.update_layout(height=400, hovermode="x unified",
                          yaxis_title="Valor cartera (CLP)", yaxis_tickformat=",",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02),
                          **LAYOUT_DARK)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("**Drawdown**")
        fig_dd = go.Figure(go.Scatter(
            x=dd_serie.index, y=dd_serie.values * 100,
            fill="tozeroy", fillcolor="rgba(231,76,60,0.2)",
            line=dict(color="#e74c3c", width=1.5),
            hovertemplate="%{x|%b %Y}<br>DD: %{y:.1f}%<extra></extra>",
        ))
        fig_dd.update_layout(height=400, yaxis_title="Drawdown (%)",
                             **LAYOUT_DARK)
        st.plotly_chart(fig_dd, use_container_width=True)


# ── Sim 4: Retiro periódico ───────────────────────────────────────────────────

def sim_retiro(retornos, meta, monto):
    st.subheader("💸 Simulador de retiro periódico")
    st.caption("¿Cuánto tiempo dura tu capital si retiras una cantidad fija cada mes?")

    col1, col2, col3, col4 = st.columns(4)
    capital   = col1.number_input("Capital acumulado (CLP)", min_value=1_000_000,
                                   max_value=5_000_000_000, value=monto,
                                   step=1_000_000)
    retiro    = col2.number_input("Retiro mensual (CLP)", min_value=10_000,
                                   max_value=50_000_000, value=300_000,
                                   step=50_000)
    crec_retiro_pct = col3.slider("Crecimiento anual del retiro", min_value=0,
                                   max_value=10, value=3, step=1, format="%d%%",
                                   help="El retiro mensual crece esta tasa cada año "
                                        "(ajuste por inflación)")
    crec_retiro = crec_retiro_pct / 100

    perfil_ret = col4.radio("Perfil",
                             ["conservador","moderado","agresivo","optimo","sp500"],
                             format_func=lambda x: {
                                 "conservador":"🟢 Conservador",
                                 "moderado":   "🟡 Moderado",
                                 "agresivo":   "🔴 Agresivo",
                                 "optimo":     "⭐ Óptimo",
                                 "sp500":      "📈 S&P 500",
                             }[x],
                             horizontal=True)

    horizonte_ret = 30   # máximo 30 años
    n_meses_ret   = horizonte_ret * 12

    pesos, pt, _ = _pesos_perfil(retornos, meta, perfil_ret)
    if pt is None:
        st.warning("No se pudo optimizar."); return

    bs_ret = run_bootstrap_retiro(retornos, pt, capital, retiro, n_meses_ret,
                                   crecimiento_anual=crec_retiro)

    # Probabilidad de supervivencia por año
    sims = bs_ret["simulaciones"]   # shape (n_sim, n_meses)
    prob_survival = []
    fechas_anual  = []
    for yr in range(1, horizonte_ret + 1):
        idx = min(yr * 12 - 1, sims.shape[1] - 1)
        prob_survival.append(float((sims[:, idx] > 0).mean()))
        fechas_anual.append(bs_ret["fechas"][idx])

    # Año estimado de agotamiento (donde prob < 50%)
    anios_sob = [yr for yr, p in enumerate(prob_survival, 1) if p >= 0.5]
    anio_50   = max(anios_sob) if anios_sob else 0

    # Retiro máximo sostenible (P50 positivo en 20 años)
    r_m = run_optimization(retornos, meta, perfil_ret)
    if r_m:
        rm = r_m["ret_anual"] / 12
        n  = 20 * 12
        if abs(rm) > 1e-6:
            retiro_max = capital * rm * (1 + rm)**n / ((1 + rm)**n - 1)
        else:
            retiro_max = capital / n
    else:
        retiro_max = capital / (20 * 12)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Capital inicial",          f"CLP {capital:,.0f}")
    c2.metric("Retiro mensual",           f"CLP {retiro:,.0f}")
    c3.metric("Capital dura > 50% prob.", f"{anio_50} años" if anio_50 else "< 1 año")
    c4.metric("Retiro máx. sostenible (20 años)", f"CLP {retiro_max:,.0f}",
              help="Retiro que agota exactamente el capital en 20 años (fórmula de anualidad)")

    st.divider()

    # Curva de supervivencia
    fig_surv = go.Figure()
    fig_surv.add_trace(go.Scatter(
        x=list(range(1, horizonte_ret + 1)), y=[p * 100 for p in prob_survival],
        fill="tozeroy", fillcolor="rgba(46,204,113,0.15)",
        line=dict(color="#2ecc71", width=2.5),
        hovertemplate="Año %{x}<br>Prob. supervivencia: %{y:.1f}%<extra></extra>",
        name="Probabilidad de supervivencia",
    ))
    fig_surv.add_hline(y=50, line_dash="dash", line_color="#f1c40f",
                       annotation_text="50% probabilidad")
    fig_surv.add_hline(y=90, line_dash="dot", line_color="#2ecc71",
                       annotation_text="90% probabilidad")
    fig_surv.update_layout(height=360,
                            xaxis_title="Años desde el retiro",
                            yaxis_title="Probabilidad de supervivencia (%)",
                            yaxis=dict(range=[0, 105]),
                            **LAYOUT_DARK)
    st.plotly_chart(fig_surv, use_container_width=True)

    # Fan chart de trayectorias
    st.subheader("Evolución del capital (fan chart)")
    fig_fan = go.Figure()
    fig_fan.add_trace(go.Scatter(
        x=list(bs_ret["fechas"]) + list(bs_ret["fechas"])[::-1],
        y=list(bs_ret["p95"]) + list(bs_ret["p5"])[::-1],
        fill="toself", fillcolor=_hex_rgba(COLORES.get(perfil_ret, "#2ecc71"), 0.10),
        line=dict(color="rgba(0,0,0,0)"), name="P5–P95",
    ))
    fig_fan.add_trace(go.Scatter(
        x=bs_ret["fechas"], y=bs_ret["p50"],
        line=dict(color=COLORES.get(perfil_ret, "#2ecc71"), width=2.5),
        name="Mediana",
        hovertemplate="%{x|%b %Y}<br>CLP %{y:,.0f}<extra></extra>",
    ))
    fig_fan.add_hline(y=0, line_dash="solid", line_color="#e74c3c",
                      annotation_text="Capital agotado")
    fig_fan.update_layout(height=420, hovermode="x unified",
                           yaxis_title="Capital restante (CLP)", yaxis_tickformat=",",
                           legend=dict(orientation="h", yanchor="bottom", y=1.02),
                           **LAYOUT_DARK)
    st.plotly_chart(fig_fan, use_container_width=True)
    st.caption(
        f"Bootstrap {bs_ret['n_sim']} simulaciones · Retiro mensual de CLP {retiro:,.0f} · "
        f"Capital se agota cuando llega a 0."
    )


# ── Sim 5: Cambio de perfil ───────────────────────────────────────────────────

def sim_cambio_perfil(retornos, meta, monto, horizonte, aporte, crecimiento_aporte=0.0):
    st.subheader("🔀 Simulador de cambio de perfil")
    st.caption(
        "¿Qué pasa si empiezas conservador y te vuelves más agresivo con los años? "
        "Compara 3 estrategias: quedarte en el perfil A, quedarte en B, o hacer el cambio."
    )

    opciones = {
        "conservador": "🟢 Conservador",
        "moderado":    "🟡 Moderado",
        "agresivo":    "🔴 Agresivo",
        "optimo":      "⭐ Óptimo",
    }
    col1, col2, col3 = st.columns(3)
    perfil_a  = col1.selectbox("Perfil fase A (inicio)", list(opciones.keys()),
                                format_func=lambda x: opciones[x], index=0)
    perfil_b  = col2.selectbox("Perfil fase B (después)", list(opciones.keys()),
                                format_func=lambda x: opciones[x], index=2)
    cambio_yr = col3.slider("Cambiar perfil en el año…", 1, horizonte - 1,
                             max(1, horizonte // 2))

    n_total = horizonte * 12
    n_a     = cambio_yr * 12
    n_b     = n_total - n_a

    pesos_a, pt_a, _ = _pesos_perfil(retornos, meta, perfil_a)
    pesos_b, pt_b, _ = _pesos_perfil(retornos, meta, perfil_b)
    if pt_a is None or pt_b is None:
        st.warning("No se pudo optimizar uno de los perfiles."); return

    res_a = run_optimization(retornos, meta, perfil_a)
    res_b = run_optimization(retornos, meta, perfil_b)

    # Bootstrap para cada estrategia pura
    bs_a, _ = run_bootstrap(retornos, pt_a, monto, n_total,
                            aporte_mensual=aporte, crecimiento_anual=crecimiento_aporte)
    bs_b, _ = run_bootstrap(retornos, pt_b, monto, n_total,
                            aporte_mensual=aporte, crecimiento_anual=crecimiento_aporte)

    # Estrategia mixta: fase A hasta el año de cambio, luego fase B desde la mediana
    bs_a_corto, _ = run_bootstrap(retornos, pt_a, monto, n_a,
                                   aporte_mensual=aporte, crecimiento_anual=crecimiento_aporte)
    capital_cambio = float(bs_a_corto["mediana"])
    # En fase B el aporte ya lleva n_a meses de crecimiento
    tasa_m_c = (1 + crecimiento_aporte) ** (1 / 12) - 1
    aporte_en_cambio = aporte * (1 + tasa_m_c) ** n_a
    bs_b_largo, _  = run_bootstrap(retornos, pt_b, capital_cambio, n_b,
                                    aporte_mensual=aporte_en_cambio,
                                    crecimiento_anual=crecimiento_aporte)

    # KPIs finales
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Solo {opciones[perfil_a]} — Mediana final",
              f"CLP {bs_a['mediana']:,.0f}")
    c2.metric(f"Cambio {opciones[perfil_a]}→{opciones[perfil_b]} — Mediana",
              f"CLP {bs_b_largo['mediana']:,.0f}",
              delta=f"CLP {bs_b_largo['mediana'] - bs_a['mediana']:,.0f} vs solo A")
    c3.metric(f"Solo {opciones[perfil_b]} — Mediana final",
              f"CLP {bs_b['mediana']:,.0f}")

    st.divider()

    # Fan chart combinado
    fig = go.Figure()
    color_a   = COLORES.get(perfil_a, "#2ecc71")
    color_b   = COLORES.get(perfil_b, "#e74c3c")
    color_mix = "#9b59b6"

    fechas_a = list(bs_a["fechas"])
    fechas_b = list(bs_b["fechas"])
    fechas_mix = list(bs_a_corto["fechas"]) + list(bs_b_largo["fechas"])
    p50_mix    = list(bs_a_corto["p50"])    + list(bs_b_largo["p50"])
    p5_mix     = list(bs_a_corto["p5"])     + list(bs_b_largo["p5"])
    p95_mix    = list(bs_a_corto["p95"])    + list(bs_b_largo["p95"])

    for fechas, p5, p50, p95, color, label in [
        (fechas_a, bs_a["p5"], bs_a["p50"], bs_a["p95"], color_a,   f"Solo {perfil_a}"),
        (fechas_b, bs_b["p5"], bs_b["p50"], bs_b["p95"], color_b,   f"Solo {perfil_b}"),
        (fechas_mix, p5_mix,   p50_mix,     p95_mix,     color_mix, f"Cambio año {cambio_yr}"),
    ]:
        fig.add_trace(go.Scatter(
            x=fechas + fechas[::-1],
            y=list(p95) + list(p5)[::-1],
            fill="toself", fillcolor=_hex_rgba(color, 0.10),
            line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=fechas, y=list(p50),
            line=dict(color=color, width=2.5),
            name=label,
            hovertemplate=f"{label}<br>%{{x|%b %Y}}<br>CLP %{{y:,.0f}}<extra></extra>",
        ))

    # Línea vertical en el momento del cambio
    if bs_a_corto["fechas"] is not None and len(bs_a_corto["fechas"]) > 0:
        fecha_cambio = bs_a_corto["fechas"][-1]
        fig.add_shape(type="line",
                      x0=fecha_cambio, x1=fecha_cambio, y0=0, y1=1,
                      xref="x", yref="paper",
                      line=dict(color="#aaaaaa", width=1.5, dash="dash"))
        fig.add_annotation(x=fecha_cambio, y=0.97, xref="x", yref="paper",
                           text=f"Cambio → {perfil_b}", showarrow=False,
                           yanchor="top", font=dict(color="#aaaaaa", size=11))

    capital_acum = _capital_acum(monto, aporte, crecimiento_aporte, n_total)
    fig.add_trace(go.Scatter(
        x=fechas_b, y=capital_acum,
        line=dict(color="#888", width=1, dash="dot"),
        name="Capital aportado acumulado",
    ))

    fig.update_layout(height=500, hovermode="x unified",
                      yaxis_title="Valor cartera (CLP)", yaxis_tickformat=",",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02),
                      **LAYOUT_DARK)
    st.plotly_chart(fig, use_container_width=True)

    # Tabla resumen
    st.dataframe(pd.DataFrame([
        {"Estrategia": f"Solo {perfil_a.capitalize()}",
         "Sharpe":     fmt_n(res_a["sharpe"]) if res_a else "—",
         "Ret. anual": fmt_pct(res_a["ret_anual"]) if res_a else "—",
         "Mediana final": f"CLP {bs_a['mediana']:,.0f}",
         "P5 final":   f"CLP {bs_a['p5'][-1]:,.0f}",
         "P95 final":  f"CLP {bs_a['p95'][-1]:,.0f}"},
        {"Estrategia": f"Cambio año {cambio_yr}: {perfil_a}→{perfil_b}",
         "Sharpe":     "—",
         "Ret. anual": "—",
         "Mediana final": f"CLP {bs_b_largo['mediana']:,.0f}",
         "P5 final":   f"CLP {bs_b_largo['p5'][-1]:,.0f}",
         "P95 final":  f"CLP {bs_b_largo['p95'][-1]:,.0f}"},
        {"Estrategia": f"Solo {perfil_b.capitalize()}",
         "Sharpe":     fmt_n(res_b["sharpe"]) if res_b else "—",
         "Ret. anual": fmt_pct(res_b["ret_anual"]) if res_b else "—",
         "Mediana final": f"CLP {bs_b['mediana']:,.0f}",
         "P5 final":   f"CLP {bs_b['p5'][-1]:,.0f}",
         "P95 final":  f"CLP {bs_b['p95'][-1]:,.0f}"},
    ]), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIM 6: INDEPENDENCIA FINANCIERA
# ─────────────────────────────────────────────────────────────────────────────

def _retiro_max_analitico(capital, r_m_ret, n_retiro, crec_ret=0.0):
    """Retiro mensual que agota exactamente el capital en n_retiro meses."""
    tasa_real = max(r_m_ret - crec_ret / 12, 1e-8)
    if abs(tasa_real) > 1e-8:
        return capital * tasa_real / (1 - (1 + tasa_real) ** (-n_retiro))
    return capital / max(n_retiro, 1)


def sim_independencia(retornos, meta, monto, aporte, crecimiento_aporte):
    st.subheader("🏝️ Independencia Financiera")
    st.caption(
        "Dos fases encadenadas: **Acumulación** → **Retiro**. "
        "¿Cuánto necesitas invertir hoy para vivir de tus inversiones?"
    )

    PERFILES_LABELS = {
        "conservador": "🟢 Conservador",
        "moderado":    "🟡 Moderado",
        "agresivo":    "🔴 Agresivo",
        "optimo":      "⭐ Óptimo",
        "sp500":       "📈 S&P 500",
    }

    # ── Configuración Fase 1 ──────────────────────────────────────────────────
    with st.expander("⚙️ Fase 1 — Acumulación", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        capital_ini  = c1.number_input("Capital inicial (CLP)", min_value=0,
                                        max_value=5_000_000_000, value=int(monto),
                                        step=1_000_000)
        aporte_ac    = c2.number_input("Aporte mensual (CLP)", min_value=0,
                                        max_value=50_000_000, value=int(aporte),
                                        step=100_000)
        crec_ac_pct  = c3.slider("Crecimiento aporte/año", 0, 10,
                                  int(round(crecimiento_aporte * 100)),
                                  format="%d%%",
                                  help="El aporte crece esta tasa cada año")
        años_acum    = c4.slider("Años de acumulación", 1, 40, 20)

        perfil_acum  = st.radio("Perfil durante acumulación",
                                 list(PERFILES_LABELS.keys()),
                                 format_func=lambda x: PERFILES_LABELS[x],
                                 index=2, horizontal=True,
                                 key="ind_perfil_acum")

    # ── Configuración Fase 2 ──────────────────────────────────────────────────
    with st.expander("⚙️ Fase 2 — Retiro", expanded=True):
        c1, c2, c3 = st.columns(3)
        retiro_mensual = c1.number_input("Retiro mensual deseado (CLP)",
                                          min_value=10_000, max_value=100_000_000,
                                          value=500_000, step=50_000)
        crec_ret_pct   = c2.slider("Crecimiento retiro/año", 0, 10, 3,
                                    format="%d%%",
                                    help="Ajuste por inflación del retiro mensual")
        años_retiro    = c3.slider("Años de retiro", 5, 50, 30,
                                    help="50 años ≈ a perpetuidad")

        perfil_ret = st.radio("Perfil durante retiro",
                               list(PERFILES_LABELS.keys()),
                               format_func=lambda x: PERFILES_LABELS[x],
                               index=0, horizontal=True,
                               key="ind_perfil_ret")

    crec_ac  = crec_ac_pct / 100
    crec_ret = crec_ret_pct / 100
    n_acum   = años_acum * 12
    n_retiro = años_retiro * 12

    # ── Optimización de portafolios ───────────────────────────────────────────
    pesos_ac,  pt_ac,  _ = _pesos_perfil(retornos, meta, perfil_acum)
    pesos_ret, pt_ret, _ = _pesos_perfil(retornos, meta, perfil_ret)
    if pt_ac is None or pt_ret is None:
        st.warning("No se pudo optimizar uno de los perfiles."); return

    res_ac  = run_optimization(retornos, meta, perfil_acum)
    res_ret = run_optimization(retornos, meta, perfil_ret)
    r_m_ac  = res_ac["ret_anual"]  / 12 if res_ac  else 0.005
    r_m_ret = res_ret["ret_anual"] / 12 if res_ret else 0.005

    # ── Bootstrap Fase 1 ─────────────────────────────────────────────────────
    with st.spinner("Simulando fase de acumulación…"):
        bs_acum, _ = run_bootstrap(retornos, pt_ac, capital_ini, n_acum,
                                   aporte_mensual=aporte_ac, crecimiento_anual=crec_ac)

    capital_p5  = float(bs_acum["p5"][-1])
    capital_p25 = float(bs_acum["p25"][-1])
    capital_p50 = float(bs_acum["p50"][-1])
    capital_p75 = float(bs_acum["p75"][-1])
    capital_p95 = float(bs_acum["p95"][-1])

    # ── Back-solve: capital necesario para el retiro deseado ─────────────────
    # Anualidad presente ajustada por crecimiento del retiro
    tasa_real_ret = max(r_m_ret - crec_ret / 12, 1e-8)
    if abs(tasa_real_ret) > 1e-8:
        capital_necesario = retiro_mensual * (1 - (1 + tasa_real_ret) ** (-n_retiro)) / tasa_real_ret
    else:
        capital_necesario = retiro_mensual * n_retiro

    # Aporte mensual necesario para acumular capital_necesario en n_acum meses
    factor_ac = (1 + r_m_ac) ** n_acum
    if abs(r_m_ac) > 1e-6 and factor_ac > 1:
        aporte_nec = max((capital_necesario - capital_ini * factor_ac) * r_m_ac / (factor_ac - 1), 0)
    else:
        aporte_nec = max((capital_necesario - capital_ini) / max(n_acum, 1), 0)

    # ── Bootstrap Fase 2 (desde capital P50 de acumulación) ──────────────────
    with st.spinner("Simulando fase de retiro…"):
        bs_retiro = run_bootstrap_retiro(retornos, pt_ret, capital_p50,
                                         retiro_mensual, n_retiro,
                                         crecimiento_anual=crec_ret)

    sims_ret = bs_retiro["simulaciones"]   # (n_sim, n_retiro)

    # Curva de supervivencia anual
    prob_surv = []
    for yr in range(1, años_retiro + 1):
        idx_yr = min(yr * 12 - 1, sims_ret.shape[1] - 1)
        prob_surv.append(float((sims_ret[:, idx_yr] > 0).mean()))

    prob_total = prob_surv[-1]

    # Año en que la mediana P50 se agota
    p50_ret_arr  = list(bs_retiro["p50"])
    idx_agot_p50 = next((i for i, v in enumerate(p50_ret_arr) if v <= 0), None)
    años_dur_p50 = (idx_agot_p50 / 12) if idx_agot_p50 is not None else años_retiro

    # Retiros máximos sostenibles analíticos por escenario de capital
    ret_max_p5  = _retiro_max_analitico(capital_p5,  r_m_ret, n_retiro, crec_ret)
    ret_max_p50 = _retiro_max_analitico(capital_p50, r_m_ret, n_retiro, crec_ret)
    ret_max_p95 = _retiro_max_analitico(capital_p95, r_m_ret, n_retiro, crec_ret)
    ret_4pct    = capital_p50 * 0.04 / 12
    ret_3pct    = capital_p50 * 0.03 / 12

    # ── KPIs ─────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("##### 📈 Fase 1: Acumulación")
    total_ap = _capital_acum(capital_ini, aporte_ac, crec_ac, n_acum)[-1]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Capital inicial",         f"CLP {capital_ini:,.0f}")
    c2.metric("Total aportado",          f"CLP {total_ap:,.0f}",
              help=f"Capital inicial + aportes en {años_acum} años (creciendo {crec_ac:.0%}/año)")
    c3.metric("Capital proyectado P5",   f"CLP {capital_p5:,.0f}",
              delta=f"CLP {capital_p5 - total_ap:,.0f}")
    c4.metric("Capital proyectado P50",  f"CLP {capital_p50:,.0f}",
              delta=f"CLP {capital_p50 - total_ap:,.0f}")
    c5.metric("Capital proyectado P95",  f"CLP {capital_p95:,.0f}",
              delta=f"CLP {capital_p95 - total_ap:,.0f}")

    st.markdown("##### 🔄 Back-solve: ¿Cuánto necesitas?")
    alcanza = capital_p50 >= capital_necesario
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Capital necesario (analítico)",    f"CLP {capital_necesario:,.0f}",
              help=f"Capital requerido para sostener CLP {retiro_mensual:,.0f}/mes "
                   f"durante {años_retiro} años (crece {crec_ret:.0%}/año)")
    c2.metric("Aporte mensual necesario",         f"CLP {aporte_nec:,.0f}",
              delta=f"CLP {aporte_nec - aporte_ac:,.0f} vs actual" if aporte_ac > 0 else None,
              help="Aporte fijo necesario para acumular el capital requerido")
    c3.metric("¿Capital P50 alcanza la meta?",
              "✅ Sí" if alcanza else "❌ No",
              delta=f"CLP {capital_p50 - capital_necesario:,.0f}")
    c4.metric("Brecha P50 vs necesario",
              f"CLP {abs(capital_p50 - capital_necesario):,.0f}",
              delta="Superávit" if alcanza else "Déficit",
              delta_color="normal" if alcanza else "inverse")

    st.markdown("##### 💸 Fase 2: Retiro")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Retiro deseado/mes",             f"CLP {retiro_mensual:,.0f}")
    c2.metric("Retiro máx. sostenible (P50)",   f"CLP {ret_max_p50:,.0f}",
              help=f"Agota exactamente el capital P50 en {años_retiro} años")
    c3.metric("Regla del 4%/año (P50)",         f"CLP {ret_4pct:,.0f}",
              help="Regla empírica clásica: retira 4% anual del capital inicial")
    c4.metric("Regla del 3%/año (P50)",         f"CLP {ret_3pct:,.0f}",
              help="Versión conservadora: 3% anual (mayor protección contra inflación)")
    c5.metric("P(capital sobrevive retiro)",    f"{prob_total:.1%}",
              help=f"Prob. de que el capital dure {años_retiro} años con retiro deseado (bootstrap)")
    c6.metric("Capital P50 dura",
              f"{años_dur_p50:.1f} años" if idx_agot_p50 is not None else f">{años_retiro} años",
              delta="✅ Todo el horizonte" if idx_agot_p50 is None else "⚠️ Se agota antes")

    st.divider()

    # ── Fan chart encadenado: Acumulación + Retiro ────────────────────────────
    st.subheader("Fan chart completo: Acumulación → Retiro")

    fechas_acum = list(bs_acum["fechas"])
    fecha_ini_ret = pd.Timestamp(fechas_acum[-1]) + pd.DateOffset(months=1)
    fechas_ret = list(pd.date_range(start=fecha_ini_ret, periods=n_retiro, freq="MS"))

    color_acum = COLORES.get(perfil_acum, "#FFD700")
    color_ret  = COLORES.get(perfil_ret,  "#e74c3c")

    fig = go.Figure()

    # Bandas de acumulación
    fig.add_trace(go.Scatter(
        x=fechas_acum + fechas_acum[::-1],
        y=list(bs_acum["p95"]) + list(bs_acum["p5"])[::-1],
        fill="toself", fillcolor=_hex_rgba(color_acum, 0.12),
        line=dict(color="rgba(0,0,0,0)"), name="Acumulación P5–P95",
    ))
    fig.add_trace(go.Scatter(
        x=fechas_acum, y=list(bs_acum["p50"]),
        line=dict(color=color_acum, width=2.5),
        name=f"Acumulación P50 ({PERFILES_LABELS[perfil_acum]})",
        hovertemplate="%{x|%b %Y}<br>CLP %{y:,.0f}<extra></extra>",
    ))

    # Bandas de retiro
    fig.add_trace(go.Scatter(
        x=fechas_ret + fechas_ret[::-1],
        y=list(bs_retiro["p95"]) + list(bs_retiro["p5"])[::-1],
        fill="toself", fillcolor=_hex_rgba(color_ret, 0.12),
        line=dict(color="rgba(0,0,0,0)"), name="Retiro P5–P95",
    ))
    fig.add_trace(go.Scatter(
        x=fechas_ret, y=list(bs_retiro["p50"]),
        line=dict(color=color_ret, width=2.5),
        name=f"Retiro P50 ({PERFILES_LABELS[perfil_ret]})",
        hovertemplate="%{x|%b %Y}<br>CLP %{y:,.0f}<extra></extra>",
    ))

    # Curva de capital aportado acumulado
    cap_acum_curve = _capital_acum(capital_ini, aporte_ac, crec_ac, n_acum)
    fig.add_trace(go.Scatter(
        x=fechas_acum, y=cap_acum_curve,
        line=dict(color="#aaaaaa", width=1, dash="dot"),
        name="Capital aportado acumulado",
        hovertemplate="%{x|%b %Y}<br>Aportado: CLP %{y:,.0f}<extra></extra>",
    ))

    # Línea horizontal en capital_necesario
    fig.add_hline(y=capital_necesario, line_dash="dot", line_color="#f1c40f",
                  annotation_text=f"Capital necesario: CLP {capital_necesario:,.0f}",
                  annotation_font_color="#f1c40f")
    fig.add_hline(y=0, line_dash="solid", line_color="#e74c3c", line_width=1,
                  annotation_text="Capital agotado")

    # Línea vertical separando fases
    fecha_sep = fechas_acum[-1]
    fig.add_shape(type="line",
                  x0=fecha_sep, x1=fecha_sep, y0=0, y1=1,
                  xref="x", yref="paper",
                  line=dict(color="white", width=1.5, dash="dash"))
    fig.add_annotation(x=fecha_sep, y=0.97, xref="x", yref="paper",
                       text=f"Inicio retiro (año {años_acum})", showarrow=False,
                       yanchor="top", font=dict(color="white", size=11))

    fig.update_layout(height=560, hovermode="x unified",
                      yaxis_title="Capital (CLP)", yaxis_tickformat=",",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02),
                      **LAYOUT_DARK)
    st.plotly_chart(fig, use_container_width=True)

    # ── Curva de supervivencia + tabla de escenarios ──────────────────────────
    col_surv, col_tabla = st.columns([1, 1])

    with col_surv:
        st.subheader("Curva de supervivencia del capital")
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(
            x=list(range(1, años_retiro + 1)),
            y=[p * 100 for p in prob_surv],
            fill="tozeroy", fillcolor=_hex_rgba(color_ret, 0.15),
            line=dict(color=color_ret, width=2.5),
            hovertemplate="Año %{x} de retiro<br>P(capital > 0): %{y:.1f}%<extra></extra>",
            name="Prob. supervivencia",
        ))
        fig_s.add_hline(y=90, line_dash="dot",  line_color="#2ecc71",
                        annotation_text="90%", annotation_font_color="#2ecc71")
        fig_s.add_hline(y=50, line_dash="dash", line_color="#f1c40f",
                        annotation_text="50%", annotation_font_color="#f1c40f")
        fig_s.update_layout(height=380,
                             xaxis_title="Años desde inicio del retiro",
                             yaxis_title="P(capital > 0) %",
                             yaxis=dict(range=[0, 105]),
                             **LAYOUT_DARK)
        st.plotly_chart(fig_s, use_container_width=True)

    with col_tabla:
        st.subheader("Escenarios: capital acumulado vs retiro sostenible")
        rows_sc = []
        for lbl, cap in [("P5 (pesimista)",  capital_p5),
                          ("P50 (mediana)",   capital_p50),
                          ("P95 (optimista)", capital_p95)]:
            rm  = _retiro_max_analitico(cap, r_m_ret, n_retiro, crec_ret)
            r4  = cap * 0.04 / 12
            r3  = cap * 0.03 / 12
            rows_sc.append({
                "Escenario":            lbl,
                "Capital acumulado":    f"CLP {cap:,.0f}",
                "Retiro máx. analítico":f"CLP {rm:,.0f}",
                "Regla 4%/año":         f"CLP {r4:,.0f}",
                "Regla 3%/año":         f"CLP {r3:,.0f}",
            })
        st.dataframe(pd.DataFrame(rows_sc), use_container_width=True, hide_index=True)

        st.caption(f"Probabilidad de supervivencia con retiro CLP {retiro_mensual:,.0f}/mes:")
        años_check = [y for y in range(5, años_retiro + 1, 5)]
        df_ps = pd.DataFrame({
            "Año de retiro":       años_check,
            "P(capital > 0)":      [f"{prob_surv[y - 1]:.1%}" for y in años_check],
        })
        st.dataframe(df_ps, use_container_width=True, hide_index=True)

    st.caption(
        f"Bootstrap {bs_acum['n_sim']} simulaciones · "
        f"Fase 1: {años_acum} años acumulando en perfil {perfil_acum} · "
        f"Fase 2: {años_retiro} años retirando en perfil {perfil_ret} · "
        f"Retiro crece {crec_ret:.0%}/año"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB SIMULADOR (contenedor de los 6)
# ─────────────────────────────────────────────────────────────────────────────

def tab_simulador(retornos, meta, monto, horizonte, aporte, crecimiento_aporte=0.0):
    st.header("🎮 Simuladores")
    st.caption("6 simuladores interactivos independientes del sidebar principal.")

    sim_tabs = st.tabs([
        "🎯 Meta financiera",
        "⚖️ Rebalanceo",
        "📅 ¿Qué habría pasado?",
        "💸 Retiro periódico",
        "🔀 Cambio de perfil",
        "🏝️ Independencia Financiera",
    ])
    with sim_tabs[0]: sim_meta(retornos, meta, monto, aporte, crecimiento_aporte)
    with sim_tabs[1]: sim_rebalanceo(retornos, meta, monto)
    with sim_tabs[2]: sim_historico(retornos, meta, monto)
    with sim_tabs[3]: sim_retiro(retornos, meta, monto)
    with sim_tabs[4]: sim_cambio_perfil(retornos, meta, monto, horizonte, aporte, crecimiento_aporte)
    with sim_tabs[5]: sim_independencia(retornos, meta, monto, aporte, crecimiento_aporte)


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT EXCEL
# ─────────────────────────────────────────────────────────────────────────────

def export_excel(res, stats, df_backtest, meta):
    """Genera un Excel con 3 hojas: Portafolio, Métricas, Backtesting."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # Hoja 1: Composición del portafolio
        df_port = pd.DataFrame([{
            "Fondo":     meta.loc[f, "nombre"] if f in meta.index else f,
            "Perfil":    meta.loc[f, "perfil"] if f in meta.index else "—",
            "Corredora": meta.loc[f, "corredora"] if f in meta.index else "—",
            "Peso (%)":  round(w * 100, 2),
        } for f, w in sorted(res["composicion"].items(), key=lambda x: -x[1])])
        df_port.to_excel(writer, sheet_name="Portafolio", index=False)

        # Hoja 2: Métricas de todos los fondos
        cols_export = ["nombre", "perfil", "corredora", "retorno_anual",
                       "volatilidad_anual", "sharpe", "sortino",
                       "var_95", "cvar_95", "max_drawdown", "n_meses"]
        stats[cols_export].to_excel(writer, sheet_name="Fondos", index=False)

        # Hoja 3: Backtesting
        if df_backtest is not None and not df_backtest.empty:
            df_backtest.to_excel(writer, sheet_name="Backtesting", index=False)

    buf.seek(0)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    perfil, monto, horizonte, aporte, crecimiento_aporte, brokers = render_sidebar()

    # Carga de datos
    try:
        df_long, retornos_all, precios_all, meta_all = load_data(_fingerprint=_data_fingerprint())
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        st.caption("Asegúrate de ejecutar `streamlit run app.py` desde la carpeta raíz del proyecto.")
        st.stop()

    # Filtrar por brokers seleccionados
    retornos, meta = filtrar_universo(retornos_all, meta_all, brokers)

    # Stats de fondos filtrados
    stats = get_stats(retornos, meta)

    # Header
    n_ch = len(meta[meta["corredora"] != "externo"])
    perfil_labels = {"conservador": "🟢 Conservador", "moderado": "🟡 Moderado",
                     "agresivo": "🔴 Agresivo", "optimo": "⭐ Óptimo Global"}
    st.title("🇨🇱 Chilean Mutual Funds — Portfolio Optimizer")
    st.markdown(
        "**Junwei He** · MSc Data Science (c), Universidad de Chile &nbsp;|&nbsp; "
        "[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?logo=linkedin&logoColor=white&style=flat-square)](https://www.linkedin.com/in/junwei-he-mai-96bb83131/) &nbsp;"
        "[![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white&style=flat-square)](https://github.com/JunTierSS)"
    )
    st.caption(
        f"{n_ch} fondos · {len(retornos)} meses · "
        f"Perfil: **{perfil_labels[perfil]}** · "
        f"Capital: **CLP {monto:,.0f}** · "
        f"Horizonte: **{horizonte} años**"
    )

    # Tabs
    tabs = st.tabs([
        "📊 Mercado",
        "🔍 Fondos",
        "🎯 Portafolio",
        "🌊 Regímenes",
        "🚀 Proyección",
        "🔄 Backtesting",
        "📉 Dinámico",
        "⚡ Stress Test",
        "🔬 Clustering",
        "🎮 Simulador",
        "📅 Día Óptimo",
    ])

    with tabs[0]: tab_mercado(retornos, meta, brokers)
    with tabs[1]: tab_fondos(retornos, meta, stats)
    with tabs[2]: tab_portafolio(retornos, meta, perfil, monto)
    with tabs[3]: tab_regimenes(retornos, meta)
    with tabs[4]: tab_proyeccion(retornos, meta, perfil, monto, horizonte, aporte, crecimiento_aporte)
    with tabs[5]: tab_backtesting(retornos, meta, perfil)
    with tabs[6]: tab_dinamico(retornos, meta, perfil, monto)
    with tabs[7]: tab_stress(retornos, meta, monto)
    with tabs[8]: tab_clustering(retornos, meta)
    with tabs[9]: tab_simulador(retornos, meta, monto, horizonte, aporte, crecimiento_aporte)
    with tabs[10]: tab_dia_optimo(retornos, meta, monto, aporte)


# ---------------------------------------------------------------------------
def tab_dia_optimo(retornos, meta, monto_ini, aporte_mensual):
    """Simula DCA: ¿qué día del mes conviene más invertir?"""
    import pandas as _pd_local
    st.header("📅 Día Óptimo de Inversión")
    st.caption(
        "Simula un aporte mensual fijo durante N años y calcula el capital final "
        "según qué día del mes se invierte. Requiere datos diarios."
    )

    # ── Verificar datos diarios ──────────────────────────────────────────────
    import glob as _glob

    daily_files = _glob.glob("data/**/daily/*.csv", recursive=True)
    if not daily_files:
        st.warning(
            "⚠️ No hay datos diarios descargados aún. "
            "Ejecuta en tu terminal:\n\n"
            "```bash\npython3 scripts/fetch_investing.py --interval daily --no-push\n```"
        )
        return

    # ── Cargar datos diarios ─────────────────────────────────────────────────
    @st.cache_data(show_spinner="Cargando precios diarios…")
    def _load_daily():
        import pathlib as _pl
        import pandas as pd
        base = _pl.Path(__file__).parent / "data"
        frames = []
        parse_errors = []
        all_files = list(base.glob("**/daily/*.csv"))
        for fp in all_files:
            try:
                df = pd.read_csv(fp, encoding="utf-8-sig", dtype=str)
                df.columns = df.columns.str.strip()
                col_u = next((c for c in df.columns if "ltimo" in c.lower()), None)
                if "Fecha" not in df.columns or col_u is None:
                    parse_errors.append(f"cols: {fp.name} → {df.columns.tolist()}")
                    continue
                mid = fp.stem.replace("Datos_diarios_", "")
                df = df[["Fecha", col_u]].copy()
                df.columns = ["fecha", "precio"]
                df["fecha"] = pd.to_datetime(df["fecha"], dayfirst=True, errors="coerce")
                df["precio"] = (
                    df["precio"]
                    .str.replace('"', "", regex=False)
                    .str.replace(".", "", regex=False)
                    .str.replace(",", ".", regex=False)
                    .apply(pd.to_numeric, errors="coerce")
                )
                df = df.dropna().sort_values("fecha").set_index("fecha")
                df.columns = [mid]
                frames.append(df)
            except Exception as e:
                parse_errors.append(f"{fp.name}: {e}")
        if not frames:
            return None, all_files, parse_errors
        return pd.concat(frames, axis=1).sort_index(), all_files, parse_errors

    result = _load_daily()
    precios_diarios, _daily_files_found, _daily_errors = result if isinstance(result, tuple) else (result, [], [])
    if precios_diarios is None or precios_diarios.empty:
        st.error("No se pudieron cargar los datos diarios.")
        with st.expander("Debug info"):
            st.write(f"Archivos encontrados: {len(_daily_files_found)}")
            st.write([str(f) for f in _daily_files_found[:5]])
            st.write("Errores:", _daily_errors[:5])
        return

    # ── Controles ────────────────────────────────────────────────────────────
    fondos_diarios = list(precios_diarios.columns)
    nombres_diarios = {f: meta.get(f, {}).get("nombre", f) for f in fondos_diarios}

    col1, col2, col3 = st.columns(3)
    with col1:
        fondo_sel = st.selectbox(
            "Fondo",
            fondos_diarios,
            format_func=lambda x: nombres_diarios.get(x, x),
        )
    with col2:
        aporte = st.number_input(
            "Aporte mensual (CLP)",
            min_value=10_000,
            max_value=50_000_000,
            value=int(aporte_mensual) if aporte_mensual > 0 else 100_000,
            step=10_000,
            format="%d",
        )
    with col3:
        anos = st.slider("Horizonte (años)", min_value=1, max_value=20, value=5)

    st.caption(f"CLP {aporte:,.0f}/mes · {anos} años · {12*anos} aportes")

    # ── Simulación ───────────────────────────────────────────────────────────
    serie = precios_diarios[fondo_sel].dropna()
    fecha_fin = serie.index.max()
    fecha_ini = fecha_fin - _pd_local.DateOffset(years=anos)
    serie = serie[serie.index >= fecha_ini]

    if len(serie) < 30:
        st.warning("Hay muy pocos datos diarios para este fondo. Descarga más histórico.")
        return

    resultados = {}
    for dia in range(1, 29):
        unidades = 0.0
        for anio in range(fecha_ini.year, fecha_fin.year + 1):
            for mes in range(1, 13):
                # Busca el precio en el día pedido o el siguiente día hábil
                try:
                    fecha_objetivo = _pd_local.Timestamp(year=anio, month=mes, day=dia)
                except ValueError:
                    continue
                if fecha_objetivo < fecha_ini or fecha_objetivo > fecha_fin:
                    continue
                # Precio más cercano (desde ese día hacia adelante dentro del mes)
                candidatos = serie[
                    (serie.index >= fecha_objetivo) &
                    (serie.index <= fecha_objetivo + _pd_local.Timedelta(days=7))
                ]
                if candidatos.empty:
                    continue
                precio_compra = candidatos.iloc[0]
                if precio_compra > 0:
                    unidades += aporte / precio_compra

        precio_actual = serie.iloc[-1]
        resultados[dia] = unidades * precio_actual

    if not resultados:
        st.error("No se pudo simular ningún día.")
        return

    df_res = _pd_local.DataFrame.from_dict(resultados, orient="index", columns=["capital_final"])
    df_res.index.name = "dia_mes"
    total_invertido = aporte * anos * 12
    df_res["rentabilidad_pct"] = (df_res["capital_final"] / total_invertido - 1) * 100

    dia_optimo = df_res["capital_final"].idxmax()
    dia_peor   = df_res["capital_final"].idxmin()
    diff_clp   = df_res["capital_final"].max() - df_res["capital_final"].min()
    diff_pct   = df_res["rentabilidad_pct"].max() - df_res["rentabilidad_pct"].min()

    # ── Métricas resumen ─────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mejor día", f"Día {dia_optimo}", f"CLP {df_res.loc[dia_optimo,'capital_final']:,.0f}")
    c2.metric("Peor día",  f"Día {dia_peor}",  f"CLP {df_res.loc[dia_peor,'capital_final']:,.0f}")
    c3.metric("Diferencia capital", f"CLP {diff_clp:,.0f}")
    c4.metric("Diferencia rentab.", f"{diff_pct:.1f} pp")

    # ── Gráfico barras ───────────────────────────────────────────────────────
    colors = [
        "#2ecc71" if d == dia_optimo else
        "#e74c3c" if d == dia_peor else
        "#4C9BE8"
        for d in df_res.index
    ]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_res.index,
        y=df_res["capital_final"],
        marker_color=colors,
        text=[f"CLP {v:,.0f}" for v in df_res["capital_final"]],
        textposition="outside",
        hovertemplate="Día %{x}<br>Capital: CLP %{y:,.0f}<extra></extra>",
    ))
    fig.add_hline(
        y=total_invertido,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Invertido: CLP {total_invertido:,.0f}",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title=f"Capital final por día de aporte — {nombres_diarios.get(fondo_sel, fondo_sel)}",
        xaxis_title="Día del mes",
        yaxis_title="Capital final (CLP)",
        xaxis=dict(tickmode="linear", dtick=1),
        height=500,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Tabla detalle ────────────────────────────────────────────────────────
    with st.expander("Ver tabla completa"):
        st.dataframe(
            df_res.rename(columns={
                "capital_final": "Capital final (CLP)",
                "rentabilidad_pct": "Rentabilidad (%)",
            }).style.format({
                "Capital final (CLP)": "{:,.0f}",
                "Rentabilidad (%)": "{:.2f}%",
            }),
            use_container_width=True,
        )

    st.info(
        f"💡 Invertir el **día {dia_optimo}** habría generado "
        f"**CLP {diff_clp:,.0f} más** que el peor día (día {dia_peor}) "
        f"en los últimos {anos} años."
    )


if __name__ == "__main__":
    main()
