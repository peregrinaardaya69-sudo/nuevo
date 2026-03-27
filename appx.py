import math
from dataclasses import dataclass
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="Simulación de Sistemas - Problema 1",
    page_icon="🚑",
    layout="wide",
)

st.title("🚑 Centro de Atención de Emergencias Médicas - Modelo M/M/1")
st.caption(
    "Aplicación en Streamlit para resolver el Problema 1 del práctico integrador. "
    "Incluye análisis teórico M/M/1, simulación discreta y análisis de escenarios."
)


# =========================================================
# ESTRUCTURAS DE DATOS
# =========================================================
@dataclass
class QueueMetrics:
    lambd: float
    mu: float
    rho: float
    p0: float
    lq: float
    l: float
    wq_hours: float
    w_hours: float
    p_n_or_more: float
    p_wait_more_than_t: float
    stable: bool


# =========================================================
# FUNCIONES TEÓRICAS
# =========================================================
def validate_rates(lambd: float, mu: float) -> Tuple[bool, str]:
    """Valida tasas positivas y estabilidad."""
    if lambd <= 0:
        return False, "La tasa de llegada (λ) debe ser mayor que 0."
    if mu <= 0:
        return False, "La tasa de servicio (μ) debe ser mayor que 0."
    if lambd >= mu:
        return False, (
            "El sistema no es estable porque λ ≥ μ. "
            "En una cola M/M/1 estable debe cumplirse λ < μ."
        )
    return True, "Parámetros válidos."


def mm1_metrics(lambd: float, mu: float, threshold_minutes: float, min_calls: int) -> QueueMetrics:
    """Calcula métricas teóricas del modelo M/M/1."""
    rho = lambd / mu
    p0 = 1 - rho
    lq = rho**2 / (1 - rho)
    l = rho / (1 - rho)
    wq_hours = lq / lambd
    w_hours = l / lambd

    # P(N >= k) = rho^k para M/M/1 estable, con k >= 0
    min_calls = max(0, int(min_calls))
    p_n_or_more = rho**min_calls

    # Probabilidad de esperar más de t en cola:
    # P(Wq > t) = rho * exp(-(mu-lambda)*t), para t >= 0
    t_hours = max(0.0, threshold_minutes) / 60.0
    p_wait_more_than_t = rho * math.exp(-(mu - lambd) * t_hours)

    return QueueMetrics(
        lambd=lambd,
        mu=mu,
        rho=rho,
        p0=p0,
        lq=lq,
        l=l,
        wq_hours=wq_hours,
        w_hours=w_hours,
        p_n_or_more=p_n_or_more,
        p_wait_more_than_t=p_wait_more_than_t,
        stable=True,
    )


def automatic_interpretation(metrics: QueueMetrics, threshold_minutes: float) -> str:
    """Genera una interpretación técnica automática."""
    wq_min = metrics.wq_hours * 60
    w_min = metrics.w_hours * 60

    if metrics.rho < 0.70:
        load_msg = "El sistema trabaja con holgura razonable."
    elif metrics.rho < 0.85:
        load_msg = "El sistema opera con carga media-alta y debe monitorearse."
    elif metrics.rho < 0.95:
        load_msg = "El sistema está en zona de riesgo operativo."
    else:
        load_msg = "El sistema está extremadamente tensionado y cualquier variación puede deteriorar el servicio."

    if wq_min <= threshold_minutes:
        threshold_msg = (
            f"El tiempo promedio de espera en cola ({wq_min:.2f} min) se mantiene por debajo "
            f"del umbral de referencia de {threshold_minutes:.2f} min."
        )
    else:
        threshold_msg = (
            f"El tiempo promedio de espera en cola ({wq_min:.2f} min) supera el umbral de referencia "
            f"de {threshold_minutes:.2f} min."
        )

    if metrics.rho >= 0.85:
        recommendation = (
            "Recomendación: evaluar aumento de capacidad (mejorar μ o agregar un segundo operador), "
            "segmentar llamadas por prioridad o rediseñar el flujo de despacho."
        )
    else:
        recommendation = (
            "Recomendación: el sistema es funcional, pero conviene seguir monitoreando la relación λ/μ "
            "y preparar planes de contingencia para horas pico."
        )

    return (
        f"**Interpretación técnica:** La utilización es {metrics.rho:.2%}. {load_msg} "
        f"En promedio, el sistema mantiene {metrics.l:.2f} llamadas y la espera media en cola es "
        f"{wq_min:.2f} minutos; el tiempo total en el sistema es {w_min:.2f} minutos. "
        f"{threshold_msg} {recommendation}"
    )


# =========================================================
# SIMULACIÓN DISCRETA
# =========================================================
def simulate_mm1(
    lambd: float,
    mu: float,
    n_calls: int = 5000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    """
    Simula una cola M/M/1 mediante eventos de llegada y servicio.
    Devuelve:
    - Tabla por llamada.
    - Métricas simuladas.
    - Distribución temporal estimada de estados.
    """
    rng = np.random.default_rng(seed)

    interarrivals = rng.exponential(scale=1 / lambd, size=n_calls)
    arrivals = np.cumsum(interarrivals)
    service_times = rng.exponential(scale=1 / mu, size=n_calls)

    service_start = np.zeros(n_calls)
    departures = np.zeros(n_calls)
    wait_queue = np.zeros(n_calls)
    time_in_system = np.zeros(n_calls)
    idle_before_service = np.zeros(n_calls)

    for i in range(n_calls):
        if i == 0:
            service_start[i] = arrivals[i]
        else:
            service_start[i] = max(arrivals[i], departures[i - 1])

        wait_queue[i] = service_start[i] - arrivals[i]
        departures[i] = service_start[i] + service_times[i]
        time_in_system[i] = departures[i] - arrivals[i]
        idle_before_service[i] = max(0.0, service_start[i] - (departures[i - 1] if i > 0 else 0.0))

    df = pd.DataFrame(
        {
            "llamada": np.arange(1, n_calls + 1),
            "tiempo_entre_llegadas_h": interarrivals,
            "llegada_h": arrivals,
            "servicio_h": service_times,
            "inicio_servicio_h": service_start,
            "salida_h": departures,
            "espera_cola_h": wait_queue,
            "tiempo_sistema_h": time_in_system,
        }
    )

    total_time = departures[-1]
    busy_time = service_times.sum()
    utilization = busy_time / total_time if total_time > 0 else np.nan

    # Little aplicado a datos simulados
    avg_wq = wait_queue.mean()
    avg_w = time_in_system.mean()
    avg_lq = lambd * avg_wq
    avg_l = lambd * avg_w

    # Distribución temporal exacta aproximada por barrido de eventos
    events: List[Tuple[float, int]] = []
    for a in arrivals:
        events.append((float(a), +1))
    for d in departures:
        events.append((float(d), -1))
    events.sort(key=lambda x: (x[0], x[1]))  # salida antes de llegada si hubiera empate

    durations_by_state: Dict[int, float] = {}
    state = 0
    prev_time = 0.0

    for current_time, delta in events:
        duration = current_time - prev_time
        durations_by_state[state] = durations_by_state.get(state, 0.0) + duration
        state += delta
        prev_time = current_time

    state_probabilities = {
        n: dur / total_time for n, dur in durations_by_state.items() if total_time > 0
    }

    state_df = pd.DataFrame(
        {
            "n": list(state_probabilities.keys()),
            "P_n_simulado": list(state_probabilities.values()),
        }
    ).sort_values("n").reset_index(drop=True)

    sim_metrics = {
        "rho_sim": utilization,
        "p0_sim": state_probabilities.get(0, 0.0),
        "lq_sim": avg_lq,
        "l_sim": avg_l,
        "wq_h_sim": avg_wq,
        "w_h_sim": avg_w,
        "max_espera_h": float(wait_queue.max()),
        "total_time_h": float(total_time),
        "busy_time_h": float(busy_time),
    }

    return df, sim_metrics, state_df


def compare_theory_vs_simulation(
    theory: QueueMetrics,
    sim_metrics: Dict[str, float],
) -> pd.DataFrame:
    """Tabla comparativa teoría vs simulación."""
    rows = [
        ("Utilización ρ", theory.rho, sim_metrics["rho_sim"]),
        ("P0", theory.p0, sim_metrics["p0_sim"]),
        ("Lq", theory.lq, sim_metrics["lq_sim"]),
        ("L", theory.l, sim_metrics["l_sim"]),
        ("Wq (horas)", theory.wq_hours, sim_metrics["wq_h_sim"]),
        ("W (horas)", theory.w_hours, sim_metrics["w_h_sim"]),
        ("Wq (minutos)", theory.wq_hours * 60, sim_metrics["wq_h_sim"] * 60),
        ("W (minutos)", theory.w_hours * 60, sim_metrics["w_h_sim"] * 60),
    ]
    out = pd.DataFrame(rows, columns=["Métrica", "Teórico", "Simulado"])
    out["Error absoluto"] = (out["Teórico"] - out["Simulado"]).abs()
    out["Error %"] = np.where(
        out["Teórico"] != 0,
        out["Error absoluto"] / out["Teórico"] * 100,
        np.nan,
    )
    return out


# =========================================================
# VISUALIZACIONES
# =========================================================
def plot_main_metrics(theory: QueueMetrics):
    fig, ax = plt.subplots(figsize=(8, 4))
    metric_names = ["L", "Lq", "W (min)", "Wq (min)"]
    metric_values = [
        theory.l,
        theory.lq,
        theory.w_hours * 60,
        theory.wq_hours * 60,
    ]
    ax.bar(metric_names, metric_values)
    ax.set_title("Métricas principales del sistema")
    ax.set_ylabel("Valor")
    ax.grid(axis="y", alpha=0.3)
    return fig


def plot_state_probabilities(theory: QueueMetrics, state_df: pd.DataFrame, max_n: int = 12):
    n_values = np.arange(0, max_n + 1)
    p_theory = [(1 - theory.rho) * (theory.rho**n) for n in n_values]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(n_values, p_theory, marker="o", label="Teórico M/M/1")

    if not state_df.empty:
        sim_subset = state_df[state_df["n"] <= max_n]
        ax.scatter(sim_subset["n"], sim_subset["P_n_simulado"], label="Simulado")

    ax.set_title("Curva de probabilidad de estados Pn")
    ax.set_xlabel("Número de llamadas en el sistema (n)")
    ax.set_ylabel("Probabilidad")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


def plot_wait_histogram(df_sim: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df_sim["espera_cola_h"] * 60, bins=30)
    ax.set_title("Distribución simulada de espera en cola")
    ax.set_xlabel("Espera en cola (minutos)")
    ax.set_ylabel("Frecuencia")
    ax.grid(alpha=0.3)
    return fig


def scenario_analysis(base_lambda: float, base_mu: float, threshold_minutes: float, min_calls: int):
    scenarios = pd.DataFrame(
        [
            ("Base", base_lambda, base_mu),
            ("Optimista (+20% μ)", base_lambda, base_mu * 1.20),
            ("Pesimista (+20% λ)", base_lambda * 1.20, base_mu),
        ],
        columns=["Escenario", "λ", "μ"],
    )

    records = []
    for _, row in scenarios.iterrows():
        if row["λ"] < row["μ"]:
            m = mm1_metrics(row["λ"], row["μ"], threshold_minutes, min_calls)
            records.append(
                {
                    "Escenario": row["Escenario"],
                    "λ (llamadas/hora)": row["λ"],
                    "μ (llamadas/hora)": row["μ"],
                    "ρ": m.rho,
                    "Lq": m.lq,
                    "L": m.l,
                    "Wq (min)": m.wq_hours * 60,
                    "W (min)": m.w_hours * 60,
                    f"P(Wq > {threshold_minutes:.0f} min)": m.p_wait_more_than_t,
                }
            )
        else:
            records.append(
                {
                    "Escenario": row["Escenario"],
                    "λ (llamadas/hora)": row["λ"],
                    "μ (llamadas/hora)": row["μ"],
                    "ρ": np.nan,
                    "Lq": np.nan,
                    "L": np.nan,
                    "Wq (min)": np.nan,
                    "W (min)": np.nan,
                    f"P(Wq > {threshold_minutes:.0f} min)": np.nan,
                }
            )
    return pd.DataFrame(records)


def plot_scenario_waits(df_scenarios: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df_scenarios["Escenario"], df_scenarios["Wq (min)"])
    ax.set_title("Comparación de espera promedio en cola por escenario")
    ax.set_ylabel("Wq (min)")
    ax.grid(axis="y", alpha=0.3)
    return fig


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙️ Parámetros del modelo")

default_lambda = 18.0
default_mu = 24.0

lambd = st.sidebar.number_input(
    "Tasa de llegada λ (llamadas por hora)",
    min_value=0.1,
    value=default_lambda,
    step=0.1,
)

mu = st.sidebar.number_input(
    "Tasa de servicio μ (llamadas por hora)",
    min_value=0.1,
    value=default_mu,
    step=0.1,
)

threshold_minutes = st.sidebar.number_input(
    "Tiempo umbral de espera (minutos)",
    min_value=0.0,
    value=8.0,
    step=0.5,
)

min_calls = st.sidebar.number_input(
    "Número mínimo de llamadas para P(N ≥ k)",
    min_value=0,
    value=4,
    step=1,
)

st.sidebar.subheader("Parámetros de simulación")
n_calls = st.sidebar.slider("Número de llamadas a simular", 500, 20000, 5000, 500)
seed = st.sidebar.number_input("Semilla aleatoria", min_value=0, value=42, step=1)

# Controles de escenario interactivo
st.sidebar.subheader("Escenario interactivo")
lambda_slider = st.sidebar.slider("Multiplicador de λ", 0.50, 1.80, 1.00, 0.05)
mu_slider = st.sidebar.slider("Multiplicador de μ", 0.50, 1.80, 1.00, 0.05)

lambda_scenario = lambd * lambda_slider
mu_scenario = mu * mu_slider


# =========================================================
# BLOQUE PRINCIPAL
# =========================================================
is_valid, validation_msg = validate_rates(lambd, mu)
if not is_valid:
    st.error(validation_msg)
    st.stop()

theory = mm1_metrics(lambd, mu, threshold_minutes, int(min_calls))
df_sim, sim_metrics, state_df = simulate_mm1(lambd, mu, n_calls=n_calls, seed=int(seed))
comparison_df = compare_theory_vs_simulation(theory, sim_metrics)

st.subheader("1) Resultados teóricos del modelo M/M/1")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Utilización ρ", f"{theory.rho:.2%}")
col2.metric("Probabilidad P0", f"{theory.p0:.4f}")
col3.metric("Lq", f"{theory.lq:.4f}")
col4.metric("L", f"{theory.l:.4f}")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Wq (min)", f"{theory.wq_hours * 60:.4f}")
col6.metric("W (min)", f"{theory.w_hours * 60:.4f}")
col7.metric(f"P(N ≥ {int(min_calls)})", f"{theory.p_n_or_more:.4f}")
col8.metric(f"P(Wq > {threshold_minutes:.1f} min)", f"{theory.p_wait_more_than_t:.4f}")

if theory.rho >= 0.85:
    st.warning(
        "Advertencia: la utilización es alta. En un sistema de emergencias esto implica riesgo "
        "de congestión y deterioro del nivel de servicio."
    )
else:
    st.success("El sistema es estable bajo los parámetros actuales.")

st.markdown(automatic_interpretation(theory, threshold_minutes))

st.subheader("2) Comparación teórica vs simulada")
st.dataframe(comparison_df, use_container_width=True)

st.subheader("3) Visualizaciones")
viz_col1, viz_col2 = st.columns(2)
with viz_col1:
    st.pyplot(plot_main_metrics(theory), clear_figure=True)
with viz_col2:
    st.pyplot(plot_state_probabilities(theory, state_df), clear_figure=True)

st.pyplot(plot_wait_histogram(df_sim), clear_figure=True)

st.subheader("4) Tabla de simulación (primeras llamadas)")
display_df = df_sim.copy()
for col in display_df.columns[1:]:
    display_df[col] = display_df[col].round(5)
st.dataframe(display_df.head(25), use_container_width=True)

st.subheader("5) Análisis de escenarios")
scenario_df = scenario_analysis(lambd, mu, threshold_minutes, int(min_calls))
st.dataframe(scenario_df, use_container_width=True)
st.pyplot(plot_scenario_waits(scenario_df), clear_figure=True)

st.subheader("6) Escenario interactivo con sliders")
valid_scenario, msg_scenario = validate_rates(lambda_scenario, mu_scenario)
if valid_scenario:
    m_scenario = mm1_metrics(lambda_scenario, mu_scenario, threshold_minutes, int(min_calls))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("λ ajustado", f"{lambda_scenario:.2f}")
    c2.metric("μ ajustado", f"{mu_scenario:.2f}")
    c3.metric("ρ ajustado", f"{m_scenario.rho:.2%}")
    c4.metric("Wq ajustado (min)", f"{m_scenario.wq_hours * 60:.2f}")
    st.markdown(automatic_interpretation(m_scenario, threshold_minutes))
else:
    st.error(f"Escenario interactivo inválido: {msg_scenario}")

st.subheader("7) Conclusión ejecutiva automática")
base_wq = theory.wq_hours * 60
base_w = theory.w_hours * 60

if theory.rho < 0.80:
    conclusion = (
        "El sistema base es operativamente aceptable para una primera evaluación, "
        "aunque debe monitorearse en franjas pico por tratarse de emergencias médicas."
    )
elif theory.rho < 0.90:
    conclusion = (
        "El sistema base presenta presión considerable. Aunque sigue siendo estable, "
        "la espera puede volverse crítica ante variaciones moderadas de la demanda."
    )
else:
    conclusion = (
        "El sistema base es riesgoso para una operación de emergencias. "
        "Se recomienda incrementar capacidad o rediseñar la atención de inmediato."
    )

st.markdown(
    f"""
**Resumen técnico final**

- λ = **{lambd:.2f}** llamadas/hora  
- μ = **{mu:.2f}** llamadas/hora  
- Utilización = **{theory.rho:.2%}**  
- Espera promedio en cola = **{base_wq:.2f} min**  
- Tiempo promedio total en el sistema = **{base_w:.2f} min**  
- Probabilidad de esperar más de **{threshold_minutes:.1f} min** = **{theory.p_wait_more_than_t:.2%}**

**Conclusión:** {conclusion}
"""
)

with st.expander("Ver fórmulas utilizadas"):
    st.markdown(
        r"""
Para una cola **M/M/1** estable, con \( \lambda < \mu \):

- \( \rho = \frac{\lambda}{\mu} \)
- \( P_0 = 1 - \rho \)
- \( L_q = \frac{\rho^2}{1-\rho} \)
- \( L = \frac{\rho}{1-\rho} \)
- \( W_q = \frac{L_q}{\lambda} \)
- \( W = \frac{L}{\lambda} \)
- \( P(N \geq k) = \rho^k \)
- \( P(W_q > t) = \rho e^{-(\mu-\lambda)t} \)
"""
    )

st.caption(
    "Desarrollado para el Problema 1 del práctico integrador de Simulación de Sistemas."
)
