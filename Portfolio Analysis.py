import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# Configiracion de la pagina
st.set_page_config(page_title="Portfolio Analysis", page_icon="📈", layout="wide")
st.sidebar.title("Analizador de Portafolios de Inversion")

# Creamos pestañas para la aplicacion
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Analisis individual del Activo", "Analisis de Portafolio", "Optimizacion Markowitz","Optimización Black-Litterman","Simulación Monte Carlo"])

# Entrada de simbolos y pesos 
simbolos = st.sidebar.text_input("Ingrese los simbolos de las acciones (separados por comas)", "AAPL, MSFT, GOOG, AMZN, NVDA")
pesos = st.sidebar.text_input("Ingrese los pesos de las acciones (separados por comas)", "0.2,0.2,0.2,0.2,0.2")

simbolos = [s.strip().upper() for s in simbolos.split(",")]
pesos = [float(p) for p in pesos.split(",")]    

# Seleccion de benchmark
benchmark_options = {
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Dow Jones": "^DJI",
    "Russell 2000": "^RUT",
    "ACWI": "ACWI"
}

selected_benchmark = st.sidebar.selectbox("Seleccione un benchmark", list(benchmark_options.keys()))

#Periodo de tiempo
end_date = datetime.today().date()
start_date_options = { 

    "1 mes": end_date - timedelta(days=30),
    "3 meses": end_date - timedelta(days=90),
    "6 meses": end_date - timedelta(days=180),
    "1 año": end_date - timedelta(days=365),
    "2 años": end_date - timedelta(days=365*2),
    "5 años": end_date - timedelta(days=365*5),
    "10 años": end_date - timedelta(days=365*10) }

selected_timeframe = st.sidebar.selectbox("Seleccione el periodo de tiempo", list(start_date_options.keys()))
start_date = start_date_options[selected_timeframe]

# FUNCIONES AUXILIARES

def obtener_datos(simbolos, start_date, end_date):
    """Obtiene los datos de precios ajustados de los simbolos especificados entre las fechas dadas."""
    data = yf.download(simbolos, start=start_date, end=end_date)["Close"]
    return data.ffill().dropna()

def calcular_metricas(data):
    """Calcula los rendimientos diarios y acumulados de los precios ajustados."""
    returns = data.pct_change().dropna()
    returns_acumulados = (1 + returns).cumprod() - 1
    normalized_prices = data / data.iloc[0] * 100
    return returns, returns_acumulados, normalized_prices

def calcular_rendimiento_portafolio(returns, pesos):
    
    portafolio_returns = (returns * pesos).sum(axis=1)
    return portafolio_returns

def Calcular_Var(returns, confidence_level=0.95):
    """Calcula el VaR del portafolio."""
    var = np.percentile(returns, (1 - confidence_level) * 100)
    return var
def Calcular_CVaR(returns, var):
    """Calcula el CVaR del portafolio."""
    cvar = returns[returns <= var].mean()
    return cvar

def calcular_sharpe_dinamico(rendimientos, selected_timeframe, rf_anual=0.0449):
    """
    Calcula el Sharpe Ratio ajustado al horizonte temporal seleccionado.
    
    Parámetros:
    - rendimientos: pd.Series con retornos diarios del activo.
    - selected_timeframe: str, clave del periodo seleccionado por el usuario (ej. '3 meses').
    - rf_anual: float, tasa libre de riesgo anualizada (por defecto 4.49%).

    Retorna:
    - sharpe_ratio ajustado al periodo seleccionado.
    """

    # Diccionario de días hábiles estimados por periodo
    period_days = {
        "1 mes": 21,
        "3 meses": 63,
        "6 meses": 126,
        "1 año": 252,
        "2 años": 504,
        "5 años": 1260,
        "10 años": 2520
    }

    dias_periodo = period_days.get(selected_timeframe, 252)  # por defecto 1 año

    # Tasa libre de riesgo ajustada al periodo (compuesta)
    rf_periodo = (1 + rf_anual) ** (dias_periodo / 252) - 1

    # Retorno esperado y volatilidad ajustados al periodo
    retorno_esperado = rendimientos.mean() * dias_periodo
    volatilidad_ajustada = rendimientos.std() * np.sqrt(dias_periodo)

    sharpe_ratio = (retorno_esperado - rf_periodo) / volatilidad_ajustada
    return sharpe_ratio



if len(simbolos) != len(pesos) or abs(sum(pesos) - 1) > 1e-6:
    # Mensaje de error si los simbolos y pesos no coinciden
    st.sidebar.error("El número de símbolos y pesos no coincide. Por favor, verifique los datos ingresados.")
else:

    # Descarga de datos

    all_symbols = simbolos + [benchmark_options[selected_benchmark]]
    data_stocks = obtener_datos(all_symbols, start_date, end_date)
    returns, returns_acumulados, precios_norm = calcular_metricas(data_stocks)


    # TAB 1: ANALISIS INDIVIDUAL DEL ACTIVO 
    with tab1:
        st.header("📊 Análisis Individual del Activo")

        selected_asset = st.selectbox("Seleccione un activo para analizar", simbolos)

        # Extraemos series de tiempo específicas del activo
        precios = data_stocks[selected_asset]
        rendimientos = returns[selected_asset]
        rend_acumulado = returns_acumulados[selected_asset]

        # ================================
        # 1️⃣ RESUMEN GENERAL DE RENDIMIENTO
        # ================================
        st.subheader("🔹 Resumen de Rendimiento")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rendimiento Acumulado (%)", f"{rend_acumulado.iloc[-1] * 100:.2f}%")
        col2.metric("Media de Retornos Diarios (%)", f"{rendimientos.mean() * 100:.4f}%")
        col3.metric("Volatilidad Anualizada (%)", f"{rendimientos.std() * np.sqrt(252) * 100:.2f}%")

        # ================================
        # 2️⃣ INDICADORES DE RIESGO
        # ================================
        st.subheader("🔸 Indicadores de Riesgo")
        sharpe = calcular_sharpe_dinamico(rendimientos, selected_timeframe)
        sortino = rendimientos.mean() / rendimientos[rendimientos < 0].std()
        var_95 = Calcular_Var(rendimientos)
        cvar_95 = Calcular_CVaR(rendimientos, var_95)
        beta = np.cov(rendimientos, returns[benchmark_options[selected_benchmark]])[0][1] / np.var(returns[benchmark_options[selected_benchmark]])

        max_drawdown = (rend_acumulado.cummax() - rend_acumulado).max()

        col4, col5, col6 = st.columns(3)
        col4.metric("Sharpe Ratio", f"{sharpe:.2f}")
        col5.metric("Sortino Ratio", f"{sortino:.2f}")
        col6.metric("Max Drawdown (%)", f"{max_drawdown * 100:.2f}%")
        



        col7, col8, colbeta = st.columns(3)
        colbeta.metric("Beta", f"{beta:.2f}")
        col7.metric("VaR 95% (%)", f"{var_95 * 100:.2f}%")
        col8.metric("CVaR 95% (%)", f"{cvar_95 * 100:.2f}%")

        # ================================
        # 3️⃣ ESTADÍSTICAS AVANZADAS
        # ================================
        st.subheader("📐 Estadísticas de Retornos")
        skewness = rendimientos.skew()
        kurtosis = rendimientos.kurtosis()

        col9, col10 = st.columns(2)
        col9.metric("Skewness", f"{skewness:.3f}")
        col10.metric("Curtosis", f"{kurtosis:.3f}")

        # ================================
        # 4️⃣ GRÁFICOS INTERACTIVOS
        # ================================

        st.subheader("📊 Comparaciones Visuales: Activo vs Benchmark")
        benchmark_symbol = benchmark_options[selected_benchmark]
        benchmark_norm = precios_norm[benchmark_symbol]
        benchmark_returns = returns[benchmark_symbol]

        # === Gráfico principal de precios normalizados (100 base)
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=precios_norm.index, y=precios_norm[selected_asset], 
                                    name=selected_asset, line=dict(color='royalblue')))
        fig_price.add_trace(go.Scatter(x=benchmark_norm.index, y=benchmark_norm, 
                                    name=selected_benchmark, line=dict(color='firebrick')))
        fig_price.update_layout(title=f"Precio Normalizado: {selected_asset} vs {selected_benchmark} (Base 100)",
                                xaxis_title="Fecha", yaxis_title="Precio Normalizado")
        st.plotly_chart(fig_price, use_container_width=True)



        # === Histogramas por separado
        st.subheader(f"Distribución de Retornos: {selected_asset} vs {selected_benchmark}")
        col_hist1, col_hist2 = st.columns(2)

        # Histograma del activo seleccionado
        with col_hist1:
            fig_hist_asset = px.histogram(rendimientos, nbins=50, title=f"Distribución de Retornos - {selected_asset}",
                                        labels={"value": "Retornos"}, color_discrete_sequence=["#1f77b4"])
            st.plotly_chart(fig_hist_asset, use_container_width=True)

        # Histograma del benchmark
        with col_hist2:
            fig_hist_benchmark = px.histogram(benchmark_returns, nbins=50, title=f"Distribución de Retornos - {selected_benchmark}",
                                            labels={"value": "Retornos"}, color_discrete_sequence=["#ff7f0e"])
            st.plotly_chart(fig_hist_benchmark, use_container_width=True)

    # ---------------------------------------------------------
    # TAB 2: ANALISIS DEL PORTAFOLIO
    # ---------------------------------------------------------
    with tab2:
        st.header("📈 Análisis del Portafolio")

        # Calculamos los retornos del benchmark y del portafolio
        benchmark_symbol = benchmark_options[selected_benchmark]
        benchmark_returns = returns[benchmark_symbol]
        portfolio_returns = calcular_rendimiento_portafolio(returns[simbolos], pesos)

        # Rendimientos acumulados para portafolio y benchmark
        portfolio_cumreturns = (1 + portfolio_returns).cumprod() - 1
        benchmark_cumreturns = (1 + benchmark_returns).cumprod() - 1

        # Calculamos las principales métricas del portafolio
        total_return_portfolio = portfolio_cumreturns.iloc[-1] * 100  # en porcentaje
        sharpe_portfolio = calcular_sharpe_dinamico(portfolio_returns, selected_timeframe)
        sortino_portfolio = portfolio_returns.mean() / portfolio_returns[portfolio_returns < 0].std()
        var_95_portfolio = Calcular_Var(portfolio_returns)
        cvar_95_portfolio = Calcular_CVaR(portfolio_returns, var_95_portfolio)
        max_dd_portfolio = (portfolio_cumreturns.cummax() - portfolio_cumreturns).max() * 100

        # NUEVO: Cálculo del beta del portafolio
        beta_portfolio = np.cov(portfolio_returns, benchmark_returns)[0][1] / np.var(benchmark_returns)

        # Mostramos las métricas utilizando columnas (se agregan 6 columnas para incluir el beta)
        colp1, colp2, colp3 = st.columns(3)
        colp1.metric("Rendimiento Total", f"{total_return_portfolio:.2f}%")
        colp2.metric("Sharpe Ratio", f"{sharpe_portfolio:.2f}")
        colp3.metric("Sortino Ratio", f"{sortino_portfolio:.2f}")

        colp4, colp5, colp6 = st.columns(3)
        colp4.metric("VaR 95%", f"{var_95_portfolio * 100:.2f}%")
        colp5.metric("CVaR 95%", f"{cvar_95_portfolio * 100:.2f}%")
        colp6.metric("Beta del Portafolio", f"{beta_portfolio:.2f}")

        # Gráfico comparativo: Rendimientos Acumulados del Portafolio vs Benchmark

        st.subheader(f"Rendimientos Acumulados: Portafolio vs {selected_benchmark}")
        fig_port = go.Figure()
        fig_port.add_trace(go.Scatter(
            x=portfolio_cumreturns.index,
            y=portfolio_cumreturns,
            name='Portafolio',
            line=dict(color='blue')
        ))
        fig_port.add_trace(go.Scatter(
            x=benchmark_cumreturns.index,
            y=benchmark_cumreturns,
            name=selected_benchmark,
            line=dict(color='orange')
        ))
        fig_port.update_layout(
            title=f"Rendimientos Acumulados: Portafolio vs {selected_benchmark}",
            xaxis_title="Fecha",
            yaxis_title="Rendimiento Acumulado"
        )
        st.plotly_chart(fig_port, use_container_width=True)

        # ================================
        # DISTRIBUCIÓN DE RETORNOS - PORTAFOLIO vs BENCHMARK
        # ================================
        st.subheader("Distribución de Retornos del Portafolio vs Benchmark")

        # 1. Cálculo de VaR y CVaR del Portafolio y del Benchmark
        var_95_portfolio = Calcular_Var(portfolio_returns, confidence_level=0.95)
        cvar_95_portfolio = Calcular_CVaR(portfolio_returns, var_95_portfolio)

        var_95_benchmark = Calcular_Var(benchmark_returns, confidence_level=0.95)
        cvar_95_benchmark = Calcular_CVaR(benchmark_returns, var_95_benchmark)

        # 2. Histograma de Retornos del Portafolio con líneas VaR y CVaR
        fig_dist_port = go.Figure()

        fig_dist_port.add_trace(
            go.Histogram(
                x=portfolio_returns,
                nbinsx=50,
                name='Retornos',
                marker_color='green',
                opacity=0.75
            )
        )

        # Línea de VaR 95%
        fig_dist_port.add_vline(
            x=var_95_portfolio,
            line_width=2,
            line_dash="dash",
            line_color="blue",
            annotation_text="VaR 95%",
            annotation_position="top left"
        )

        # Línea de CVaR 95%
        fig_dist_port.add_vline(
            x=cvar_95_portfolio,
            line_width=2,
            line_dash="dot",
            line_color="red",
            annotation_text="CVaR 95%",
            annotation_position="top left"
        )

        fig_dist_port.update_layout(
            title="Distribución de Retornos - Portafolio",
            xaxis_title="Retornos",
            yaxis_title="Frecuencia",
            bargap=0.2,
            template="plotly_white",   # Puedes probar "ggplot2", "seaborn", etc.
            showlegend=False
        )

        # 3. Histograma de Retornos del Benchmark con líneas VaR y CVaR
        fig_dist_bench = go.Figure()

        fig_dist_bench.add_trace(
            go.Histogram(
                x=benchmark_returns,
                nbinsx=50,
                name='Retornos',
                marker_color='orange',
                opacity=0.75
            )
        )

        # Línea de VaR 95%
        fig_dist_bench.add_vline(
            x=var_95_benchmark,
            line_width=2,
            line_dash="dash",
            line_color="blue",
            annotation_text="VaR 95%",
            annotation_position="top left"
        )

        # Línea de CVaR 95%
        fig_dist_bench.add_vline(
            x=cvar_95_benchmark,
            line_width=2,
            line_dash="dot",
            line_color="red",
            annotation_text="CVaR 95%",
            annotation_position="top left"
        )

        fig_dist_bench.update_layout(
            title=f"Distribución de Retornos - {selected_benchmark}",
            xaxis_title="Retornos",
            yaxis_title="Frecuencia",
            bargap=0.2,
            template="plotly_white",
            showlegend=False
        )

        # 4. Mostrar las gráficas lado a lado
        col_left, col_right = st.columns(2)
        with col_left:
            st.plotly_chart(fig_dist_port, use_container_width=True)
        with col_right:
            st.plotly_chart(fig_dist_bench, use_container_width=True)


        # ================================
        # 🔄 MATRIZ DE CORRELACIÓN ENTRE ACTIVOS
        # ================================
        st.subheader("📌 Correlación entre Activos del Portafolio")
        corr_matrix = returns[simbolos].corr()

        fig_corr = px.imshow(corr_matrix, 
                             text_auto=True, 
                             color_continuous_scale='RdBu_r', 
                             zmin=-1, zmax=1,
                             title="Matriz de Correlación entre Activos")
        st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("📊 Contribución Individual al Portafolio")

        # Contribución al rendimiento promedio
        contrib_rend = (returns[simbolos].mean() * pesos)
        contrib_riesgo = (returns[simbolos].std() * pesos)

        contrib_df = pd.DataFrame({
            "Activo": simbolos,
            "Contribución al Retorno (%)": contrib_rend * 100,
            "Contribución al Riesgo (%)": contrib_riesgo * 100
        })

        st.dataframe(contrib_df.set_index("Activo").style
                     .background_gradient(cmap="YlGn", axis=0), use_container_width=True)

        st.subheader("📉 Máximo Drawdown del Portafolio")

        # Cálculo y gráfico del drawdown
        rolling_max = portfolio_cumreturns.cummax()
        drawdown = (portfolio_cumreturns - rolling_max) / rolling_max

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown * 100,
                                    fill='tozeroy',
                                    name='Drawdown (%)',
                                    line=dict(color='crimson')))
        fig_dd.update_layout(title="Drawdown del Portafolio (%)",
                             xaxis_title="Fecha",
                             yaxis_title="Drawdown (%)")
        st.plotly_chart(fig_dd, use_container_width=True)


# ---------------------------------------------------------
# TAB 3: OPTIMIZACIÓN DEL PORTAFOLIO - MARKOWITZ
# ---------------------------------------------------------



    with tab3:
        st.header("Optimización del Portafolio - Markowitz")

        if len(simbolos) < 2:
            st.error("Se requiere al menos dos activos para la optimización.")
        else:
            # --- Selección del método para calcular rendimientos esperados ---
            rend_model = st.radio("Método para calcular rendimientos esperados", ["Histórico", "CAPM"], horizontal=True)
            risk_free_rate = st.number_input("Tasa libre de riesgo (anual)", value=0.0449, step=0.001)
            
            # Cálculo de rendimientos esperados según método seleccionado
            if rend_model == "Histórico":
                exp_returns = returns[simbolos].mean()
            else:
                # Para CAPM se convierte la tasa anual a diaria (aprox. 252 días hábiles)
                rf_daily = (1 + risk_free_rate) ** (1/252) - 1
                benchmark_symbol = benchmark_options[selected_benchmark]
                bench_exp = returns[benchmark_symbol].mean()
                exp_returns = pd.Series(index=simbolos, dtype=float)
                # Se calculan los beta individuales
                beta_values = {}
                for s in simbolos:
                    beta_i = np.cov(returns[s], returns[benchmark_symbol])[0, 1] / np.var(returns[benchmark_symbol])
                    beta_values[s] = beta_i
                    exp_returns[s] = rf_daily + beta_i * (bench_exp - rf_daily)
            
            st.subheader("Rendimientos Esperados por Activo")
            st.dataframe(exp_returns.to_frame(name="Rendimiento Esperado"))

            # --- Cálculo de la matriz de covarianza y configuración de la optimización ---
            cov_matrix = returns[simbolos].cov()
            num_activos = len(simbolos)
            opt_type = st.selectbox("Tipo de optimización", ["Minimizar Varianza", "Maximizar Sharpe Ratio"])
            
            # Restricciones: suma de pesos igual a 1 y cada peso entre 0 y 1
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for _ in range(num_activos))
            initial_weights = np.array(num_activos * [1. / num_activos])
            
            def portfolio_performance(weights, mean_returns, cov_matrix):
                ret = np.dot(weights, mean_returns)
                vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return ret, vol
            
            # --- Optimización ---
            if opt_type == "Minimizar Varianza":
                def portfolio_variance(weights):
                    return np.dot(weights.T, np.dot(cov_matrix, weights))
                optimizacion = minimize(portfolio_variance, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
            else:
                def portfolio_neg_sharpe(weights):
                    ret, vol = portfolio_performance(weights, exp_returns, cov_matrix)
                    return -(ret - risk_free_rate) / vol
                optimizacion = minimize(portfolio_neg_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if optimizacion.success:
                optimal_weights = optimizacion.x
                # --- Cálculo de métricas del portafolio ---
                ret_opt, vol_opt = portfolio_performance(optimal_weights, exp_returns, cov_matrix)
                sharpe_opt = (ret_opt - risk_free_rate) / vol_opt
                
                # Cálculo de la beta del portafolio utilizando retornos históricos
                portfolio_returns_hist = returns[simbolos].dot(optimal_weights)
                benchmark_symbol = benchmark_options[selected_benchmark]
                benchmark_returns = returns[benchmark_symbol]
                portfolio_beta = np.cov(portfolio_returns_hist, benchmark_returns)[0, 1] / np.var(benchmark_returns)
                
                # --- Presentación estética de las métricas en columnas ---
                st.subheader("Resultados de la Optimización")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Retorno Esperado", f"{ret_opt*100:.2f}%")
                col2.metric("Volatilidad", f"{vol_opt*100:.2f}%")
                col3.metric("Ratio Sharpe", f"{sharpe_opt:.2f}")
                col4.metric("Beta del Portafolio", f"{portfolio_beta:.2f}")
                
                # Mostrar los pesos óptimos en una tabla
                st.subheader("Pesos Óptimos")
                weights_df = pd.DataFrame({
                    "Activo": simbolos,
                    "Peso Óptimo": optimal_weights
                })
                st.dataframe(weights_df.set_index("Activo"))
                
                # --- Cálculo y presentación de la Frontera Eficiente ---
                st.subheader("Frontera Eficiente")
                target_returns = np.linspace(exp_returns.min(), exp_returns.max(), 50)
                frontier_vol = []
                for target in target_returns:
                    constraints_target = (
                        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                        {'type': 'eq', 'fun': lambda x, target=target: np.dot(x, exp_returns) - target}
                    )
                    opt_min = minimize(lambda w: np.dot(w.T, np.dot(cov_matrix, w)), initial_weights, 
                                    method='SLSQP', bounds=bounds, constraints=constraints_target)
                    if opt_min.success:
                        frontier_vol.append(np.sqrt(opt_min.fun))
                    else:
                        frontier_vol.append(np.nan)
                
                fig_front = go.Figure()
                fig_front.add_trace(go.Scatter(
                    x=np.array(frontier_vol) * 100,
                    y=target_returns * 100,
                    mode='lines',
                    name="Frontera Eficiente"
                ))
                fig_front.add_trace(go.Scatter(
                    x=[vol_opt * 100],
                    y=[ret_opt * 100],
                    mode='markers',
                    name="Portafolio Óptimo",
                    marker=dict(color='red', size=12)
                ))
                fig_front.update_layout(
                    title="Frontera Eficiente del Portafolio",
                    xaxis_title="Volatilidad (%)",
                    yaxis_title="Retorno Esperado (%)",
                    template="plotly_white"
                )
                st.plotly_chart(fig_front, use_container_width=True)
            else:
                st.error("La optimización no fue exitosa. Intente modificar los parámetros.")



# ---------------------------------------------------------
# TAB 4: OPTIMIZACIÓN BLACK-LITTERMAN
# ---------------------------------------------------------

with tab4:
    st.header("Optimización Black-Litterman")

    if len(simbolos) < 2:
        st.error("Se requiere al menos dos activos para la optimización.")
    else:
        # 1. Cálculo de la matriz de covarianza y rendimientos de equilibrio implícitos
        cov_matrix = returns[simbolos].cov()
        delta = st.number_input("Coeficiente de aversión al riesgo (δ)", value=2.5, step=0.1)
        tau = st.number_input("τ (tau)", value=0.05, step=0.01)
        # Supongamos que los pesos de mercado sean los mismos que los ingresados (estos deben ser representativos)
        w_m = np.array(pesos)
        pi = delta * np.dot(cov_matrix, w_m)
        
        st.subheader("Retornos de Equilibrio Implícitos (π)")
        st.dataframe(pd.DataFrame(pi, index=simbolos, columns=["π (Implícito)"]))
        
        # 2. Definir las opiniones del inversor (Views)
        st.subheader("Opiniones del Inversor")
        view_count = st.number_input("Número de vistas", min_value=1, max_value=len(simbolos), value=1, step=1)
        
        P_list = []
        Q_list = []
        for i in range(int(view_count)):
            st.markdown(f"**Vista {i+1}:**")
            activo_vista = st.selectbox(f"Selecciona el activo para la vista {i+1}", simbolos, key=f"vista_activo_{i}")
            q_val = st.number_input(
                f"Rendimiento esperado para {activo_vista} (en decimal, ej. 0.01 para 1%)",
                value=0.01, step=0.001, key=f"vista_q_{i}"
            )
            # Para este ejemplo se asume una vista sobre un solo activo: P tendrá 1 en la posición del activo y 0 en el resto
            P_row = [0] * len(simbolos)
            P_row[simbolos.index(activo_vista)] = 1
            P_list.append(P_row)
            Q_list.append(q_val)
        
        P = np.array(P_list)
        Q = np.array(Q_list)
        
        # 3. Matriz de incertidumbre Ω para las vistas:
        # Usamos una aproximación simple: Ω = diagonal de (τ * P Σ P^T)
        Omega = np.diag(np.diag(tau * np.dot(np.dot(P, cov_matrix), P.T)))
        st.subheader("Matriz de Incertidumbre (Ω)")
        st.write(Omega)
        
        # 4. Cálculo de los rendimientos ajustados por Black-Litterman:
        # Fórmula: μ = [ (τΣ)^(-1) + P^T Ω^(-1) P ]^(-1) [ (τΣ)^(-1) π + P^T Ω^(-1) Q ]
        inv_tauSigma = np.linalg.inv(tau * cov_matrix)
        inv_Omega = np.linalg.inv(Omega)
        middle_term = np.linalg.inv(inv_tauSigma + np.dot(np.dot(P.T, inv_Omega), P))
        adjusted_returns = np.dot(middle_term, (np.dot(inv_tauSigma, pi) + np.dot(np.dot(P.T, inv_Omega), Q)))
        
        st.subheader("Rendimientos Ajustados (Black-Litterman)")
        st.dataframe(pd.DataFrame(adjusted_returns, index=simbolos, columns=["Rendimiento Ajustado"]))
        
        # 5. Optimización del portafolio usando los rendimientos ajustados
        opt_type_bl = st.selectbox("Tipo de optimización", ["Minimizar Varianza", "Maximizar Sharpe Ratio"], key="opt_bl")
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(len(simbolos)))
        initial_weights = np.array(len(simbolos) * [1. / len(simbolos)])
        
        def portfolio_performance(weights, mean_returns, cov_matrix):
            ret = np.dot(weights, mean_returns)
            vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return ret, vol
        
        if opt_type_bl == "Minimizar Varianza":
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            opt_bl = minimize(portfolio_variance, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        else:
            def portfolio_neg_sharpe(weights):
                ret, vol = portfolio_performance(weights, adjusted_returns, cov_matrix)
                return -(ret - risk_free_rate) / vol
            opt_bl = minimize(portfolio_neg_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if opt_bl.success:
            opt_weights = opt_bl.x
            ret_bl, vol_bl = portfolio_performance(opt_weights, adjusted_returns, cov_matrix)
            sharpe_bl = (ret_bl - risk_free_rate) / vol_bl
            # Cálculo de la beta del portafolio con los pesos optimizados
            portfolio_returns_bl = returns[simbolos].dot(opt_weights)
            benchmark_returns = returns[benchmark_options[selected_benchmark]]
            beta_bl = np.cov(portfolio_returns_bl, benchmark_returns)[0, 1] / np.var(benchmark_returns)
            
            st.subheader("Resultados de la Optimización Black-Litterman")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Retorno Esperado", f"{ret_bl*100:.2f}%")
            col2.metric("Volatilidad", f"{vol_bl*100:.2f}%")
            col3.metric("Ratio Sharpe", f"{sharpe_bl:.2f}")
            col4.metric("Beta Portafolio", f"{beta_bl:.2f}")
            
            st.subheader("Pesos Óptimos")
            weights_bl_df = pd.DataFrame({"Activo": simbolos, "Peso Óptimo": opt_weights})
            st.dataframe(weights_bl_df.set_index("Activo"))
            
            # 6. Frontera Eficiente (Black-Litterman)
            st.subheader("Frontera Eficiente - Black-Litterman")
            target_returns = np.linspace(adjusted_returns.min(), adjusted_returns.max(), 50)
            frontier_vol = []
            for target in target_returns:
                constraints_target = (
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                    {'type': 'eq', 'fun': lambda x, target=target: np.dot(x, adjusted_returns) - target}
                )
                opt_min = minimize(lambda w: np.dot(w.T, np.dot(cov_matrix, w)), initial_weights,
                                   method='SLSQP', bounds=bounds, constraints=constraints_target)
                if opt_min.success:
                    frontier_vol.append(np.sqrt(opt_min.fun))
                else:
                    frontier_vol.append(np.nan)
            
            fig_bl = go.Figure()
            fig_bl.add_trace(go.Scatter(
                x=np.array(frontier_vol) * 100,
                y=target_returns * 100,
                mode='lines',
                name="Frontera Eficiente"
            ))
            fig_bl.add_trace(go.Scatter(
                x=[vol_bl * 100],
                y=[ret_bl * 100],
                mode='markers',
                name="Portafolio Óptimo",
                marker=dict(color='red', size=12)
            ))
            fig_bl.update_layout(
                title="Frontera Eficiente - Black-Litterman",
                xaxis_title="Volatilidad (%)",
                yaxis_title="Retorno Esperado (%)",
                template="plotly_white"
            )
            st.plotly_chart(fig_bl, use_container_width=True)
        else:
            st.error("La optimización Black-Litterman no fue exitosa. Intente modificar los parámetros.")







# ---------------------------------------------------------
# TAB 5: SIMULACIÓN MONTE CARLO
# ---------------------------------------------------------



with tab5: 
        st.header("Parámetros de la Simulación")

        # Entrada de parámetros
        S0 = st.number_input("Precio actual del activo (S0)", value=100.0, min_value=0.0, step=1.0)
        K = st.number_input("Precio de ejercicio (K)", value=105.0, min_value=0.0, step=1.0)
        T = st.number_input("Tiempo hasta vencimiento (T, años)", value=1.0, min_value=0.1, step=0.1)
        r = st.number_input("Tasa libre de riesgo (r)", value=0.05, step=0.01)
        sigma = st.number_input("Volatilidad (σ)", value=0.2, step=0.01)
        N = st.number_input("Número de simulaciones (N)", value=100000, step=1000)

        # Opción para mostrar trayectorias de precios
        mostrar_paths = st.checkbox("Mostrar sample paths simulados")
        if mostrar_paths:
            n_paths = st.number_input("Número de sample paths", value=10, step=1)
            n_steps = st.number_input("Número de pasos en el tiempo", value=100, step=10)

        if st.button("Ejecutar Simulación"):
            # Semilla para reproducibilidad
            np.random.seed(42)
            
            ## SIMULACIÓN DE VALOR FINAL DEL ACTIVO (ST)
            # Generación de variables aleatorias para la simulación (valor final)
            Z = np.random.standard_normal(int(N))
            ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
            
            # Cálculo del payoff para cada simulación
            payoffs = np.maximum(ST - K, 0)
            call_price_mc = np.exp(-r * T) * np.mean(payoffs)
            
            # Cálculo del precio según la fórmula de Black-Scholes
            d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            call_price_bs = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            
            # Mostrar resultados
            st.write(f"**Precio de la opción (Monte Carlo):** {call_price_mc:.2f}")
            st.write(f"**Precio de la opción (Black-Scholes):** {call_price_bs:.2f}")
            
            ## GRÁFICO 1: Histograma de ST con línea en K
            fig1, ax1 = plt.subplots()
            ax1.hist(ST, bins=50, density=True, alpha=0.7)
            ax1.axvline(K, color='red', linestyle='dashed', linewidth=2, label=f'Precio de ejercicio (K={K})')
            ax1.set_title("Distribución de precios al vencimiento (ST)")
            ax1.set_xlabel("Precio del activo")
            ax1.set_ylabel("Densidad")
            ax1.legend()
            st.pyplot(fig1)
            
            ## GRÁFICO 2: Convergencia de la estimación de la opción
            # Se calcula la media acumulada de los payoffs descontados
            running_avg = np.cumsum(payoffs) / np.arange(1, int(N)+1)
            running_price = np.exp(-r * T) * running_avg

            fig2, ax2 = plt.subplots()
            ax2.plot(running_price, lw=1)
            ax2.axhline(call_price_mc, color='red', linestyle='dashed', linewidth=2, label=f'Valor final (MC = {call_price_mc:.2f})')
            ax2.set_title("Convergencia del precio de la opción (Monte Carlo)")
            ax2.set_xlabel("Número de simulaciones")
            ax2.set_ylabel("Precio estimado")
            ax2.legend()
            st.pyplot(fig2)
            
            ## GRÁFICO 3: Simulación de sample paths (si se selecciona)
            if mostrar_paths:
                dt = T / n_steps
                time_grid = np.linspace(0, T, n_steps+1)
                paths = np.zeros((n_paths, n_steps+1))
                paths[:, 0] = S0
                
                for i in range(n_paths):
                    # Simulación de una trayectoria
                    z = np.random.standard_normal(n_steps)
                    for j in range(1, n_steps+1):
                        paths[i, j] = paths[i, j-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z[j-1])
                
                fig3, ax3 = plt.subplots()
                for i in range(n_paths):
                    ax3.plot(time_grid, paths[i, :], lw=1, label=f'Trayectoria {i+1}' if n_paths<=10 else None)
                ax3.set_title("Sample paths simulados")
                ax3.set_xlabel("Tiempo")
                ax3.set_ylabel("Precio del activo")
                if n_paths <= 10:
                    ax3.legend()
                st.pyplot(fig3)

    
