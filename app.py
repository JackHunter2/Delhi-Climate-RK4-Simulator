import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from io import StringIO

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Delhi Climate RK4 Simulator",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# FUNGSI UTILITAS
# =========================
@st.cache_data
def load_data():
    """Load dan cache data untuk performa lebih baik"""
    df = pd.read_csv("DailyDelhiClimateTest.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

def model_temperature(t, T, k, T_eq):
    """Model persamaan diferensial untuk dinamika suhu"""
    return k * (T_eq - T)

def rk4_solver(f, t0, y0, t_end, h, params):
    """Implementasi metode Runge-Kutta Orde 4"""
    t_values = np.arange(t0, t_end + h, h)
    y_values = np.zeros(len(t_values))
    y_values[0] = y0
    
    for i in range(len(t_values) - 1):
        t = t_values[i]
        y = y_values[i]
        k1 = h * f(t, y, *params)
        k2 = h * f(t + h/2, y + k1/2, *params)
        k3 = h * f(t + h/2, y + k2/2, *params)
        k4 = h * f(t + h, y + k3, *params)
        y_values[i+1] = y + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return t_values, y_values

def calculate_metrics(y_true, y_pred):
    """Hitung metrik evaluasi model"""
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # R-squared
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R²': r2
    }

def optimize_parameters(temp_data, T0, t_end, k_range=None, T_eq_range=None):
    """Optimasi parameter menggunakan grid search"""
    if k_range is None:
        k_range = np.linspace(0.01, 0.5, 20)
    if T_eq_range is None:
        T_eq_range = np.linspace(np.min(temp_data), np.max(temp_data), 20)
    
    best_rmse = np.inf
    best_params = None
    best_sim = None
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_iterations = len(k_range) * len(T_eq_range)
    current_iteration = 0
    
    for k in k_range:
        for T_eq in T_eq_range:
            _, T_sim = rk4_solver(
                model_temperature,
                0,
                T0,
                t_end,
                1,
                params=(k, T_eq)
            )
            rmse = np.sqrt(np.mean((temp_data - T_sim)**2))
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = (k, T_eq)
                best_sim = T_sim
            
            current_iteration += 1
            progress = current_iteration / total_iterations
            progress_bar.progress(progress)
            status_text.text(f"Memproses: {current_iteration}/{total_iterations} kombinasi parameter...")
    
    progress_bar.empty()
    status_text.empty()
    
    return best_params, best_rmse, best_sim

# =========================
# LOAD DATA
# =========================
df = load_data()
temp_data = df["meantemp"].values
t_data = np.arange(len(temp_data))
T0 = temp_data[0]
t_end = len(temp_data) - 1

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Pengaturan Simulasi")

# Mode simulasi
simulation_mode = st.sidebar.radio(
    "Mode Simulasi",
    ["Manual", "Optimasi Otomatis"],
    help="Pilih mode manual untuk mengatur parameter sendiri, atau optimasi otomatis untuk mencari parameter terbaik"
)

st.sidebar.markdown("---")

if simulation_mode == "Manual":
    st.sidebar.subheader("📊 Parameter Model")
    
    k = st.sidebar.slider(
        "Laju perubahan suhu (k)",
        0.01, 0.5, 0.1,
        step=0.01,
        help="Konstanta yang menentukan kecepatan respons suhu terhadap perubahan lingkungan"
    )
    
    T_eq = st.sidebar.slider(
        "Suhu keseimbangan (T_eq) °C",
        float(temp_data.min()),
        float(temp_data.max()),
        float(temp_data.mean()),
        step=0.1,
        help="Suhu jangka panjang yang menjadi titik konvergensi sistem"
    )
    
    # Simulasi dengan parameter manual
    t_sim, T_sim = rk4_solver(
        model_temperature,
        0,
        T0,
        len(temp_data) - 1,
        1,
        params=(k, T_eq)
    )
    
else:  # Optimasi Otomatis
    st.sidebar.subheader("🔍 Pengaturan Optimasi")
    
    use_custom_range = st.sidebar.checkbox("Gunakan rentang kustom", False)
    
    if use_custom_range:
        k_min = st.sidebar.number_input("k minimum", 0.01, 0.5, 0.01, step=0.01)
        k_max = st.sidebar.number_input("k maksimum", 0.01, 0.5, 0.5, step=0.01)
        k_steps = st.sidebar.number_input("Jumlah langkah k", 5, 50, 20, step=5)
        
        T_eq_min = st.sidebar.number_input("T_eq minimum (°C)", 
                                           float(temp_data.min()), 
                                           float(temp_data.max()), 
                                           float(temp_data.min()), 
                                           step=0.1)
        T_eq_max = st.sidebar.number_input("T_eq maksimum (°C)", 
                                           float(temp_data.min()), 
                                           float(temp_data.max()), 
                                           float(temp_data.max()), 
                                           step=0.1)
        T_eq_steps = st.sidebar.number_input("Jumlah langkah T_eq", 5, 50, 20, step=5)
        
        k_range = np.linspace(k_min, k_max, k_steps)
        T_eq_range = np.linspace(T_eq_min, T_eq_max, T_eq_steps)
    else:
        k_range = None
        T_eq_range = None
    
    if st.sidebar.button("🚀 Jalankan Optimasi", type="primary"):
        with st.spinner("Mencari parameter optimal..."):
            best_params, best_rmse, T_sim = optimize_parameters(
                temp_data, T0, t_end, k_range, T_eq_range
            )
            k, T_eq = best_params
            st.session_state['optimized_params'] = {
                'k': k,
                'T_eq': T_eq,
                'rmse': best_rmse,
                'T_sim': T_sim
            }
            st.sidebar.success(f"✅ Optimasi selesai! RMSE: {best_rmse:.4f}")
    
    if 'optimized_params' in st.session_state:
        k = st.session_state['optimized_params']['k']
        T_eq = st.session_state['optimized_params']['T_eq']
        T_sim = st.session_state['optimized_params']['T_sim']
        t_sim = np.arange(len(T_sim))
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("📌 Parameter Optimal")
        st.sidebar.metric("k", f"{k:.4f}")
        st.sidebar.metric("T_eq", f"{T_eq:.2f} °C")
        st.sidebar.metric("RMSE", f"{st.session_state['optimized_params']['rmse']:.4f}")
    else:
        st.sidebar.info("👆 Klik tombol 'Jalankan Optimasi' untuk mencari parameter terbaik")
        # Default values untuk preview
        k = 0.1
        T_eq = temp_data.mean()
        t_sim, T_sim = rk4_solver(
            model_temperature,
            0,
            T0,
            t_end,
            1,
            params=(k, T_eq)
        )

# =========================
# HEADER UTAMA
# =========================
st.title("🌡️ Delhi Climate RK4 Simulator")
st.markdown("### Simulasi Dinamika Suhu Harian Menggunakan Metode Runge-Kutta Orde 4 (RK4)")

# =========================
# INFORMASI MODEL
# =========================
with st.expander("📖 Tentang Model dan Metode", expanded=False):
    st.markdown("""
    #### Model Persamaan Diferensial
    
    Model yang digunakan adalah model relaksasi suhu orde satu:
    
    $$
    \\frac{dT}{dt} = k (T_{eq} - T)
    $$
    
    **Parameter:**
    - $T(t)$: Suhu harian pada waktu $t$
    - $k$: Konstanta laju perubahan suhu (menentukan kecepatan respons)
    - $T_{eq}$: Suhu keseimbangan lingkungan (titik konvergensi jangka panjang)
    
    #### Metode Runge-Kutta Orde 4 (RK4)
    
    RK4 adalah metode numerik yang menggunakan empat estimasi kemiringan pada setiap langkah waktu:
    
    $$
    k_1 = h \\cdot f(t_n, y_n)
    $$
    $$
    k_2 = h \\cdot f(t_n + \\frac{h}{2}, y_n + \\frac{k_1}{2})
    $$
    $$
    k_3 = h \\cdot f(t_n + \\frac{h}{2}, y_n + \\frac{k_2}{2})
    $$
    $$
    k_4 = h \\cdot f(t_n + h, y_n + k_3)
    $$
    $$
    y_{n+1} = y_n + \\frac{k_1 + 2k_2 + 2k_3 + k_4}{6}
    $$
    
    Metode ini memberikan akurasi orde 4 dengan stabilitas numerik yang baik.
    """)

# =========================
# STATISTIK DATA
# =========================
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("📅 Total Hari", len(temp_data))
with col2:
    st.metric("🌡️ Suhu Awal", f"{T0:.2f} °C")
with col3:
    st.metric("📊 Suhu Rata-rata", f"{temp_data.mean():.2f} °C")
with col4:
    st.metric("⬆️ Suhu Maksimum", f"{temp_data.max():.2f} °C")
with col5:
    st.metric("⬇️ Suhu Minimum", f"{temp_data.min():.2f} °C")

st.markdown("---")

# =========================
# METRIK EVALUASI
# =========================
metrics = calculate_metrics(temp_data, T_sim)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📉 RMSE", f"{metrics['RMSE']:.4f} °C", 
              help="Root Mean Square Error - mengukur kesalahan rata-rata")
with col2:
    st.metric("📊 MAE", f"{metrics['MAE']:.4f} °C",
              help="Mean Absolute Error - rata-rata kesalahan absolut")
with col3:
    st.metric("📈 MAPE", f"{metrics['MAPE']:.2f}%",
              help="Mean Absolute Percentage Error - kesalahan dalam persentase")
with col4:
    st.metric("🎯 R² Score", f"{metrics['R²']:.4f}",
              help="Koefisien determinasi - seberapa baik model menjelaskan variasi data")

st.markdown("---")

# =========================
# VISUALISASI INTERAKTIF
# =========================
st.subheader("📈 Visualisasi Hasil Simulasi")

# Plot utama
fig = go.Figure()

# Pastikan panjang array sama untuk plotting
min_len_plot = min(len(t_data), len(temp_data), len(t_sim), len(T_sim))

# Data asli
fig.add_trace(go.Scatter(
    x=t_data[:min_len_plot],
    y=temp_data[:min_len_plot],
    mode='markers',
    name='Data Asli',
    marker=dict(
        color='#1f77b4',
        size=4,
        opacity=0.6
    ),
    hovertemplate='<b>Hari %{x}</b><br>Suhu: %{y:.2f} °C<extra></extra>'
))

# Simulasi RK4
fig.add_trace(go.Scatter(
    x=t_sim[:min_len_plot],
    y=T_sim[:min_len_plot],
    mode='lines',
    name='Simulasi RK4',
    line=dict(
        color='#ff7f0e',
        width=2
    ),
    hovertemplate='<b>Hari %{x}</b><br>Suhu Simulasi: %{y:.2f} °C<extra></extra>'
))

fig.update_layout(
    title="Perbandingan Data Asli dan Simulasi RK4",
    xaxis_title="Hari",
    yaxis_title="Suhu (°C)",
    hovermode='closest',
    template='plotly_white',
    height=500,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

st.plotly_chart(fig, use_container_width=True)

# Plot residual
st.subheader("📊 Analisis Residual")
# Pastikan panjang array sama
min_len = min(len(temp_data), len(T_sim))
residuals = temp_data[:min_len] - T_sim[:min_len]

col1, col2 = st.columns(2)

with col1:
    fig_res = go.Figure()
    min_len_res = min(len(t_data), len(residuals))
    fig_res.add_trace(go.Scatter(
        x=t_data[:min_len_res],
        y=residuals[:min_len_res],
        mode='markers+lines',
        name='Residual',
        marker=dict(color='#2ca02c', size=4),
        line=dict(color='#2ca02c', width=1)
    ))
    fig_res.add_hline(y=0, line_dash="dash", line_color="red", 
                     annotation_text="Nol")
    fig_res.update_layout(
        title="Residual Plot",
        xaxis_title="Hari",
        yaxis_title="Residual (Data - Simulasi) °C",
        template='plotly_white',
        height=400
    )
    st.plotly_chart(fig_res, use_container_width=True)

with col2:
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=residuals,
        nbinsx=30,
        name='Distribusi Residual',
        marker_color='#9467bd'
    ))
    fig_hist.update_layout(
        title="Histogram Residual",
        xaxis_title="Residual (°C)",
        yaxis_title="Frekuensi",
        template='plotly_white',
        height=400
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# =========================
# TABEL HASIL
# =========================
st.subheader("📋 Tabel Hasil Simulasi")

# Pastikan semua array memiliki panjang yang sama
min_len = min(len(t_data), len(df['date'].values), len(temp_data), len(T_sim), len(residuals))
results_df = pd.DataFrame({
    'Hari': t_data[:min_len],
    'Tanggal': df['date'].values[:min_len],
    'Data Asli (°C)': temp_data[:min_len],
    'Simulasi RK4 (°C)': T_sim[:min_len],
    'Residual (°C)': residuals[:min_len],
    'Error Absolut': np.abs(residuals[:min_len])
})

st.dataframe(
    results_df.style.format({
        'Data Asli (°C)': '{:.2f}',
        'Simulasi RK4 (°C)': '{:.2f}',
        'Residual (°C)': '{:.2f}',
        'Error Absolut': '{:.2f}'
    }),
    use_container_width=True,
    height=400
)

# =========================
# DOWNLOAD HASIL
# =========================
st.markdown("---")
st.subheader("💾 Download Hasil")

col1, col2 = st.columns(2)

with col1:
    # Download CSV
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Hasil sebagai CSV",
        data=csv,
        file_name=f"delhi_climate_rk4_simulation_k{k:.4f}_Teq{T_eq:.2f}.csv",
        mime="text/csv"
    )

with col2:
    # Download summary
    summary_text = f"""
DELHI CLIMATE RK4 SIMULATION SUMMARY
=====================================

Parameter Model:
- k (Laju perubahan suhu): {k:.6f}
- T_eq (Suhu keseimbangan): {T_eq:.2f} °C

Metrik Evaluasi:
- RMSE: {metrics['RMSE']:.4f} °C
- MAE: {metrics['MAE']:.4f} °C
- MAPE: {metrics['MAPE']:.2f}%
- R² Score: {metrics['R²']:.4f}

Statistik Data:
- Total Hari: {len(temp_data)}
- Suhu Awal: {T0:.2f} °C
- Suhu Rata-rata: {temp_data.mean():.2f} °C
- Suhu Maksimum: {temp_data.max():.2f} °C
- Suhu Minimum: {temp_data.min():.2f} °C
"""
    st.download_button(
        label="📄 Download Summary",
        data=summary_text,
        file_name=f"delhi_climate_rk4_summary.txt",
        mime="text/plain"
    )

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>🌡️ Delhi Climate RK4 Simulator | Metode Runge-Kutta Orde 4</p>
        <p>Dibuat dengan ❤️ menggunakan Streamlit dan Plotly</p>
    </div>
    """,
    unsafe_allow_html=True
)