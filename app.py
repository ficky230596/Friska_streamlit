import streamlit as st
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet.diagnostics import cross_validation, performance_metrics
import numpy as np
import io

# Set seaborn style
sns.set(style="whitegrid")

# Custom CSS for enhanced UI and animations
st.markdown("""
    <style>
    /* Main container */
    .main .block-container {
        max-width: 1200px;
        padding: 20px;
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 20px auto;
    }
    /* Title */
    h1 {
        color: #1a3c5e;
        font-size: 28px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 20px;
    }
    /* Subheader */
    h2 {
        color: #2c3e50;
        font-size: 22px;
        font-weight: 600;
        margin-bottom: 15px;
        border-bottom: 2px solid #3498db;
        padding-bottom: 5px;
    }
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .sidebar .sidebar-content h2 {
        color: #1a3c5e;
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 20px;
        text-align: center;
    }
    .sidebar .stButton>button {
        width: 100%;
        background-color: #3498db;
        color: white;
        border: none;
        padding: 12px;
        margin: 8px 0;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 500;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .sidebar .stButton>button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }
    /* Dataframe styling */
    .stDataFrame {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        overflow: hidden;
        background-color: #ffffff;
    }
    /* Messages */
    .stSuccess {
        background-color: #e6f4ea;
        border: 1px solid #28a745;
        border-radius: 8px;
        padding: 10px;
    }
    .stWarning, .stError {
        background-color: #ffe6e6;
        border: 1px solid #dc3545;
        border-radius: 8px;
        padding: 10px;
    }
    /* Plot container */
    .stPlotlyChart, .stImage {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    /* Animation for loading */
    .spinner {
        text-align: center;
        font-size: 16px;
        color: #3498db;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Cache data cleaning function
@st.cache_data
def bersihkan_data(file, tahun):
    try:
        data = pd.read_excel(file, sheet_name='Data Rekon', header=2)
        cleaned_data = data[['Kabupaten/Kota', 'Jenjang', 'Siswa', 'Dana']].copy()
        cleaned_data.columns = ['Kabupaten/Kota', 'Jenjang', 'Jumlah Siswa', 'Dana Disalurkan']
        cleaned_data['Tahun'] = tahun
        cleaned_data['Jumlah Siswa'] = pd.to_numeric(
            cleaned_data['Jumlah Siswa'].astype(str).str.replace(',', ''),
            errors='coerce'
        )
        cleaned_data['Dana Disalurkan'] = pd.to_numeric(
            cleaned_data['Dana Disalurkan'].astype(str).str.replace(',', ''),
            errors='coerce'
        )
        cleaned_data.dropna(subset=['Jumlah Siswa', 'Dana Disalurkan'], inplace=True)
        return cleaned_data
    except Exception as e:
        st.error(f"Error processing {file.name}: {e}")
        return pd.DataFrame()

# Cache file processing function
@st.cache_data
def process_files(uploaded_files):
    all_data = pd.DataFrame()
    for file in uploaded_files:
        try:
            tahun = file.name.split('Tahun')[1].split(' ')[1].split('.')[0].strip()
            cleaned_data = bersihkan_data(file, tahun)
            if not cleaned_data.empty:
                all_data = pd.concat([all_data, cleaned_data], ignore_index=True)
        except IndexError:
            st.warning(f"Skipping file {file.name}: Cannot extract year from filename.")
        except Exception as e:
            st.error(f"An unexpected error occurred while processing {file.name}: {e}")
    return all_data

# Visualization functions
def plot_tren_siswa(data):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data, x='Tahun', y='Jumlah Siswa', hue='Jenjang', marker='o', linewidth=2)
    plt.title('Tren Jumlah Siswa Penerima Dana PIP (2018-2024)', fontsize=14, pad=15)
    plt.ylabel('Jumlah Siswa', fontsize=12)
    plt.xlabel('Tahun', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Jenjang', loc='upper right')
    return plt.gcf()

def plot_tren_dana(data):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data, x='Tahun', y='Dana Disalurkan', hue='Jenjang', marker='o', linewidth=2)
    plt.title('Tren Total Dana Disalurkan PIP per Jenjang (2018-2024)', fontsize=14, pad=15)
    plt.ylabel('Dana Disalurkan (Rp)', fontsize=12)
    plt.xlabel('Tahun', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Jenjang', loc='upper right')
    return plt.gcf()

def plot_distribusi_2021(data):
    tahun_tertentu = data[data['Tahun'] == '2021']
    plt.figure(figsize=(14, 7))
    sns.barplot(data=tahun_tertentu, x='Kabupaten/Kota', y='Dana Disalurkan', hue='Jenjang')
    plt.title('Distribusi Dana Disalurkan per Kabupaten/Kota (Tahun 2021)', fontsize=14, pad=15)
    plt.ylabel('Dana Disalurkan (Rp)', fontsize=12)
    plt.xlabel('Kabupaten/Kota', fontsize=12)
    plt.xticks(rotation=90)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Jenjang', loc='upper right')
    return plt.gcf()

def plot_pivot_siswa(data):
    pivot = data.pivot_table(index='Tahun', columns='Jenjang', values='Jumlah Siswa', aggfunc='sum')
    plt.figure(figsize=(12, 6))
    pivot.plot(kind='line', marker='o', linewidth=2)
    plt.title('Pola Peningkatan/Penurunan Jumlah Siswa Penerima Dana PIP', fontsize=14, pad=15)
    plt.ylabel('Jumlah Siswa', fontsize=12)
    plt.xlabel('Tahun', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Jenjang', loc='upper right')
    return plt.gcf()

def plot_pivot_dana(data):
    pivot = data.pivot_table(index='Tahun', columns='Jenjang', values='Dana Disalurkan', aggfunc='sum')
    plt.figure(figsize=(12, 6))
    pivot.plot(kind='line', marker='o', linewidth=2)
    plt.title('Pola Peningkatan/Penurunan Dana PIP yang Disalurkan', fontsize=14, pad=15)
    plt.ylabel('Dana Disalurkan (Rp)', fontsize=12)
    plt.xlabel('Tahun', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Jenjang', loc='upper right')
    return plt.gcf()

def plot_prophet_forecast(model, forecast):
    fig = model.plot(forecast)
    plt.title('Prediksi Penyaluran Dana PIP (2025-2029)', fontsize=14, pad=15)
    plt.ylabel('Dana Disalurkan (Rp)', fontsize=12)
    plt.xlabel('Tahun', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    return fig

def plot_prophet_components(model, forecast):
    fig = model.plot_components(forecast)
    for ax in fig.axes:
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel('Tahun', fontsize=12)
        ax.set_ylabel(ax.get_ylabel(), fontsize=12)
    return fig

def plot_prediksi_vs_aktual(df_cv):
    plt.figure(figsize=(12, 6))
    plt.plot(df_cv['ds'], df_cv['y'], label='Aktual', marker='o', linewidth=2)
    plt.plot(df_cv['ds'], df_cv['yhat'], label='Prediksi', marker='x', linewidth=2)
    plt.title('Prediksi vs Data Aktual Dana PIP', fontsize=14, pad=15)
    plt.ylabel('Dana Disalurkan (Rp)', fontsize=12)
    plt.xlabel('Tahun', fontsize=12)
    plt.legend(title='Data', loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    return plt.gcf()

# Main Streamlit app
def main():
    # Sidebar navigation with buttons
    st.sidebar.markdown("<h2>Navigasi</h2>", unsafe_allow_html=True)
    sections = [
        ("Unggah File", "📤"),
        ("Data Awal", "📋"),
        ("Pivot Table", "📑"),
        ("Tren Jumlah Siswa", "📈"),
        ("Tren Dana Disalurkan", "💰"),
        ("Distribusi Dana 2021", "📊"),
        ("Pola Jumlah Siswa", "📉"),
        ("Pola Dana Disalurkan", "💸"),
        ("Prediksi Dana PIP", "🔮"),
        ("Prediksi vs Aktual", "📅"),
        ("Komponen Prediksi", "🧩")
    ]
    if 'selected_section' not in st.session_state:
        st.session_state['selected_section'] = "Data Awal"

    for sec, icon in sections:
        if st.sidebar.button(f"{icon} {sec}"):
            st.session_state['selected_section'] = sec

    section = st.session_state['selected_section']

    # File upload section
    if section == "Unggah File":
        st.markdown("<h1 style='color: #1a3c5e; text-align: center;'>Analisis dan Prediksi Dana PIP Sulawesi Utara</h1>", unsafe_allow_html=True)
        st.subheader("Unggah File Excel")
        uploaded_files = st.file_uploader("Pilih file Excel (.xlsx)", type="xlsx", accept_multiple_files=True, help="Unggah file Excel dengan sheet 'Data Rekon' dan kolom 'Kabupaten/Kota', 'Jenjang', 'Siswa', 'Dana'.")
        if not uploaded_files:
            st.info("Silakan unggah file Excel untuk memulai analisis.")
            return
        with st.spinner("Memproses file..."):
            st.session_state['all_data'] = process_files(uploaded_files)
        if st.session_state['all_data'].empty:
            st.error("No data processed. Please check the uploaded files and ensure they have the correct format.")
            return
        st.success("File berhasil diunggah dan diproses!")
        st.session_state['selected_section'] = "Data Awal"
        return

    # Exit if no data
    if 'all_data' not in st.session_state or st.session_state['all_data'].empty:
        st.warning("Silakan unggah file Excel terlebih dahulu di bagian 'Unggah File'.")
        return

    all_data = st.session_state['all_data']

    # Data initial table
    if section == "Data Awal":
        st.subheader("Data Awal (5 Baris Pertama)")
        data_head = all_data.head().copy()
        data_head['Jumlah Siswa'] = data_head['Jumlah Siswa'].map('{:.0f}'.format)
        data_head['Dana Disalurkan'] = data_head['Dana Disalurkan'].map('{:,.0f}'.format)
        st.dataframe(data_head, use_container_width=True)

    # Pivot table
    elif section == "Pivot Table":
        st.subheader("Pivot Table: Jumlah Siswa dan Dana Disalurkan per Jenjang")
        pivot_table = all_data.pivot_table(index='Tahun', columns='Jenjang', values=['Jumlah Siswa', 'Dana Disalurkan'], aggfunc='sum')
        pivot_table = pivot_table.applymap(lambda x: '{:,.0f}'.format(x) if pd.notnull(x) else '')
        st.dataframe(pivot_table, use_container_width=True)

    # Visualizations
    elif section == "Tren Jumlah Siswa":
        st.subheader("Tren Jumlah Siswa Penerima Dana PIP")
        st.pyplot(plot_tren_siswa(all_data))

    elif section == "Tren Dana Disalurkan":
        st.subheader("Tren Total Dana Disalurkan PIP per Jenjang")
        st.pyplot(plot_tren_dana(all_data))

    elif section == "Distribusi Dana 2021":
        st.subheader("Distribusi Dana Disalurkan per Kabupaten/Kota (Tahun 2021)")
        st.pyplot(plot_distribusi_2021(all_data))

    elif section == "Pola Jumlah Siswa":
        st.subheader("Pola Peningkatan/Penurunan Jumlah Siswa Penerima Dana PIP")
        st.pyplot(plot_pivot_siswa(all_data))

    elif section == "Pola Dana Disalurkan":
        st.subheader("Pola Peningkatan/Penurunan Dana PIP yang Disalurkan")
        st.pyplot(plot_pivot_dana(all_data))

    # Prophet predictions
    elif section in ["Prediksi Dana PIP", "Prediksi vs Aktual", "Komponen Prediksi"]:
        st.subheader("Prediksi Dana PIP dengan Prophet")
        data_agg = all_data.groupby('Tahun')['Dana Disalurkan'].sum().reset_index()
        data_agg_prophet = data_agg.rename(columns={'Tahun': 'ds', 'Dana Disalurkan': 'y'})
        data_agg_prophet['ds'] = pd.to_datetime(data_agg_prophet['ds'], format='%Y')

        # Handle 2024 outlier
        if '2024-01-01' in data_agg_prophet['ds'].astype(str).values:
            avg_2018_2023 = data_agg_prophet[data_agg_prophet['ds'] < '2024-01-01']['y'].mean()
            if data_agg_prophet.loc[data_agg_prophet['ds'] == '2024-01-01', 'y'].iloc[0] < avg_2018_2023 * 0.7:
                data_agg_prophet.loc[data_agg_prophet['ds'] == '2024-01-01', 'y'] = avg_2018_2023
        else:
            st.warning("Data for 2024 not found for outlier handling.")

        data_agg_prophet['floor'] = 50_000_000_000
        model = Prophet(
            yearly_seasonality=True,
            seasonality_prior_scale=5.0,
            changepoint_prior_scale=0.01
        )
        holidays = pd.DataFrame({
            'holiday': 'awal_tahun_ajaran',
            'ds': pd.to_datetime(['2018-07-01', '2019-07-01', '2020-07-01', '2021-07-01', '2022-07-01', '2023-07-01', '2024-07-01']),
            'lower_window': -7,
            'upper_window': 7
        })
        model.holidays = holidays
        model.fit(data_agg_prophet)

        # Cross-validation
        accuracy_metrics = None
        df_cv = None
        if len(data_agg_prophet) > 2:
            try:
                df_cv = cross_validation(model, initial='730 days', period='365 days', horizon='365 days', parallel="processes")
                df_p = performance_metrics(df_cv)
                rmse = df_p['rmse'].mean()
                mae = df_p['mae'].mean()
                accuracy_metrics = f"RMSE: {rmse:,.2f} | MAE: {mae:,.2f}"
            except Exception as e:
                st.error(f"Error during cross-validation: {e}")
                accuracy_metrics = "Cross-validation failed."
        else:
            accuracy_metrics = "Insufficient data for cross-validation."
            st.warning(accuracy_metrics)

        # Future predictions
        future = model.make_future_dataframe(periods=5, freq='YE')
        future['floor'] = 50_000_000_000
        forecast = model.predict(future)

        if section == "Prediksi Dana PIP":
            st.subheader("Tabel Prediksi Dana PIP (2025-2029)")
            forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            for col in ['yhat', 'yhat_lower', 'yhat_upper']:
                forecast_table[col] = forecast_table[col].map('{:,.0f}'.format)
            st.dataframe(forecast_table, use_container_width=True)
            st.markdown(f"**Metrik Akurasi Prediksi**: {accuracy_metrics}", unsafe_allow_html=True)
            st.pyplot(plot_prophet_forecast(model, forecast))

        elif section == "Prediksi vs Aktual":
            if df_cv is not None and not df_cv.empty and 'yhat' in df_cv.columns:
                st.markdown(f"**Metrik Akurasi Prediksi**: {accuracy_metrics}", unsafe_allow_html=True)
                st.pyplot(plot_prediksi_vs_aktual(df_cv))
            else:
                st.warning("Cross-validation data is empty or missing required columns.")

        elif section == "Komponen Prediksi":
            st.markdown(f"**Metrik Akurasi Prediksi**: {accuracy_metrics}", unsafe_allow_html=True)
            st.pyplot(plot_prophet_components(model, forecast))

if __name__ == "__main__":
    main()