import streamlit as st #Untuk membuat antarmuka web (dashboard) secara instan
import tensorflow as tf #Framework utama untuk memproses model Deep Learning
import joblib #Untuk memuat (load) objek Python yang disimpan dalam file eksternal
import os #Untuk manajemen direktori/folder
import numpy as np #Untuk manipulasi struktur data matriks (Array) sebagai input wajib model TensorFlow
import pandas as pd #Untuk membaca dan mengolah data berbentuk tabel (DataFrame)
import matplotlib.pyplot as plt #Untuk membuat grafik penampil data

# Metrik statistik evaluasi dan pembantu
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 
from sklearn.preprocessing import MinMaxScaler ##Skalasi Data (Normalisasi)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, LSTM, Dense, Dropout #lapisan jaringan saraf untuk menyusun struktur model LSTM
from tensorflow.keras.saving import register_keras_serializable #Untuk memuat model ke dalam sistem tensorflow
from tensorflow.keras.callbacks import EarlyStopping

#Setup awal tampilan aplikasi web (pake mode layar lebar)
st.set_page_config(
    page_title="PharmaPredict",
    page_icon="💊",
    layout="wide"
)

#Attention Mechanism layer - Diselaraskan dengan struktur kode training
@register_keras_serializable()
class AttentionLayer(Layer):

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="normal"
        )

        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros"
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(
            tf.keras.backend.dot(x, self.W) + self.b
        )
        a = tf.keras.backend.softmax(e, axis=1)
        output = tf.keras.backend.sum(x * a, axis=1)
        return output

    def get_config(self):
        config = super().get_config()
        return config

#CSS Custom Interface
st.markdown("""
<style>
html, body, [class*="css"]{
    font-family: 'Segoe UI', sans-serif;
}
.stApp{
    background:#f3f4f6;
}
#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
header{visibility:hidden;}

/* NAVBAR */
.navbar{
    background:white;
    padding:18px 30px;
    margin:-1rem -1rem 25px -1rem;
    border-bottom:1px solid #e5e7eb;
    display:flex;
    justify-content:space-between;
    align-items:center;
}
.logo-wrap{
    display:flex;
    align-items:center;
    gap:10px;
}
.logo-icon{
    width:28px;
    height:28px;
    border-radius:6px;
    background:#2563eb;
    color:white;
    display:flex;
    align-items:center;
    justify-content:center;
    font-weight:700;
}
.logo-text{
    font-size:20px;
    font-weight:700;
    color:#111827;
}

/* HERO */
.hero-box{
    background:white;
    border:2px solid #2563eb;
    border-radius:14px;
    padding:55px;
}
.hero-mini{
    color:#2563eb;
    font-size:12px;
    letter-spacing:1.2px;
    font-weight:700;
}
.hero-title{
    font-size:52px;
    line-height:1.15;
    font-weight:800;
    margin-top:18px;
    color:#111827;
}
.hero-desc{
    margin-top:24px;
    color:#4b5563;
    line-height:2;
    font-size:17px;
}

/* PANEL */
.panel{
    background:white;
    border:1px solid #e5e7eb;
    border-radius:14px;
    padding:24px;
}

/* KARTU METRIK DIUPDATE AGAR PRESISI UNTUK 5 KOLOM METRIK */
.metric-card{
    background:white;
    border:1px solid #e5e7eb;
    border-radius:12px;
    padding:16px;
    min-height: 110px;
    height: auto;
}
.metric-title{
    font-size:12px;
    color:#6b7280;
    line-height:1.4;
    font-weight: 600;
}
.metric-value{
    margin-top:8px;
    font-size:20px; 
    font-weight:700;
    color:#111827;
    white-space: nowrap; 
}

.profile-card{
    background:#eef2ff;
    border-radius:14px;
    padding:35px;
    text-align:center;
    height:100%;
}
.footer{
    margin-top:30px;
    text-align:center;
    color:#6b7280;
    font-size:13px;
}
.stButton>button{
    background:#2563eb;
    color:white;
    border:none;
    border-radius:8px;
    height:45px;
    font-weight:600;
}
.stSelectbox label,
.stDateInput label,
.stFileUploader label{
    font-weight:600;
}
</style>
""", unsafe_allow_html=True)

# Mengatur lokasi folder utama dan sub-folder
BASE_DIR = r"C:/Users/evald/OneDrive/Documents/Skripsi"
MODEL_DIR = os.path.join(BASE_DIR, "saved_model", "model")
SCALER_DIR = os.path.join(BASE_DIR, "saved_model", "scaler")
DATA_DIR = os.path.join(BASE_DIR, "Dataset")

# Menampilkan Logo dan nama aplikasi di bagian paling atas dashboard
st.markdown("""
<div class="navbar">
<div class="logo-wrap">
<div class="logo-icon">P</div>
<div class="logo-text">PharmaPredict</div>
</div>
</div>
""", unsafe_allow_html=True)

# Membuat menu navigasi halaman di sebelah kiri (sidebar)
page = st.sidebar.radio(
    "Menu Navigasi",
    [
        "Halaman Utama",
        "Pengujian & Prediksi",
        "Uji Data Kustom",
        "Info Pembuat"
    ]
)

# Fungsi bantuan buat benerin format data saham
def clean_currency(x):
    if isinstance(x, str):
        x = x.replace('.', '').replace(',', '.')
        return float(x)
    return x

def clean_volume(x):
    if isinstance(x, str):
        x = x.replace('.', '').replace(',', '.')
        if 'M' in x:
            return float(x.replace('M', '')) * 1_000_000
        if 'K' in x:
            return float(x.replace('K', '')) * 1_000
        try:
            return float(x)
        except:
            return 0.0
    return x

# =====================================================================
# SINKRONISASI: Menyelaraskan inverse transform dengan skrip Univariate
# =====================================================================
def inverse_transform_target(y_pred, scaler_obj):
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    n_days = y_pred.shape[1]
    result = []

    # Proses inverse dilakukan langsung per hari target karena scaler dikonfigurasi hanya untuk 1 fitur
    for i in range(n_days):
        col_data = y_pred[:, i].reshape(-1, 1)
        inv = scaler_obj.inverse_transform(col_data)[:, 0]
        result.append(inv)

    return np.array(result).T

# Halaman Utama
if page == "Halaman Utama":
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("""
        <div class="hero-box">
        <div class="hero-mini">BURSA EFEK INDONESIA (BEI)</div>
        <div class="hero-title">Sistem Prediksi Harga<br>Saham Sektor Farmasi</div>
        <div class="hero-desc">
        Platform berbasis Deep Learning yang mengintegrasikan arsitektur Long Short-Term Memory (LSTM)
        dan Attention Mechanism. Dirancang khusus untuk memproses data non-linear dan volatil demi membantu investor mengambil 
        keputusan yang objektif dan memitigasi risiko pasar.
        </div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        fig, ax = plt.subplots(figsize=(5, 4))
        data = [100, 112, 108, 105, 130, 122, 150]
        ax.plot(data, linewidth=3, color="#2563eb")
        ax.grid(True, alpha=0.2)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        st.pyplot(fig)

# Halaman Pengujian & Prediksi
elif page == "Pengujian & Prediksi":
    left, right = st.columns([1, 2.5]) 

    with left:
        st.markdown("""
        <div class="panel">
        <h3>Parameter Prediksi</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Mapping nama emiten agar sesuai dengan format nama file hasil training Anda
        stock_selection = st.selectbox(
            "Pilih Emiten Farmasi",
            [
                "Kalbe Farma Tbk PT (KLBF)", 
                "Phapros Tbk PT (PEHA)", 
                "Sido Muncul Tbk PT (SIDO)", 
                "Soho Global Health Tbk Pt (SOHO)" 
            ]
        )
        
        # Mengambil kode singkat emiten untuk pencarian dataset csv instan
        stock_code = stock_selection.split('(')[-1].replace(')', '').strip()

        predict_btn = st.button(
            "Jalankan Prediksi",
            use_container_width=True
        )

    with right:
        if predict_btn:
            try:
                # Membaca Data historis emiten terpilih berdasarkan kode singkat
                file_name = f"Data Historis {stock_code}.csv"
                full_path = os.path.join(DATA_DIR, file_name)

                if not os.path.exists(full_path):
                    st.error(f"Dataset {file_name} tidak ditemukan di folder {DATA_DIR}")
                    st.stop()

                df = pd.read_csv(full_path, sep=';')
                df.columns = [c.replace('\ufeff', '').strip() for c in df.columns]

                if 'Volume' in df.columns:
                    df.rename(columns={'Volume': 'Vol.'}, inplace=True)

                cols_to_fix = ['Terakhir', 'Pembukaan', 'Tertinggi', 'Terendah']
                for col in cols_to_fix:
                    df[col] = df[col].apply(clean_currency).astype(float)

                df['Vol.'] = df['Vol.'].apply(clean_volume)
                df['Tanggal'] = pd.to_datetime(df['Tanggal'], dayfirst=True)
                df = df.sort_values('Tanggal')

                win = 30 # Konfigurasi lookback window tetap 30 hari

                # Validasi jumlah minimum baris data (Lookback 30 + Target Horizon 7)
                if len(df) < (win + 7):
                    st.error(f"Data historis terlalu sedikit. Minimal dibutuhkan {win + 7} baris data.")
                    st.stop()

                # Memanggil Model Keras dan scaler pkl sesuai nama lengkap stock hasil training
                model_path = os.path.join(MODEL_DIR, f"{stock_selection}_lstm_attention.keras")
                scaler_path = os.path.join(SCALER_DIR, f"{stock_selection}_scaler.pkl")

                if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                    st.error(f"File Model atau Scaler untuk emiten '{stock_selection}' belum tersedia di folder saved_model.")
                    st.stop()

                model = tf.keras.models.load_model(
                    model_path,
                    custom_objects={'AttentionLayer': AttentionLayer}
                )
                scaler = joblib.load(scaler_path)
                
                # =====================================================================
                # SINKRONISASI: Mengubah Pemrosesan Menjadi Univariate (1 Fitur: 'Terakhir')
                # =====================================================================
                dataset = df[['Terakhir']].values
                scaled_data = scaler.transform(dataset)

                if scaled_data.ndim == 1:
                    scaled_data = scaled_data.reshape(-1, 1)

                # 1. PROSES PREDIKSI MASA DEPAN (Menggunakan data 30 hari terakhir dengan bentuk [1, 30, 1])
                last_window = scaled_data[-win:].reshape(1, win, 1)
                future_pred = model.predict(last_window, verbose=0)
                future_prices = inverse_transform_target(future_pred, scaler).flatten()
                pred_h7 = future_prices[6] # Ambil Hari ke-7 (Indeks 6)

                # 2. PROSES BACKTESTING DATA UJI SINKRON DENGAN BACKEND
                X_u = []
                y_u = []
                for i in range(len(scaled_data) - win - 7 + 1):
                    X_u.append(scaled_data[i:i+win])
                    y_u.append(scaled_data[i+win:i+win+7, 0])

                X_u = np.array(X_u)
                y_u = np.array(y_u)

                # Pembagian data latih/uji berdasarkan panjang total DataFrame asli agar sekuensial ideal
                total_len = len(df)
                split_idx = int(0.8 * total_len)
                split = split_idx - win
                
                # Memisahkan Data Uji (Test) dan Data Latih (Train)
                X_test = X_u[split:]
                y_test = y_u[split:]
                X_train = X_u[:split] # <--- Tambahan untuk Train Data
                y_train = y_u[:split] # <--- Tambahan untuk Train Data
                
                # Melakukan Prediksi
                preds_test = model.predict(X_test, verbose=0)
                preds_train = model.predict(X_train, verbose=0) # <--- Prediksi Train Data
                
                # Mengembalikan skala data (Inverse Transform)
                y_test_inv = inverse_transform_target(y_test, scaler)
                preds_test_inv = inverse_transform_target(preds_test, scaler)
                
                y_train_inv = inverse_transform_target(y_train, scaler) # <--- Inverse Train Data
                preds_train_inv = inverse_transform_target(preds_train, scaler) # <--- Inverse Train Data
                
                # Ekstraksi Hari ke-7 (H+7) untuk Data Uji (RMSE, MAE, MAPE)
                y_true_h7 = y_test_inv[:, 6]
                y_pred_h7 = preds_test_inv[:, 6]
                
                # Ekstraksi Hari ke-7 (H+7) untuk Data Latih (KHUSUS R2 Score)
                y_true_train_h7 = y_train_inv[:, 6]
                y_pred_train_h7 = preds_train_inv[:, 6]
                
                # Mengantisipasi division by zero pas hitung MAPE
                y_true_safe = np.where(y_true_h7 == 0, 1e-8, y_true_h7)
                
                # Kalkulasi metrik akurasi evaluasi model
                rmse = np.sqrt(mean_squared_error(y_true_h7, y_pred_h7))
                mae = mean_absolute_error(y_true_h7, y_pred_h7)
                mape = np.mean(np.abs((y_true_h7 - y_pred_h7) / y_true_safe)) * 100
                
                # R2 SCORE DIHITUNG MENGGUNAKAN DATA TRAIN
                r2 = r2_score(y_true_train_h7, y_pred_train_h7)

                # Render metrik evaluasi ke dalam komponen interface dashboard (5 kolom)
                c1, c2, c3, c4, c5 = st.columns(5)
                
                cards = [
                    ("Estimasi Harga H+7", f"Rp {pred_h7:,.2f}", "#2563eb"),
                    ("RMSE H+7", f"{rmse:.2f}", "#111827"),
                    ("MAE H+7", f"{mae:.2f}", "#111827"),
                    ("MAPE H+7", f"{mape:.2f}%", "#111827"),
                    ("R² Score H+7", f"{r2:.4f}", "#111827")
                ]

                for col, card in zip([c1, c2, c3, c4, c5], cards):
                    with col:
                        st.markdown(f"""
                        <div class="metric-card">
                        <div class="metric-title">{card[0]}</div>
                        <div class="metric-value" style="color:{card[2]};">{card[1]}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Visualisasi Bagian 1: Perbandingan Aktual vs Prediksi H+7 (Data Test)
                st.markdown("""
                <div class="panel">
                <h3>Visualisasi Perbandingan Harga Aktual vs Prediksi H+7</h3>
                </div>
                """, unsafe_allow_html=True)

                fig1, ax1 = plt.subplots(figsize=(14, 5))
                ax1.plot(y_true_h7, label='Harga Aktual H+7', color='black', alpha=0.5, linewidth=2)
                ax1.plot(y_pred_h7, label='Harga Prediksi H+7', color='red', linestyle='--', linewidth=2)
                ax1.set_title(f'Perbandingan Harga Aktual vs Prediksi H+7 {stock_selection}')
                ax1.set_xlabel('Data Points')
                ax1.set_ylabel('Harga Saham')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                st.pyplot(fig1, use_container_width=True)
                plt.close(fig1)

                st.markdown("<br>", unsafe_allow_html=True)

                # Visualisasi Bagian 2: Proyeksi Tren Harga ke Depan (7 Hari ke Depan)
                st.markdown("""
                <div class="panel">
                <h3>Proyeksi Tren Nilai Saham 7 Hari ke Depan</h3>
                </div>
                """, unsafe_allow_html=True)

                fig2, ax2 = plt.subplots(figsize=(10, 5))
                x_days = np.arange(1, 8)
                ax2.plot(x_days, future_prices, marker='o', linestyle='-', color='green', linewidth=2, label='Forecast H+1 s/d H+7')
                ax2.set_xticks(x_days)
                ax2.set_xticklabels([f'H+{i}' for i in range(1, 8)])
                ax2.set_title(f'Proyeksi Harga {stock_selection} 7 Hari ke Depan')
                ax2.set_xlabel('Hari')
                ax2.set_ylabel('Harga Prediksi')
                
                for idx, price in enumerate(future_prices):
                    ax2.text(idx + 1, price, f'{price:.2f}', ha='center', va='bottom', fontsize='small', weight='bold')
                
                ax2.legend()
                ax2.grid(True, linestyle=':', alpha=0.7)
                plt.tight_layout()
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)

            except Exception as e:
                st.error(f"Terjadi masalah pemrosesan data: {e}")

# =========================================================================================
# HALAMAN UJI DATA KUSTOM (SUDAH DIUBAH MENJADI MODEL DYNAMIC ON-THE-FLY SINKRON 80:20)
# =========================================================================================
elif page == "Uji Data Kustom":
    left, right = st.columns([1, 2.5]) 

    with left:
        st.markdown("""
        <div class="panel">
        <h3>Pengujian Data Kustom</h3>
        <p style="font-size:13px; color:#6b7280;">Unggah file dataset kustom Anda (.csv) dengan jumlah baris berapa pun. Sistem akan melatih model LSTM + Attention baru khusus untuk menganalisis karakteristik data yang diunggah.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Fitur Upload File CSV
        uploaded_file = st.file_uploader("Unggah File Dataset (.csv)", type=["csv"])

        predict_btn = st.button(
            "Jalankan Training & Prediksi Baru",
            use_container_width=True
        )

    with right:
        if predict_btn:
            if uploaded_file is None:
                st.warning("⚠️ Silakan unggah file CSV terlebih dahulu di panel sebelah kiri.")
            else:
                try:
                    # 1. Baca data langsung dari file yang diunggah (Mendukung pembatas koma atau titik koma)
                    try:
                        df = pd.read_csv(uploaded_file, sep=';')
                        if len(df.columns) <= 1:
                            df = pd.read_csv(uploaded_file, sep=',')
                    except:
                        df = pd.read_csv(uploaded_file)
                    
                    # Bersihkan nama kolom dari karakter tersembunyi
                    df.columns = [c.replace('\ufeff', '').strip() for c in df.columns]

                    # Standarisasi nama kolom volume jika ada
                    if 'Volume' in df.columns:
                        df.rename(columns={'Volume': 'Vol.'}, inplace=True)

                    cols_to_fix = ['Terakhir', 'Pembukaan', 'Tertinggi', 'Terendah']
                    
                    # Validasi ketersediaan kolom
                    missing_cols = [col for col in cols_to_fix if col not in df.columns]
                    if missing_cols:
                        st.error(f"File gagal diproses. Kolom berikut tidak ditemukan: {', '.join(missing_cols)}")
                        st.stop()

                    # 2. Proses pembersihan data mata uang & filter agar tidak ada angka 0
                    for col in cols_to_fix:
                        df[col] = df[col].astype(str).apply(clean_currency).astype(float)

                    if 'Vol.' in df.columns:
                        df['Vol.'] = df['Vol.'].apply(clean_volume)
                    
                    if 'Tanggal' in df.columns:
                        df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
                        df = df.dropna(subset=['Tanggal', 'Terakhir'])
                        df = df[df['Terakhir'] > 0] # Proteksi nilai 0 penyebab MAPE hancur
                        df = df.sort_values('Tanggal').reset_index(drop=True)

                    st.success(f"✅ Pembersihan Berhasil! Model akan dilatih menggunakan {len(df)} baris data kustom.")

                    win = 30 # Konfigurasi lookback window tetap 30 hari
                    horizon = 7

                    if len(df) < (win + horizon + 5):
                        st.error(f"Data historis terlalu sedikit. Minimal dibutuhkan {win + horizon + 5} baris data.")
                        st.stop()

                    # 3. Pemrosesan Univariate & Inisialisasi Scaler Baru khusus data kustom ini
                    dataset = df[['Terakhir']].values
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled_data = scaler.fit_transform(dataset)

                    # --- PROSES PEMBENTUKAN SLIDING WINDOWS ---
                    X_u, y_u = [], []
                    for i in range(len(scaled_data) - win - horizon + 1):
                        X_u.append(scaled_data[i:i+win])
                        y_u.append(scaled_data[i+win:i+win+horizon, 0])

                    X_u, y_u = np.array(X_u), np.array(y_u)

                    # --- PROSES SPLIT DATA 80:20 SINKRON KHUSUS DATA BARU (Sesuai Permintaan: Tidak Diubah) ---
                    split_idx = int(0.8 * len(X_u))
                    X_train, X_test = X_u[:split_idx], X_u[split_idx:]
                    y_train, y_test = y_u[:split_idx], y_u[split_idx:]

                    st.info(f"📊 Pembagian Sekuensial Data Kustom: Train Set = {len(X_train)} | Test Set = {len(X_test)}")

                    with st.spinner("🔄 Sedang melatih Arsitektur LSTM + Attention baru di latar belakang... Mohon tunggu..."):
                        # --- MEMBANGUN MODEL SECARA ON-THE-FLY SINKRON DENGAN BACKEND ---
                        inputs = tf.keras.Input(shape=(win, 1))
                        
                        # 1 Layer LSTM (32 Units) tanpa Dropout
                        lstm = tf.keras.layers.LSTM(units=32, return_sequences=True)(inputs)
                        attn_out = AttentionLayer()(lstm)
                        outputs = tf.keras.layers.Dense(units=horizon)(attn_out)
                        
                        model = tf.keras.Model(inputs=inputs, outputs=outputs)
                        model.compile(optimizer='adam', loss='mean_squared_error')

                        # Early stopping memantau val_loss hasil split internal data train
                        early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
                        model.fit(
                            X_train, 
                            y_train, 
                            validation_split=0.2, 
                            epochs=150, 
                            batch_size=32, 
                            shuffle=False, 
                            verbose=1, 
                            callbacks=[early_stop]
                        )

                    st.success("✨ Model Baru berhasil dilatih optimal!")

                    # --- PROSES PREDIKSI MASA DEPAN H+1 s/d H+7 ---
                    last_window = scaled_data[-win:].reshape(1, win, 1)
                    future_pred = model.predict(last_window, verbose=0)
                    future_prices = inverse_transform_target(future_pred, scaler).flatten()
                    pred_h7 = future_prices[6]

                    # --- PROSES EVALUASI DATA TEST (20%) ---
                    preds_test = model.predict(X_test, verbose=0)
                    preds_train = model.predict(X_train, verbose=0) # Prediksi Train Data
                    
                    y_test_inv = inverse_transform_target(y_test, scaler)
                    preds_test_inv = inverse_transform_target(preds_test, scaler)
                    
                    y_train_inv = inverse_transform_target(y_train, scaler) # Inverse Train Data
                    preds_train_inv = inverse_transform_target(preds_train, scaler) # Inverse Train Data
                    
                    # Isolasi target uji
                    y_true_h7 = y_test_inv[:, 6]
                    y_pred_h7 = preds_test_inv[:, 6]
                    y_true_safe = np.where(y_true_h7 == 0, 1e-8, y_true_h7)
                    
                    # Isolasi target latih khusus R2
                    y_true_train_h7 = y_train_inv[:, 6]
                    y_pred_train_h7 = preds_train_inv[:, 6]
                    
                    # Kalkulasi metrik akurasi evaluasi model kustom
                    rmse = np.sqrt(mean_squared_error(y_true_h7, y_pred_h7))
                    mae = mean_absolute_error(y_true_h7, y_pred_h7)
                    mape = np.mean(np.abs((y_true_h7 - y_pred_h7) / y_true_safe)) * 100
                    
                    # R2 SCORE DIHITUNG MENGGUNAKAN DATA TRAIN
                    r2 = r2_score(y_true_train_h7, y_pred_train_h7)

                    # --- RENDER METRIK EVALUASI ---
                    c1, c2, c3, c4, c5 = st.columns(5)
                    cards = [
                        ("Estimasi Harga H+7", f"Rp {pred_h7:,.2f}", "#059669"), 
                        ("RMSE H+7", f"{rmse:.2f}", "#111827"),
                        ("MAE H+7", f"{mae:.2f}", "#111827"),
                        ("MAPE H+7", f"{mape:.2f}%", "#111827"),
                        ("R² Score H+7", f"{r2:.4f}", "#111827")
                    ]

                    for col, card in zip([c1, c2, c3, c4, c5], cards):
                        with col:
                            st.markdown(f"""
                            <div class="metric-card">
                            <div class="metric-title">{card[0]}</div>
                            <div class="metric-value" style="color:{card[2]};">{card[1]}</div>
                            </div>
                            """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # --- VISUALISASI GRAFIK AKTUAL VS PRED ---
                    st.markdown("""
                    <div class="panel">
                    <h3>Visualisasi Perbandingan Harga Aktual vs Prediksi Data Kustom</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Ambil komponen tanggal murni dari df (bukan df_dates global) agar sinkron dengan file yang di-upload
                    test_dates_h7 = df['Tanggal'].iloc[split_idx + 6 : split_idx + 6 + len(preds_test_inv)].reset_index(drop=True)
                    
                    fig1, ax1 = plt.subplots(figsize=(14, 6))
                    
                    # Plot data aktual dan prediksi menggunakan koordinat murni (.values)
                    ax1.plot(test_dates_h7.values, y_true_h7, label='Harga Aktual H+7', color='black', alpha=0.5, linewidth=1.5)
                    ax1.plot(test_dates_h7.values, y_pred_h7, label='Harga Prediksi H+7', color='red', linestyle='--', linewidth=2)
                    
                    ax1.set_title('Perbandingan Harga Aktual vs Prediksi H+7 Data Kustom\n', fontsize=14)
                    ax1.set_xlabel('Tanggal', fontsize=12)
                    ax1.set_ylabel('Harga Saham', fontsize=12)
                    
                    # Optimalisasi Skala Sumbu Y secara lokal
                    y_min = min(np.min(y_true_h7), np.min(y_pred_h7))
                    y_max = max(np.max(y_true_h7), np.max(y_pred_h7))
                    buffer = (y_max - y_min) * 0.2  
                    ax1.set_ylim(y_min - buffer, y_max + buffer)
                    
                    # Kelola rotasi label tanggal sumbu X
                    ax1.tick_params(axis='x', rotation=45)
                    ax1.legend(fontsize=10)
                    ax1.grid(True, linestyle='--', alpha=0.7)
                    
                    fig1.tight_layout() 
                    st.pyplot(fig1, use_container_width=True)
                    plt.close(fig1)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # --- VISUALISASI PROYEKSI DEPAN ---
                    st.markdown("""
                    <div class="panel">
                    <h3>Proyeksi Tren Nilai Saham 7 Hari ke Depan (Data Kustom)</h3>
                    </div>
                    """, unsafe_allow_html=True)

                    fig2, ax2 = plt.subplots(figsize=(10, 5))
                    x_days = np.arange(1, 8)
                    ax2.plot(x_days, future_prices, marker='o', linestyle='-', color='green', linewidth=2, label='Forecast H+1 s/d H+7')
                    ax2.set_xticks(x_days)
                    ax2.set_xticklabels([f'H+{i}' for i in range(1, 8)])
                    ax2.set_title('Proyeksi Finansial 7 Hari ke Depan dari Titik Akhir Data Kustom')
                    ax2.set_xlabel('Hari Proyeksi')
                    ax2.set_ylabel('Harga Prediksi')
                    
                    for idx, price in enumerate(future_prices):
                        ax2.text(idx + 1, price, f'{price:.2f}', ha='center', va='bottom', fontsize='small', weight='bold')
                    
                    ax2.legend()
                    ax2.grid(True, linestyle=':', alpha=0.7)
                    plt.tight_layout()
                    st.pyplot(fig2, use_container_width=True)
                    plt.close(fig2)

                except Exception as e:
                    st.error(f"Terjadi kesalahan teknis saat memproses file kustom: {e}")

# Halaman Info Pembuat
elif page == "Info Pembuat":
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("""
        <div class="profile-card">
        <h1 style="font-size:70px;">👤</h1>
        <h2>Rafael Evaldo Setianto</h2>
        <p style="color:#2563eb;font-weight:700;">NIM: 535220086</p>
        <hr>
        <p><b>Program Studi</b><br>Teknik Informatika</p>
        <p><b>Fakultas</b><br>Teknologi Informasi</p>
        <p><b>Instansi</b><br>Universitas Tarumanagara</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="panel">
        <h1>Profil Peneliti</h1>
        <h3 style="color:#2563eb;">FOKUS PENELITIAN</h3>
        <p>
        Mahasiswa Program Studi Teknik Informatika yang berfokus pada penerapan Data Science
        dan Deep Learning dalam sektor pasar modal. Sistem ini dikembangkan dengan
        mengedepankan akurasi prediksi menggunakan metode Long Short-Term Memory (LSTM) untuk
        membantu analisis pergerakan harga saham.
        </p>
        <h3 style="color:#2563eb;">ABSTRAK SISTEM</h3>
        <p>
        Aplikasi ini dirancang untuk mengatasi volatilitas tinggi pada harga saham perusahaan
        farmasi melalui pemodelan deret waktu (Time Series). Dengan menggabungkan Long Short-
        Term Memory (LSTM) untuk memori jangka panjang dan Attention Mechanism untuk pembobotan
        data penting, sistem ini memberikan insight objektif bagi para pemangku kepentingan.  
        </p>
        <h3 style="color:#2563eb;">KONTAK & INFORMASI</h3>
        <p>
        Untuk diskusi lebih lanjut mengenai pengembangan algoritma, kolaborasi penelitian, atau
        pertanyaan teknis terkait sistem ini, silakan menghubungi langsung melalui kontak personal
        pengembang aplikasi.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
    © 2026 Rafael Evaldo Setianto - PharmaPredict Developer
    </div>
    """, unsafe_allow_html=True)