# 📈 Prediksi Harga Saham Farmasi Menggunakan LSTM dengan Mekanisme Attention

Aplikasi berbasis **Streamlit** (**PharmaPredict**) yang dikembangkan sebagai implementasi penelitian skripsi untuk melakukan prediksi harga saham perusahaan farmasi di Indonesia menggunakan algoritma **Long Short-Term Memory (LSTM)** yang dipadukan dengan **Attention Mechanism**. Aplikasi ini menyediakan fitur visualisasi tren historis data saham, evaluasi performa keakuratan model menggunakan metrik regresi, serta simulasi prediksi pergerakan harga saham melalui antarmuka web yang interaktif.

---

## 🌐 Demo Online

Aplikasi ini juga dapat diakses langsung melalui browser tanpa perlu instalasi, melalui tautan berikut:

🔗 https://skripsi-sistem-prediksi-harga-saham-farmasi.streamlit.app/

Aplikasi di-hosting menggunakan **Streamlit Community Cloud (tingkat gratis)**, sehingga aplikasi akan otomatis masuk ke mode **sleep (hibernasi)** apabila tidak ada pengunjung selama beberapa waktu. Hal ini bertujuan untuk menghemat resource server pada platform tersebut, bukan karena aplikasi mengalami error.

Apabila saat membuka tautan di atas muncul halaman bertuliskan **"Zzz... This app has gone to sleep"**, aplikasi dapat dibangunkan kembali dengan langkah berikut:

1. Klik tombol **"Yes, get this app back up!"** yang muncul pada halaman tersebut.
2. Tunggu beberapa saat (biasanya kurang dari satu menit) hingga aplikasi selesai dimuat ulang.
3. Aplikasi akan berjalan normal kembali setelah proses wake up selesai.

---

## ✨ Fitur

- 📊 Visualisasi interaktif tren pergerakan harga saham farmasi.
- 💊 Menampilkan informasi data historis emiten saham farmasi pilihan (KLBF, PEHA, SIDO, SOHO).
- 🤖 Estimasi/prediksi nilai nominal rupiah harga saham masa depan menggunakan kombinasi model LSTM dan Attention Layer (menggunakan optimalisasi *Huber Loss*).
- 📈 Visualisasi hasil evaluasi performa model yang meliputi:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Percentage Error (MAPE)
  - R-Squared (R² Score)
- 📁 Ekspor hasil nominal prediksi gabungan data training dan testing langsung ke format Microsoft Excel (.xlsx).

---

## 🛠️ Teknologi

- Python
- Streamlit
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib
- OpenPyXL

---

## 📂 Struktur Proyek

```text
Skripsi/
│
├── Dataset/                    # Folder berisi dataset historis saham (.csv) mentah
│   ├── Data Historis KLBF.csv
│   ├── Data Historis PEHA.csv
│   ├── Data Historis SIDO.csv
│   └── Data Historis SOHO.csv
│
├── saved_model/                # Folder penyimpanan model & bobot hasil pelatihan
│   ├── model/                  # Format file model Deep Learning (.keras)
│   │   ├── KLBF_lstm_attention_huber.keras
│   │   ├── PEHA_lstm_attention_huber.keras
│   │   ├── SIDO_lstm_attention_huber.keras
│   │   └── SOHO_lstm_attention_huber.keras
│   │
│   └── scaler/                 # Format file objek MinMaxScaler (.pkl)
│       ├── KLBF_scaler_huber.pkl
│       ├── PEHA_scaler_huber.pkl
│       ├── SIDO_scaler_huber.pkl
│       └── SOHO_scaler_huber.pkl
│
├── app.py                      # Program utama antarmuka dashboard web Streamlit
├── LSTM-Attention.ipynb        # Notebook preprocessing, eksperimen pelatihan, dan plot metrik
├── LSTM Biasa.ipynb
├── Data Historis KLBF (Data Kustom).csv
├── 535220086_BukuManualProgram.pdf      
│
├── requirements.txt            # Daftar dependensi library Python yang dibutuhkan
└── README.md                   # Dokumentasi panduan repositori
```

---

## 📥 Instalasi

### 1. Clone repository

```bash
git clone https://github.com/Xuordfess31/Skripsi.git
```

Masuk ke folder proyek.

```bash
cd Skripsi
```

---

### 2. Install seluruh dependency

```bash
pip install -r requirements.txt
```

---

## ▶️ Menjalankan Aplikasi

Jalankan aplikasi menggunakan perintah berikut.

```bash
streamlit run app.py
```

Secara otomatis browser akan membuka aplikasi pada alamat:

```text
http://localhost:8501
```

---

## 📊 Dataset

Repository ini menyertakan dataset historis harga saham perusahaan farmasi di Indonesia yang digunakan selama penelitian sebagai bentuk validasi dan testing model, meliputi emiten:

- PT Kalbe Farma Tbk (KLBF)
- PT Phapros Tbk (PEHA)
- PT Industri Jamu dan Farmasi Sido Muncul Tbk (SIDO)
- PT Soho Global Health Tbk (SOHO)

Seluruh dataset digunakan murni untuk keperluan analisis statistik, pengujian performa algoritma (Time Series Univariate berbasis Log Return), dan pengujian akademik dalam penyusunan skripsi.

---

## ⚖️ Sumber Data dan Kepatuhan terhadap *Terms of Use*

Data historis pergerakan harga saham pada penelitian ini diunduh secara manual melalui portal penyedia data pasar modal publik resmi. Seluruh proses pengumpulan berkas data dilakukan oleh penulis dengan memanfaatkan fitur unduh resmi (Download / Export to CSV) yang disediakan secara legal bagi publik oleh penyedia platform.

Repository ini **tidak menggunakan** teknik:

- Web Crawling
- Web Scraping
- Automated Data Extraction
- Bot
- Spider
- Crawler

Pendekatan ini menjunjung tinggi kepatuhan terhadap kebijakan Terms of Service / Terms of Use dari platform penyedia data pasar modal. Seluruh data yang tersedia di dalam repository ini mutlak digunakan untuk kepentingan riset sains data dan edukasi non-komersial.

## 📖 Manual Penggunaan

Panduan penggunaan aplikasi tersedia pada dokumen **535220086_BukuManualProgram** yang disertakan bersama proyek.

---

## 👨‍🎓 Penulis

**Rafael Evaldo Setianto**

Program Studi Teknik Informatika  
Fakultas Teknologi Informasi  
Universitas Tarumanagara

---

## 📜 Lisensi

Repository ini dibuat sebagai bagian dari luaran penelitian akademik (Skripsi) dan dipublikasikan secara terbuka untuk tujuan pembelajaran serta riset implementasi kecerdasan buatan (Deep Learning) di sektor finansial.

Apabila Anda mengutip, memodifikasi, atau menggunakan sebagian baris kode dari repositori ini, mohon cantumkan atribusi referensi kepada penulis asli. Segala bentuk prediksi nilai saham yang dihasilkan oleh sistem ini merupakan simulasi matematis murni berbasis probabilitas masa lalu, dan bukan merupakan rekomendasi mutlak maupun saran profesional untuk keputusan investasi riil di pasar modal.
