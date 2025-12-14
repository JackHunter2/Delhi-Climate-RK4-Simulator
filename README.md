# 🌡️ Delhi Climate RK4 Simulator

Aplikasi web interaktif untuk simulasi dinamika suhu harian di Kota Delhi menggunakan metode Runge-Kutta Orde 4 (RK4).

## 📋 Deskripsi

Aplikasi ini mensimulasikan perubahan suhu harian menggunakan model persamaan diferensial biasa dengan metode numerik RK4. Aplikasi dilengkapi dengan fitur optimasi parameter otomatis, visualisasi interaktif, dan analisis statistik yang komprehensif.

## ✨ Fitur Utama

- **Simulasi Interaktif**: Mode manual dan optimasi otomatis parameter
- **Visualisasi Dinamis**: Grafik interaktif menggunakan Plotly
- **Metrik Evaluasi**: RMSE, MAE, MAPE, dan R² Score
- **Analisis Residual**: Plot residual dan histogram untuk evaluasi model
- **Optimasi Parameter**: Grid search untuk mencari parameter optimal
- **Download Hasil**: Ekspor hasil simulasi dalam format CSV dan summary text
- **UI Modern**: Antarmuka yang user-friendly dan informatif

## 🚀 Instalasi

1. Clone repository ini atau download file-file yang diperlukan
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📖 Cara Menggunakan

1. Jalankan aplikasi Streamlit:
```bash
streamlit run app.py
```

2. Aplikasi akan terbuka di browser (biasanya di `http://localhost:8501`)

3. Pilih mode simulasi:
   - **Manual**: Atur parameter k dan T_eq secara manual menggunakan slider
   - **Optimasi Otomatis**: Klik tombol "Jalankan Optimasi" untuk mencari parameter terbaik

4. Lihat hasil simulasi, metrik evaluasi, dan visualisasi interaktif

5. Download hasil jika diperlukan

## 📊 Model Matematika

### Persamaan Diferensial

Model yang digunakan adalah model relaksasi suhu orde satu:

$$\frac{dT}{dt} = k (T_{eq} - T)$$

dengan:
- $T(t)$: Suhu harian pada waktu $t$
- $k$: Konstanta laju perubahan suhu
- $T_{eq}$: Suhu keseimbangan lingkungan

### Metode Runge-Kutta Orde 4

RK4 menggunakan empat estimasi kemiringan pada setiap langkah waktu untuk mendapatkan solusi numerik yang akurat.

## 📁 Struktur File

```
Delhi-Climate-RK4-Simulator/
├── app.py                      # Aplikasi Streamlit utama
├── requirements.txt            # Dependencies Python
├── DailyDelhiClimateTest.csv   # Dataset suhu harian Delhi
├── TA10_RK4.ipynb              # Notebook analisis (opsional)
└── README.md                   # Dokumentasi
```

## 🛠️ Teknologi yang Digunakan

- **Streamlit**: Framework untuk aplikasi web
- **Pandas**: Pengolahan data
- **NumPy**: Komputasi numerik
- **Plotly**: Visualisasi interaktif

## 📝 Metrik Evaluasi

Aplikasi menghitung beberapa metrik untuk mengevaluasi kualitas model:

- **RMSE** (Root Mean Square Error): Mengukur kesalahan rata-rata
- **MAE** (Mean Absolute Error): Rata-rata kesalahan absolut
- **MAPE** (Mean Absolute Percentage Error): Kesalahan dalam persentase
- **R² Score**: Koefisien determinasi (seberapa baik model menjelaskan variasi data)

## 👨‍💻 Pengembangan

Aplikasi ini dikembangkan untuk keperluan akademik dalam mata kuliah Pemodelan dan Simulasi.

## 📄 Lisensi

Proyek ini dibuat untuk keperluan pendidikan dan penelitian.

---

**Dibuat dengan ❤️ menggunakan Streamlit dan Plotly**
