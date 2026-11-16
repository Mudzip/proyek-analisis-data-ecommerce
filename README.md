# Proyek Analisis Data: E-Commerce Public Dataset ✨

Proyek akhir kelas **Belajar Analisis Data dengan Python** (Dicoding) oleh **Adam Himawan**.
Analisis menggunakan E-Commerce Public Dataset (marketplace Olist, Brasil) untuk periode **Januari 2017 – Agustus 2018**.

## Pertanyaan Bisnis
1. Bagaimana tren jumlah pesanan dan revenue bulanan selama Jan 2017 – Ags 2018, bulan apa yang tertinggi, dan berapa pertumbuhan revenue Jan–Ags 2018 dibanding Jan–Ags 2017?
2. Kategori produk apa saja yang termasuk 10 besar penyumbang revenue, berapa kontribusi gabungannya, dan kategori mana yang paling kecil?
3. Seberapa besar selisih rata-rata review score antara pesanan terlambat dan tepat waktu, dan bagaimana skornya berubah menurut lamanya keterlambatan?

Analisis lanjutan: **RFM Analysis**, **Clustering** (manual grouping & binning), dan **Geospatial Analysis** (peta per state).

## Struktur Folder
```
submission
├── dashboard
│   ├── dashboard.py            # aplikasi Streamlit
│   ├── main_data.csv           # data bersih hasil notebook
│   └── brazil_states.geojson   # batas wilayah state Brasil untuk peta
├── data                        # 9 file CSV dataset asli + brazil_states.geojson (hanya di GitHub)
├── notebook.ipynb              # seluruh proses analisis
├── README.md
├── requirements.txt
└── url.txt
```

## Setup Environment - Anaconda
```
conda create --name main-ds python=3.12
conda activate main-ds
pip install -r requirements.txt
```

## Setup Environment - Shell/Terminal
```
cd submission
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

## Menjalankan Notebook
Jalankan `notebook.ipynb` dari dalam folder `submission`. Notebook juga menghasilkan ulang `dashboard/main_data.csv`.

> **Catatan dataset:** folder `data/` (±125 MB) tidak disertakan di berkas zip submission karena melebihi batas upload 25 MB. Jika folder `data/` tidak ada, notebook otomatis membaca dataset yang sama dari repositori GitHub: https://github.com/Mudzip/proyek-analisis-data-ecommerce/tree/main/data (butuh koneksi internet). Dashboard tidak memerlukan folder `data/` karena memakai `dashboard/main_data.csv`.

## Run Streamlit App
```
streamlit run dashboard/dashboard.py
```

## Sumber Data
- E-Commerce Public Dataset (disediakan Dicoding, berasal dari Olist Brazilian E-Commerce Public Dataset).
- Batas wilayah state Brasil: `brazil-states.geojson` dari repositori [codeforgermany/click_that_hood](https://github.com/codeforgermany/click_that_hood), disederhanakan (pembulatan koordinat) agar ringan.
