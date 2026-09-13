# Used Car Recommender

Aplikasi web machine learning untuk merekomendasikan mobil bekas Indonesia berdasarkan anggaran, tahun, kilometer, transmisi, lokasi, merek, dan fitur kendaraan.

## Metode

Aplikasi menggunakan content-based filtering dengan K-Nearest Neighbors. Batas harga, tahun, dan kilometer menyeleksi kandidat. Preferensi lain diproses menjadi jarak fitur berbobot, lalu KNN mengambil kendaraan terdekat. Hasil akhir didiversifikasi berdasarkan nama model.

Skor yang ditampilkan adalah skor kecocokan, bukan probabilitas pembelian atau jaminan kondisi kendaraan.

## Dataset

Sumber: [Used Car Listings in Indonesia](https://www.kaggle.com/datasets/indraputra21/used-car-listings-in-indonesia), berdasarkan listing Carsome Indonesia dan tersedia dengan lisensi CC0.

Dataset aplikasi disertakan di `data/used_car.csv`. Untuk mengambil versi terbaru secara manual:

```powershell
python scripts/download_data.py
```

Kolom kilometer pada sumber memakai titik sebagai pemisah ribuan. Pipeline mengubah nilai seperti `10.508` menjadi `10.508 km` sebelum digunakan model.

## Menjalankan secara lokal

Disarankan menggunakan Python 3.12.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

## Pengujian

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```

## Deployment

1. Push proyek ke repository GitHub.
2. Masuk ke Streamlit Community Cloud.
3. Pilih repository dan gunakan `app.py` sebagai entrypoint.
4. Pilih Python 3.12 lalu jalankan deployment.

Tidak diperlukan token Kaggle saat deployment karena dataset aplikasi sudah disertakan. Folder cache Kaggle tidak ikut disimpan ke Git.

## Struktur

```text
.
|-- app.py
|-- assets/
|   `-- styles.css
|-- data/
|   `-- used_car.csv
|-- scripts/
|   `-- download_data.py
|-- src/
|   |-- data.py
|   `-- recommender.py
|-- tests/
|-- requirements.txt
`-- README.md
```

