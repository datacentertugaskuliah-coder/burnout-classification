# Streamlit Burnout - Batch Excel

## Isi paket
- `streamlit_app.py`: aplikasi prediksi batch Excel dan prediksi satu responden.
- `gaussian_nb_burnout_bundle.joblib`: model Gaussian Naive Bayes yang Anda unggah.
- `requirements.txt`: dependensi dengan scikit-learn dipatok ke 1.6.1 agar sesuai dengan versi saat model disimpan.

## Instalasi lokal
Letakkan seluruh file dalam satu folder, lalu jalankan:

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy Streamlit Community Cloud
1. Ekstrak ZIP.
2. Ganti `streamlit_app.py` dan `requirements.txt` di repository GitHub dengan file dari paket ini.
3. Tambahkan `gaussian_nb_burnout_bundle.joblib` ke root repository (sefolder dengan streamlit_app.py).
4. Commit dan Push melalui GitHub Desktop.
5. Pada Streamlit Cloud, buka Manage app lalu pilih Reboot app bila deployment tidak otomatis dimulai.

## Dataset
Model bundle memuat sembilan fitur:
- Skor_Beban_Kerja
- Skor_Tekanan_Target
- Durasi_Kerja_Harian_Jam
- Masa_Kerja_Bulan
- Skor_Kualitas_Tidur
- Waktu_Pemulihan_Jam
- Skor_Interaksi_Pelanggan
- Skor_Dukungan_Rekan
- Skor_Lingkungan_Kerja

Kolom ID_Responden dan Tingkat_Burnout dapat ikut disertakan. Pola_Kerja dapat ikut disertakan sebagai kolom informasi, tetapi tidak digunakan oleh model baseline sembilan fitur.

## Catatan metodologis
Status dalam model menyatakan dataset pelatihan bersifat sintetis, bukan hasil pengumpulan lapangan. Prediksi dari dataset sintetis hanya mendemonstrasikan proses teknis. Kesesuaian prediksi dengan label pada data yang sama tidak boleh disebut akurasi generalisasi. Untuk evaluasi, sediakan baris test/holdout independen yang ditandai pada kolom `Data_Split` dengan nilai `test` atau `holdout`.
