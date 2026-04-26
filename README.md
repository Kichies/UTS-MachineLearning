# 🍊 Klasifikasi Machine Learning: Oranges vs Grapefruit

Proyek ini merupakan implementasi Ujian Tengah Semester untuk membangun model klasifikasi *machine learning*. Tujuannya adalah membedakan buah jeruk (orange) dan anggur (grapefruit) berdasarkan fitur fisiknya menggunakan tiga algoritma berbeda.

**Dataset:** [Oranges vs Grapefruit – Kaggle](https://www.kaggle.com/datasets/joshmcadams/oranges-vs-grapefruit)  
**Model yang Diuji:** Decision Tree | Naive Bayes | Support Vector Machine (SVM)

---

## 📂 Struktur Repositori

* `citrus.csv` : Dataset mentah yang digunakan.
* `klasifikasi_uts.ipynb` : *Source code* utama (Python/Jupyter Notebook).
* `confusion_matrix.png` : Gambar visualisasi hasil prediksi model.
* `README.md` : Dokumentasi alur kerja proyek.

---

## 📋 Tahapan Pembuatan Model

Implementasi dalam proyek ini dibagi menjadi beberapa tahap komprehensif:

### Tahap 1: Pengumpulan & Pemahaman Data
Membaca dataset `citrus.csv` yang berisi sekitar 10.000 baris data. Dataset ini memiliki 5 fitur prediktor (`diameter`, `weight`, `red`, `green`, `blue`) dan 1 kolom target (`name`).

### Tahap 2: Exploratory Data Analysis (EDA)
Melakukan inspeksi kelayakan data. Hasil pemeriksaan menunjukkan:
- Kelas target (`orange` dan `grapefruit`) terdistribusi dengan sangat seimbang.
- Tidak ditemukan *missing values* (data kosong).
- Terdapat perbedaan rentang skala yang signifikan antara fitur warna (contoh: `red` di atas 150) dengan fitur ukuran (`blue` di bawah 10).

### Tahap 3: Preprocessing Data
1. **Pemisahan:** Memisahkan kolom target (`y`) dari kolom fitur (`X`).
2. **Splitting:** Membagi data menjadi 80% Data Latih (Training) dan 20% Data Uji (Testing) menggunakan parameter `stratify` agar proporsi kelas tetap terjaga.
3. **Scaling:** Menggunakan `StandardScaler` untuk menyamakan skala fitur data. Langkah ini krusial untuk algoritma berbasis jarak dan probabilitas (SVM dan Naive Bayes).

### Tahap 4: Pelatihan Model (Training)
Tiga algoritma dilatih menggunakan data latih:
* **Decision Tree Classifier:** Dilatih menggunakan data asli (tanpa scaling) dengan pembatasan kedalaman (`max_depth=6`) untuk mencegah *overfitting*.
* **Gaussian Naive Bayes:** Dilatih menggunakan data yang sudah di-scaling untuk menghitung probabilitas fitur berbasis distribusi normal.
* **Support Vector Machine (SVC):** Dilatih menggunakan data yang sudah di-scaling menggunakan *kernel RBF* untuk mencari garis batas (*hyperplane*) optimal.

### Tahap 5 & 6: Evaluasi dan Visualisasi Hasil
Model diuji menggunakan Data Uji (20% data yang belum pernah dilihat model). Evaluasi difokuskan pada tingkat akurasi dan divisualisasikan menggunakan *Confusion Matrix*.

**Hasil Komparasi Akurasi:**
* **Naive Bayes** : 92.20%
* **Support Vector Machine (SVM)** : 92.50%
* **Decision Tree** : 94.35%

*(Visualisasi Confusion Matrix dapat dilihat pada file `confusion_matrix.png` di repositori ini).*

### Tahap 7: Kesimpulan
Berdasarkan hasil pengujian di atas, algoritma **Decision Tree** menghasilkan performa paling optimal untuk dataset ini dengan akurasi tertinggi mencapai **94.35%**. Model ini mampu mengklasifikasikan jeruk dan anggur dengan margin *error* yang sangat minim dibandingkan dua model lainnya.
