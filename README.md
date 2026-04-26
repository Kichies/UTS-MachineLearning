# 🍊 Deteksi Klasifikasi: Jeruk vs Grapefruit (Anggur)

**Sumber Dataset:** [Oranges vs Grapefruit – Kaggle](https://www.kaggle.com/datasets/joshmcadams/oranges-vs-grapefruit)  
**File Data:** `citrus.csv`  
**Komparasi Algoritma:** Decision Tree | Naive Bayes | Support Vector Machine (SVM)

Proyek ini dikembangkan untuk memenuhi tugas Ujian Tengah Semester (UTS). Tujuannya adalah membangun model *Machine Learning* yang mampu mengklasifikasikan buah secara otomatis berdasarkan fitur fisik (diameter, berat, dan nilai RGB warna).

---

## 📁 Struktur Repositori

```text
📦 UTS-Klasifikasi-Jeruk
 ┣ 📜 citrus.csv                # Dataset mentah yang diunduh dari Kaggle
 ┣ 📜 klasifikasi_model.ipynb   # Source code utama berbasis Jupyter/Colab
 ┣ 📜 confusion_matrix.png      # Output visualisasi matriks evaluasi prediksi
 ┗ 📜 README.md                 # Dokumentasi alur proyek (File ini)
```

---

## 🚀 Persiapan & Cara Menjalankan

**1. Kebutuhan Library (Dependencies)**
Pastikan environment Python Anda sudah terinstal pustaka berikut:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

**2. Eksekusi Program**
Jika menggunakan Google Colab atau Jupyter Notebook, cukup jalankan `klasifikasi_model.ipynb` secara berurutan *(Run All)*. Pastikan file `citrus.csv` berada di direktori (folder) yang sama dengan notebook tersebut.

---

## 📊 Pipeline Pembuatan Model (7 Tahapan)

Proyek ini dieksekusi melalui 7 tahapan terstruktur mulai dari pemuatan data hingga pengambilan kesimpulan akhir.

### Tahap 1: Pemuatan Data (Data Loading)
**Deskripsi:** Membaca dataset ke dalam bentuk *DataFrame* agar mudah dimanipulasi.
```python
import pandas as pd
df = pd.read_csv('citrus.csv')
```
**Struktur Fitur yang Digunakan:**

| Nama Kolom | Tipe Data | Keterangan Fitur | Contoh Data |
| :--- | :--- | :--- | :--- |
| `name` | Kategorikal | Label kelas buah (Target) | `orange` / `grapefruit` |
| `diameter` | Numerik | Ukuran diameter buah (cm) | `2.96` |
| `weight` | Numerik | Berat massa buah (gram) | `86.76` |
| `red`, `green`, `blue` | Numerik | Intensitas warna RGB (0-255) | `172`, `85`, `2` |

### Tahap 2: Analisis Data Eksploratif (EDA)
**Deskripsi:** Melakukan inspeksi awal untuk melihat kualitas data.
```python
print(df.isnull().sum())         # Memeriksa data yang kosong (Missing Values)
print(df['name'].value_counts()) # Memeriksa keseimbangan kelas target
```
**Temuan Penting:**
* Tidak ada data kosong (0 missing values).
* Kelas target seimbang (5000 *orange*, 5000 *grapefruit*).
* Rentang skala antar fitur sangat timpang (Contoh: nilai `weight` puluhan, sedangkan nilai `blue` hanya satuan). Hal ini mengharuskan kita melakukan *Scaling*.

### Tahap 3: Pra-pemrosesan Data (Preprocessing)
**Deskripsi:** Menyiapkan data sebelum masuk ke fase pelatihan (training) model.
1. **Pemisahan Fitur & Target:** Memisahkan *input* `X` (fitur fisik) dan *output* `y` (label buah).
2. **Data Splitting:** Membagi rasio data menjadi **80% Latih (Training)** dan **20% Uji (Testing)**. Menggunakan argumen `stratify=y` agar proporsi jeruk dan anggur tetap seimbang di kedua bagian data.
3. **Standarisasi (Scaling):** Mengubah skala semua fitur agar memiliki rata-rata 0 dan standar deviasi 1 menggunakan `StandardScaler`. 
   > *Catatan: Decision Tree tidak terpengaruh rentang angka, tetapi algoritma berbasis probabilitas dan jarak seperti Naive Bayes dan SVM sangat membutuhkan data yang sudah di-scaling untuk mencapai akurasi optimal.*

### Tahap 4: Pelatihan Model (Model Training)
**Deskripsi:** Melatih tiga algoritma berbeda menggunakan data training (8.000 baris data).

* **Decision Tree:** Model dilatih menggunakan batasan percabangan `max_depth=6` untuk menghindari model sekadar "menghafal" data (*overfitting*). Model ini menggunakan data asli tanpa scaling.
* **Naive Bayes:** Menggunakan varian `GaussianNB` karena fitur fisik buah diasumsikan terdistribusi secara normal (membentuk kurva lonceng). Membutuhkan data latih yang di-scaling.
* **Support Vector Machine (SVM):** Menggunakan *kernel RBF* untuk menciptakan garis batas (*hyperplane*) melengkung yang fleksibel dalam memisahkan kedua kelas. Membutuhkan data latih yang di-scaling.

### Tahap 5: Evaluasi Performa
**Deskripsi:** Menguji model yang sudah "belajar" dengan menggunakan 2.000 data uji (Testing Data) yang tidak pernah dilihat sebelumnya.
Pengujian tidak hanya melihat akurasi, tetapi juga metrik fundamental lainnya:

| Metrik Evaluasi | Penjelasan Sederhana |
| :--- | :--- |
| **Accuracy** | Persentase total tebakan buah yang benar dari seluruh tebakan. |
| **Precision** | Saat model menebak itu "Jeruk", seberapa sering tebakan itu benar? |
| **Recall** | Dari seluruh "Jeruk" asli yang ada, berapa banyak yang berhasil dideteksi? |
| **F1-Score** | Nilai keseimbangan harmonis antara Precision dan Recall. |

### Tahap 6: Hasil Komparasi & Visualisasi
Hasil pengujian terhadap data *testing* menunjukkan performa yang sangat kompetitif dari ketiga algoritma:

| Algoritma Model | Nilai Akurasi | Karakteristik Performa |
| :--- | :---: | :--- |
| **Decision Tree** | **94.35%** | Sangat baik dan logikanya paling transparan. |
| **Support Vector Machine** | **92.90%** | Sangat responsif terhadap data yang telah di-scaling. |
| **Naive Bayes** | **92.00%** | Cukup akurat dan proses pelatihannya paling ringan. |

Visualisasi dari kesalahan prediksi masing-masing model dapat dilihat pada grafik *Confusion Matrix* yang tersimpan di file `confusion_matrix.png` pada repositori ini.

### Tahap 7: Kesimpulan Akhir & Rekomendasi
Berdasarkan eksperimen yang telah dilakukan, model **Decision Tree adalah algoritma terbaik** untuk dataset klasifikasi *Oranges vs Grapefruit* ini dengan akurasi tertinggi (94.35%). 

**Panduan Pemilihan Model untuk Implementasi Lanjutan:**

| Kriteria Kebutuhan | Rekomendasi Model | Alasan |
| :--- | :--- | :--- |
| **Interpretasi & Kejelasan** | **Decision Tree** | Alur keputusannya dapat divisualisasikan menjadi struktur pohon (aturan *If-Else*) yang mudah dimengerti orang awam. |
| **Efisiensi Waktu Komputasi** | **Naive Bayes** | Kalkulasi berbasis probabilitas yang sangat ringan, cocok untuk komputer dengan spesifikasi rendah atau data yang sangat masif. |
| **Pola Data Linear Tersembunyi** | **SVM** | Mampu menangani margin *error* klasifikasi yang kompleks di *dataset* lain dengan tuning *hyperparameter* (`C` dan `Gamma`) tingkat lanjut. |

---

## 📚 Referensi
* Scikit-learn Official Documentation: [https://scikit-learn.org](https://scikit-learn.org)
* Dataset *Oranges vs Grapefruit*: [Kaggle](https://www.kaggle.com/datasets/joshmcadams/oranges-vs-grapefruit)
* Materi Perkuliahan Machine Learning Pertemuan ke-8
