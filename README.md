# UTS-MachineLearning
# Klasifikasi Oranges vs Grapefruit

Proyek ini dibuat untuk memenuhi tugas Ujian Tengah Semester mata kuliah Machine Learning, dengan tujuan membangun model klasifikasi machine learning untuk membedakan antara buah jeruk (orange) dan anggur (grapefruit) menggunakan Python.

## Tahapan Pembuatan Model
1. **Load Dataset**: Membaca dataset `citrus.csv` menggunakan pandas.
2. **Data Preprocessing**: 
   - Memisahkan kolom target (`name`) dari fitur-fitur lainnya.
   - Membagi dataset menjadi data latih (80%) dan data uji (20%) menggunakan `train_test_split`.
3. **Pemodelan**: Menginisialisasi tiga model klasifikasi yang akan dibandingkan, yaitu Decision Tree, Naive Bayes, dan Support Vector Machine (SVM).
4. **Pelatihan (Training)**: Melatih ketiga model tersebut menggunakan data latih.
5. **Evaluasi (Testing)**: Melakukan prediksi pada data uji dan menghitung tingkat akurasi untuk masing-masing model.

## Hasil Komparasi Model
Berdasarkan hasil pengujian, berikut adalah tingkat akurasi dari ketiga model:
* **Decision Tree** : 94.35%
* **Support Vector Machine (SVM)** : 92.90%
* **Naive Bayes** : 92.00%

**Kesimpulan:** Dari ketiga model yang diuji, algoritma **Decision Tree** menghasilkan performa yang paling baik dengan tingkat akurasi tertinggi sebesar 94.35%.
