# Gameplay Screenshot Classification using MobileNetV3Large and ResNet-50

Project ini merupakan implementasi computer vision untuk mengklasifikasikan screenshot gameplay ke dalam beberapa kategori game menggunakan pendekatan transfer learning. Dua arsitektur deep learning yang digunakan adalah **MobileNetV3Large** dan **ResNet-50**, kemudian performanya dibandingkan berdasarkan metrik evaluasi seperti accuracy, precision, recall, F1-score, loss, dan confusion matrix.

Project ini dibuat sebagai bagian dari tugas mata kuliah **Visi Komputer**.

## Topik Penelitian

**Performance Comparison of MobileNetV3Large and ResNet-50 for Gameplay Screenshot Classification**

Penelitian ini berfokus pada klasifikasi gambar screenshot gameplay dari beberapa game berbeda. Model dilatih untuk mengenali pola visual dari setiap game, seperti warna, karakter, lingkungan, tampilan HUD, dan gaya visual gameplay.

## Dataset

Dataset yang digunakan adalah dataset screenshot gameplay yang terdiri dari 10 kelas game:

* Among Us
* Apex Legends
* Fortnite
* Forza Horizon
* Free Fire
* Genshin Impact
* God of War
* Minecraft
* Roblox
* Terraria

Total dataset berisi **10.000 gambar**, dengan distribusi seimbang yaitu **1.000 gambar per kelas**.

Dataset dibagi menjadi:

* 70% training data
* 15% validation data
* 15% testing data

Seluruh gambar di-resize menjadi ukuran **224 × 224 piksel** sebelum digunakan untuk training.

## Model yang Digunakan

### 1. MobileNetV3Large

MobileNetV3Large digunakan sebagai model transfer learning yang ringan dan efisien. Model ini cocok untuk skenario dengan keterbatasan sumber daya karena ukuran modelnya relatif kecil.

Konfigurasi utama:

* Pretrained weights: ImageNet
* Input size: 224 × 224
* Batch size: 16
* Epoch: 10
* Optimizer: Adam
* Learning rate: 0.0001
* Loss function: Sparse Categorical Crossentropy
* Dropout: 0.30

### 2. ResNet-50

ResNet-50 digunakan sebagai model pembanding dengan arsitektur yang lebih dalam. Model ini memiliki kemampuan representasi fitur yang lebih kuat, tetapi ukuran modelnya lebih besar.

Konfigurasi utama:

* Pretrained weights: ImageNet
* Input size: 224 × 224
* Batch size: 8
* Epoch: 10
* Optimizer: Adam
* Learning rate: 0.0001
* Loss function: Sparse Categorical Crossentropy
* Dropout: 0.30

## Preprocessing dan Augmentation

Tahapan preprocessing yang dilakukan:

* Membaca gambar dari folder dataset
* Resize gambar menjadi 224 × 224 piksel
* Split dataset secara stratified menjadi train, validation, dan test
* Menggunakan preprocessing sesuai model
* Menggunakan data augmentation pada data training

Augmentation yang digunakan:

* Random horizontal flip
* Random rotation
* Random zoom
* Random contrast

Augmentasi dibuat ringan agar tidak merusak struktur visual screenshot gameplay, terutama elemen seperti HUD, karakter, dan lingkungan game.

## Struktur Folder

```text
gameplay-classification/
│
├── train.py
├── data/
│   └── Dataset/
│       ├── Among Us/
│       ├── Apex Legends/
│       ├── Fortnite/
│       ├── Forza Horizon/
│       ├── Free Fire/
│       ├── Genshin Impact/
│       ├── God of War/
│       ├── Minecraft/
│       ├── Roblox/
│       └── Terraria/
│
├── models/
│   ├── MobileNetV3Large_final.keras
│   ├── MobileNetV3Large_best.keras
│   ├── ResNet50_final.keras
│   └── ResNet50_best.keras
│
├── results/
│   ├── MobileNetV3Large_accuracy.png
│   ├── MobileNetV3Large_loss.png
│   ├── MobileNetV3Large_confusion_matrix.png
│   ├── MobileNetV3Large_classification_report.csv
│   ├── ResNet50_accuracy.png
│   ├── ResNet50_loss.png
│   ├── ResNet50_confusion_matrix.png
│   ├── ResNet50_classification_report.csv
│   └── final_comparison.csv
│
└── README.md
```

## Instalasi

Project ini dijalankan menggunakan Python dan TensorFlow.

Versi environment yang digunakan:

* Python 3.10
* TensorFlow 2.10
* CUDA 11.2
* cuDNN 8.1
* NVIDIA RTX 3050

Install dependency:

```bash
pip install tensorflow==2.10.0 numpy==1.23.5 pandas==1.5.3 matplotlib==3.7.3 scikit-learn==1.3.2 opencv-python==4.8.1.78 kaggle
```

Cek apakah GPU sudah terbaca:

```bash
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

Jika berhasil, output akan menampilkan GPU seperti:

```text
2.10.0
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## Cara Menjalankan Program

Masuk ke folder project:

```bash
cd C:\Users\ASUS\gameplay-classification
```

Aktifkan virtual environment:

```bash
venv\Scripts\activate
```

### Test cepat dengan sebagian dataset

```bash
python train.py --dataset data\Dataset --epochs 2 --batch 8 --limit-per-class 100 --only mobilenetv3
```

### Training MobileNetV3Large

```bash
python train.py --dataset data\Dataset --epochs 10 --batch 16 --only mobilenetv3
```

### Training ResNet-50

```bash
python train.py --dataset data\Dataset --epochs 10 --batch 8 --only resnet50
```

### Prediksi satu gambar

Prediksi menggunakan MobileNetV3Large:

```bash
python train.py --predict --model-path models\MobileNetV3Large_final.keras --image-path path_gambar.png
```

Prediksi menggunakan ResNet-50:

```bash
python train.py --predict --model-path models\ResNet50_final.keras --image-path path_gambar.png
```

## Hasil Eksperimen

Hasil evaluasi menunjukkan bahwa ResNet-50 memiliki performa lebih tinggi dibandingkan MobileNetV3Large, tetapi MobileNetV3Large tetap memiliki keunggulan dari sisi ukuran model yang lebih kecil.

| Model            | Test Accuracy | Macro F1-score | Test Loss | Model Size |
| ---------------- | ------------: | -------------: | --------: | ---------: |
| MobileNetV3Large |        95.27% |         0.9526 |    0.1777 |     ±12 MB |
| ResNet-50        |        98.20% |         0.9820 |    0.0633 |     ±91 MB |

Berdasarkan hasil tersebut, **ResNet-50** memberikan performa klasifikasi terbaik. Namun, **MobileNetV3Large** tetap menjadi pilihan yang menarik untuk skenario deployment yang membutuhkan model ringan.

## Analisis Singkat

MobileNetV3Large dan ResNet-50 sama-sama menunjukkan performa yang stabil selama proses training. Grafik accuracy dan loss menunjukkan bahwa training accuracy dan validation accuracy meningkat secara konsisten, sedangkan training loss dan validation loss menurun. Hal ini menunjukkan bahwa tidak terdapat indikasi overfitting yang signifikan pada kedua model.

Dari confusion matrix, beberapa kelas seperti **God of War**, **Minecraft**, dan **Terraria** lebih mudah diklasifikasikan karena memiliki karakteristik visual yang sangat khas. Sementara itu, kelas seperti **Forza Horizon**, **Apex Legends**, dan **Fortnite** lebih menantang karena memiliki kemiripan visual dengan beberapa game lain, terutama dari sisi lingkungan, warna, dan gaya visual.

## Output Program

Program akan menghasilkan beberapa file penting:

```text
models/
```

Berisi model hasil training dalam format `.keras`.

```text
results/
```

Berisi grafik, confusion matrix, classification report, dan tabel perbandingan hasil evaluasi.

File penting untuk analisis:

* `MobileNetV3Large_accuracy.png`
* `MobileNetV3Large_loss.png`
* `MobileNetV3Large_confusion_matrix.png`
* `ResNet50_accuracy.png`
* `ResNet50_loss.png`
* `ResNet50_confusion_matrix.png`
* `final_comparison.csv`

## Paper

Hasil eksperimen project ini digunakan untuk menyusun paper dengan judul:

**Performance Comparison of MobileNetV3Large and ResNet-50 for Gameplay Screenshot Classification**

Paper ini telah disubmit ke:

**2026 10th International Conference on Information Technology, Information Systems and Electrical Engineering (ICITISEE)**
Track: **Information Technology**

## Author

**Yudhistira Maulana Samaratungga**
Informatics Study Program
Telkom University
Bandung, Indonesia

## License

Project ini dibuat untuk kebutuhan akademik dan pembelajaran pada mata kuliah Visi Komputer.
