# Face Attendance Verification Prototype

Prototype sistem absensi berbasis wajah yang menggabungkan beberapa **pretrained model** dalam satu **multi-stage biometric verification pipeline**.

Sistem melakukan dua pemeriksaan utama:

1. **1:1 Face Verification** — memastikan wajah di kamera cocok dengan foto referensi pegawai.
2. **Face Anti-Spoofing / Liveness Detection** — memastikan wajah berasal dari manusia nyata, bukan foto atau video yang ditampilkan dari perangkat lain.

Hasil dari kedua tahap kemudian digabungkan menggunakan **decision-level fusion** untuk menentukan keputusan akhir.

---

## 1. Arsitektur Sistem

```text
REFERENCE PHOTO
      |
      v
OpenVINO Face Detection
      |
      v
5-Point Facial Landmarks
      |
      v
Face Alignment
      |
      v
Face Re-Identification
      |
      v
Reference 256-D Embedding
      |
      |                    WEBCAM
      |                       |
      |                       v
      |              OpenVINO Face Detection
      |                       |
      |                       +----------------------+
      |                       |                      |
      |                       v                      v
      |              5-Point Landmarks         Face Crop
      |                       |                      |
      |                       v                      v
      |                Face Alignment       CVPR2024 Swin-V2
      |                       |                      |
      |                       v                      v
      |              Current 256-D Embedding    LIVE / FAKE
      |                       |
      +-----------------------+
                  |
                  v
        Cosine Similarity / Distance
                  |
                  v
             MATCH / NOT MATCH
                  |
                  +----------------------+
                                         |
                                         v
                               FINAL DECISION
```

---

## 2. Model yang Digunakan

### Stage 1 — 1:1 Face Verification

Tahap pertama menggunakan tiga pretrained model dari **Intel Open Model Zoo** dan dijalankan menggunakan **OpenVINO Runtime**.

#### 2.1 `face-detection-retail-0004`

Digunakan untuk mendeteksi wajah pada:
- foto referensi;
- frame webcam;
- memastikan hanya satu wajah yang sedang diproses.

Output utamanya adalah bounding box wajah.

#### 2.2 `landmarks-regression-retail-0009`

Digunakan untuk mendeteksi **5 facial landmarks** pada wajah.

Landmark digunakan untuk membantu proses **face alignment**, sehingga wajah referensi dan wajah dari webcam berada pada orientasi yang lebih konsisten sebelum menghasilkan embedding.

#### 2.3 `face-reidentification-retail-0095`

Digunakan untuk menghasilkan representasi numerik wajah berupa:

```text
256-dimensional face embedding
```

Embedding dari foto referensi dibandingkan dengan embedding wajah dari webcam menggunakan **cosine similarity / cosine distance**.

Pada prototype saat ini:

```text
distance = (1 - cosine_similarity) * 0.5
```

Keputusan awal:

```text
distance <= 0.30  -> MATCH
distance > 0.30   -> NOT MATCH
```

Threshold tersebut masih dapat dikalibrasi kembali menggunakan data validasi.

---

## 3. Stage 2 — Liveness / Face Anti-Spoofing

Tahap kedua menggunakan model dari:

```text
CVPR 2024 Face Anti-Spoofing Challenge
Joint Physical-Digital Facial Attack Detection
```

Model yang digunakan:

```text
face_swin_v2_base.pth
```

Arsitektur:

```text
Swin Transformer V2 Base
```

Input:

```text
face crop
    |
resize 224 x 224
    |
ImageNet normalization
    |
Swin-V2
```

Output:

```text
Class 0 -> LIVE
Class 1 -> SPOOF
```

Dalam program, probabilitas hasil beberapa inference disimpan lalu distabilkan menggunakan **temporal aggregation / median smoothing**.

---

## 4. Final Decision

Kedua hasil digabungkan menjadi keputusan akhir:

| Face Verification | Liveness | Final Result |
|---|---|---|
| MATCH | LIVE | VERIFIED |
| MATCH | FAKE | SPOOF ATTACK |
| NOT MATCH | LIVE | WRONG PERSON |
| NOT MATCH | FAKE | WRONG PERSON + SPOOF |

Contoh penting:

```text
Wajah Falah ditampilkan melalui layar HP

Stage 1:
MATCH

Stage 2:
FAKE

Final:
SPOOF ATTACK
```

Dengan demikian, foto atau video dari orang yang benar tetap dapat dikenali oleh face verification, tetapi ditolak oleh liveness detection.

---

## 5. Strategi Model

Prototype ini **tidak melatih neural network dari awal** dan belum melakukan fine-tuning terhadap bobot model.

Yang dilakukan adalah mengintegrasikan beberapa pretrained model menjadi satu sistem.

Komponen pada level sistem:

- multi-model pipeline;
- 1:1 face verification;
- face alignment;
- cosine-distance matching;
- temporal aggregation;
- asynchronous liveness inference;
- decision-level fusion;
- retry mechanism;
- decision-time measurement;
- maximum decision timeout.

Pendekatan saat ini dapat disebut sebagai:

```text
Multi-stage pretrained-model pipeline
with decision-level fusion
```

---

## 6. Kenapa Liveness Tidak Membuat Kamera Patah-Patah?

Model Swin-V2 jauh lebih berat dibanding model Stage 1.

Jika dijalankan secara synchronous:

```text
camera frame
    |
face detection
    |
face verification
    |
Swin-V2 inference
    |
tunggu sekitar ratusan ms
    |
frame berikutnya
```

Preview webcam menjadi patah-patah karena main thread harus menunggu inference liveness selesai.

Pada versi sekarang, liveness dijalankan secara **asynchronous**:

```text
MAIN THREAD
camera
  |
face detection
  |
1:1 verification
  |
update UI
  |
next frame

          face crop
              |
              v
      BACKGROUND THREAD
              |
        Swin-V2 inference
              |
         LIVE / FAKE
              |
              v
       result dikembalikan
```

Queue liveness hanya mempertahankan crop terbaru sehingga frame lama tidak terus menumpuk.

---

## 7. Temporal Smoothing

### Face Verification

Default:

```text
FR_SMOOTHING_WINDOW = 10
FR_MIN_SAMPLES = 5
```

Median distance digunakan sebagai dasar keputusan.

### Liveness

Default:

```text
FAS_SMOOTHING_WINDOW = 3
FAS_MIN_SAMPLES = 2
```

Median probability LIVE digunakan untuk menstabilkan hasil.

Tujuannya adalah mengurangi kemungkinan satu frame buruk langsung menyebabkan keputusan salah.

---

## 8. Decision Timer

Versi utama saat ini:

```text
attendance_full_commercial_candidate_async_timed.py
```

memiliki pengukuran waktu pengambilan keputusan.

Timer **tidak dimulai ketika tombol R ditekan**.

Timer mulai ketika:

```text
tepat satu wajah berhasil terdeteksi
```

Contoh:

```text
R ditekan
    |
menunggu wajah
    |
1 wajah terdeteksi
    |
timer = 0.00 s
    |
FR + liveness berjalan
    |
FINAL: VERIFIED (1.45 s)
```

Default maximum decision time:

```text
MAX_DECISION_SECONDS = 4.0
```

Jika sistem belum memperoleh bukti yang cukup setelah batas waktu tersebut:

```text
FINAL: RETRY
```

---

# Installation

## 9. Masuk ke Folder Project

```powershell
cd "E:\Performa\02 - Face Recognition\00-real-time-exp\mark-3"
```

## 10. Membuat Virtual Environment

```powershell
python -m venv .venv
```

Aktifkan:

```powershell
.\.venv\Scripts\Activate.ps1
```

Atau jalankan Python virtual environment secara langsung:

```powershell
.\.venv\Scripts\python.exe
```

## 11. Install Dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Dependency utama:

```text
numpy
opencv-python
openvino
torch
torchvision
timm
thop
```

---

# Model Setup

## 12. Struktur Folder Model

Program mengharapkan struktur berikut:

```text
models/
└── commercial_test/
    ├── openvino_fr/
    │   ├── face-detection-retail-0004/
    │   │   └── FP16/
    │   │       ├── face-detection-retail-0004.xml
    │   │       └── face-detection-retail-0004.bin
    │   ├── landmarks-regression-retail-0009/
    │   │   └── FP16/
    │   │       ├── landmarks-regression-retail-0009.xml
    │   │       └── landmarks-regression-retail-0009.bin
    │   └── face-reidentification-retail-0095/
    │       └── FP16/
    │           ├── face-reidentification-retail-0095.xml
    │           └── face-reidentification-retail-0095.bin
    └── cvpr2024_fas/
        ├── nets/
        │   └── swin_transformer_v2.py
        ├── weights/
        │   └── face_swin_v2_base.pth
        └── ...
```

## 13. Open Model Zoo Models

Stage 1 membutuhkan:

```text
face-detection-retail-0004
landmarks-regression-retail-0009
face-reidentification-retail-0095
```

Gunakan versi FP16 dan letakkan file `.xml` serta `.bin` mengikuti struktur folder di atas.

Program mencari model dari:

```text
models/commercial_test/openvino_fr/
```

## 14. CVPR2024 Face Anti-Spoofing

Clone repository ke folder model:

```powershell
git clone https://github.com/Xianhua-He/cvpr2024-face-anti-spoofing-challenge.git `
".\models\commercial_test\cvpr2024_fas"
```

Checkpoint yang digunakan:

```text
face_swin_v2_base.pth
```

Letakkan pada:

```text
models\commercial_test\cvpr2024_fas\weights\face_swin_v2_base.pth
```

Jika menggunakan `gdown`:

```powershell
.\.venv\Scripts\python.exe -m pip install gdown
```

Google Drive file ID checkpoint yang dipakai pada project ini:

```text
1E4UD8UK_KzjhpAvR6hYInlteOEaxDZbZ
```

Contoh download:

```powershell
.\.venv\Scripts\gdown.exe `
1E4UD8UK_KzjhpAvR6hYInlteOEaxDZbZ `
-O ".\models\commercial_test\cvpr2024_fas\weights\face_swin_v2_base.pth"
```

---

# Running

## 15. Siapkan Reference Image

Contoh:

```text
FALAH.jpg
```

Reference image idealnya:

- hanya berisi satu wajah;
- wajah cukup jelas;
- tidak blur;
- pencahayaan cukup;
- posisi wajah tidak terlalu ekstrem.

## 16. Menjalankan Program Utama

```powershell
.\.venv\Scripts\python.exe `
.\attendance_full_commercial_candidate_async_timed.py `
--reference ".\FALAH.jpg"
```

Versi satu baris:

```powershell
.\.venv\Scripts\python.exe .\attendance_full_commercial_candidate_async_timed.py --reference ".\FALAH.jpg"
```

## 17. Camera Index Lain

```powershell
.\.venv\Scripts\python.exe `
.\attendance_full_commercial_candidate_async_timed.py `
--reference ".\FALAH.jpg" `
--camera 1
```

---

# Controls

## 18. Keyboard

```text
R   -> Retry / reset verification session
F   -> Toggle fullscreen
Q   -> Exit
ESC -> Exit
```

## 19. Retry Button

UI juga mempunyai tombol `RETRY`.

Retry membersihkan:

- FR similarity history;
- FR distance history;
- liveness history;
- final decision;
- session timer;
- pending liveness sample yang sudah tidak relevan.

---

# Runtime Flow

## 20. Saat Program Dibuka

Program melakukan:

```text
1. Load OpenVINO Runtime
2. Load face detector
3. Load landmark model
4. Load face re-identification model
5. Read reference image
6. Detect reference face
7. Align reference face
8. Generate reference embedding
9. Load Swin-V2 anti-spoofing model
10. Start background liveness worker
11. Open webcam
```

Reference embedding dibuat satu kali pada awal program.

## 21. Saat Webcam Berjalan

```text
frame webcam
    |
face detection
    |
+--------------------------+
| 0 face                   |
| -> NO FACE               |
+--------------------------+
| >1 face                  |
| -> MULTIPLE FACES        |
+--------------------------+
| exactly 1 face           |
| -> lanjut                |
+--------------------------+
```

Jika tepat satu wajah:

```text
Stage 1
-------
landmarks
    |
alignment
    |
256-D embedding
    |
cosine distance
    |
median smoothing
    |
MATCH / NOT MATCH
```

Secara paralel:

```text
Stage 2
-------
face crop
    |
background worker
    |
Swin-V2
    |
LIVE probability
    |
median smoothing
    |
LIVE / FAKE
```

Kemudian:

```text
Stage 1 + Stage 2
        |
decision-level fusion
        |
final result
```

---

# Main Files

## 22. `attendance_full_commercial_candidate_async_timed.py`

Program utama.

Fitur:

- OpenVINO face detection;
- 5-point landmarks;
- 1:1 face verification;
- CVPR2024 Swin-V2 liveness;
- asynchronous liveness worker;
- temporal smoothing;
- decision-level fusion;
- retry/reset;
- fullscreen;
- decision timer;
- timeout;
- realtime UI.

## 23. `test_commercial_1to1_openvino.py`

Digunakan untuk menguji Stage 1 secara terpisah.

Contoh:

```powershell
.\.venv\Scripts\python.exe `
.\test_commercial_1to1_openvino.py `
--reference ".\FALAH.jpg"
```

---

# Important Parameters

## 24. Face Detection Threshold

```python
FACE_DET_THRESHOLD = 0.60
```

## 25. Face Verification Threshold

```python
FR_DISTANCE_THRESHOLD = 0.30
```

Rule:

```text
distance <= threshold -> MATCH
distance > threshold  -> NOT MATCH
```

## 26. Liveness Threshold

```python
FAS_LIVE_THRESHOLD = 0.50
```

Rule:

```text
LIVE probability >= threshold -> LIVE
LIVE probability < threshold  -> FAKE
```

## 27. Maximum Decision Time

```python
MAX_DECISION_SECONDS = 4.0
```

Jika keputusan belum tersedia hingga batas tersebut:

```text
RETRY
```

---

# Evaluation

## 28. Stage 1 — Face Verification

### Genuine Test

Orang yang benar menggunakan reference identity miliknya sendiri.

Metric:

```text
False Reject Rate (FRR)
Genuine Accept Rate
```

### Impostor Test

Orang lain mencoba menggunakan reference identity milik orang lain.

Metric:

```text
False Accept Rate (FAR)
```

## 29. Stage 2 — Face Anti-Spoofing

Metric:

```text
APCER
BPCER
ACER
```

```text
APCER =
spoof yang salah dianggap LIVE
------------------------------
seluruh spoof sample
```

```text
BPCER =
real face yang salah dianggap FAKE
----------------------------------
seluruh bona fide sample
```

```text
ACER = (APCER + BPCER) / 2
```

## 30. End-to-End Evaluation

Sistem akhir dapat dievaluasi menggunakan:

- verified success rate;
- spoof rejection rate;
- wrong-person rejection;
- retry rate;
- average decision time;
- median decision time;
- failure cases.

---

# Troubleshooting

## 31. Model Tidak Ditemukan

Periksa:

```powershell
Get-ChildItem .\models\commercial_test -Recurse
```

Pastikan struktur folder sesuai dengan bagian Model Setup.

## 32. Webcam Tidak Terbuka

Coba:

```text
--camera 1
```

atau:

```text
--camera 2
```

## 33. Liveness Lambat

Swin-V2 Base cukup berat.

Optimisasi berikutnya yang dapat dilakukan:

```text
PyTorch checkpoint
        |
        v
ONNX export
        |
        v
OpenVINO
        |
        +--> FP16
        |
        +--> INT8
```

## 34. Genuine Face Kadang Ditolak

Kemungkinan:

- wajah terlalu blur;
- pencahayaan buruk;
- sudut wajah terlalu ekstrem;
- reference image kurang baik;
- threshold belum dikalibrasi;
- wajah terlalu kecil pada frame.

## 35. Liveness Flicker

Keputusan raw per-frame dapat berubah.

Karena itu sistem memakai temporal aggregation terhadap beberapa hasil inference terbaru.

---

# Current Pipeline Summary

```text
STAGE 1 — IDENTITY

OpenVINO Runtime
    |
face-detection-retail-0004
    |
landmarks-regression-retail-0009
    |
face alignment
    |
face-reidentification-retail-0095
    |
256-D embedding
    |
cosine distance
    |
MATCH / NOT MATCH


STAGE 2 — LIVENESS

Face crop
    |
CVPR2024 Swin Transformer V2 Base
    |
background asynchronous inference
    |
temporal aggregation
    |
LIVE / FAKE


FINAL

MATCH + LIVE
    -> VERIFIED

MATCH + FAKE
    -> SPOOF ATTACK

NOT MATCH + LIVE
    -> WRONG PERSON

NOT MATCH + FAKE
    -> WRONG PERSON + SPOOF
```
