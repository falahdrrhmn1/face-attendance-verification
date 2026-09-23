# Face Attendance Verification Prototype

Prototype sistem absensi wajah berbasis multi-stage biometric verification.

## Pipeline

### Stage 1 - 1:1 Face Verification

Intel Open Model Zoo pretrained models:

- face-detection-retail-0004
- landmarks-regression-retail-0009
- face-reidentification-retail-0095

Runtime: OpenVINO.

Face recognition menghasilkan embedding 256-D dan melakukan 1:1 verification menggunakan cosine distance.

### Stage 2 - Face Anti-Spoofing

- CVPR 2024 Face Anti-Spoofing
- face_swin_v2_base.pth
- Swin Transformer V2 Base
- Class 0: LIVE
- Class 1: SPOOF

Liveness inference dijalankan secara asynchronous agar model tidak memblokir camera preview.

## Final Decision

| Face Verification | Liveness | Result |
|---|---|---|
| MATCH | LIVE | VERIFIED |
| MATCH | FAKE | SPOOF ATTACK |
| NOT MATCH | LIVE | WRONG PERSON |
| NOT MATCH | FAKE | WRONG PERSON + SPOOF |

## Model Strategy

Semua neural-network model yang digunakan merupakan pretrained model.

Tidak dilakukan fine-tuning pada prototype saat ini.

Kontribusi implementasi berada pada level sistem:

- multi-model pipeline
- face alignment
- cosine-distance verification
- temporal aggregation
- asynchronous liveness inference
- decision-level fusion
- retry and decision-time mechanism

## Run

Run the main prototype with:

    .\.venv\Scripts\python.exe .\attendance_full_commercial_candidate_async_timed.py --reference ".\FALAH.jpg"

## License Note

OpenVINO/Open Model Zoo digunakan sebagai commercial-friendly candidate untuk face verification.

CVPR2024 Swin-V2 anti-spoofing berasal dari repository berlisensi MIT. Hak dan provenance checkpoint tetap perlu diverifikasi sebelum digunakan untuk deployment komersial produksi.

Model weights dan biometric reference images tidak disertakan di repository ini.

## Status

Research prototype. Not yet intended as a production biometric attendance system.
