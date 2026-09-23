from pathlib import Path
import argparse
import time
from collections import deque

import cv2
import numpy as np
import openvino as ov


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_DIR = (
    ROOT
    / "models"
    / "commercial_test"
    / "openvino_fr"
)

FD_XML = (
    MODEL_DIR
    / "face-detection-retail-0004"
    / "FP16"
    / "face-detection-retail-0004.xml"
)

LM_XML = (
    MODEL_DIR
    / "landmarks-regression-retail-0009"
    / "FP16"
    / "landmarks-regression-retail-0009.xml"
)

REID_XML = (
    MODEL_DIR
    / "face-reidentification-retail-0095"
    / "FP16"
    / "face-reidentification-retail-0095.xml"
)

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

FACE_DET_THRESHOLD = 0.60
BBOX_EXPAND_RATIO = 1.15

# Open Model Zoo face-recognition demo default:
# cosine distance threshold = 0.3
#
# Their distance is:
#   distance = (1 - cosine_similarity) * 0.5
#
# MATCH if distance <= threshold.
ID_DISTANCE_THRESHOLD = 0.30

SMOOTHING_WINDOW = 10

WINDOW_NAME = "COMMERCIAL 1:1 FACE VERIFICATION - OpenVINO"


# Official reference landmarks used by the Open Model Zoo
# Face Recognition demo.
REFERENCE_LANDMARKS = np.array(
    [
        (30.2946 / 96, 51.6963 / 112),
        (65.5318 / 96, 51.5014 / 112),
        (48.0252 / 96, 71.7366 / 112),
        (33.5493 / 96, 92.3655 / 112),
        (62.7299 / 96, 92.2041 / 112),
    ],
    dtype=np.float32,
)


# ============================================================
# ARGUMENT
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "1:1 face verification using only OpenVINO/Open Model Zoo "
        "commercial-friendly models."
    )
)

parser.add_argument(
    "--reference",
    required=True,
    help="Path ke foto referensi, contoh: FALAH.jpg",
)

parser.add_argument(
    "--camera",
    type=int,
    default=CAMERA_INDEX,
)

args = parser.parse_args()

REFERENCE_PATH = Path(args.reference)

if not REFERENCE_PATH.exists():
    raise FileNotFoundError(
        f"Reference image tidak ditemukan: {REFERENCE_PATH}"
    )


# ============================================================
# CHECK MODEL FILES
# ============================================================

for model_path in [
    FD_XML,
    LM_XML,
    REID_XML,
]:
    if not model_path.exists():
        raise FileNotFoundError(
            "\nModel belum ditemukan:\n"
            f"{model_path}\n"
            "\nPastikan model OpenVINO sudah didownload."
        )


# ============================================================
# LOAD OPENVINO MODELS
# ============================================================

print("=" * 76)
print("LOADING COMMERCIAL-FRIENDLY 1:1 FACE VERIFICATION STACK")
print("=" * 76)

core = ov.Core()

fd_model = core.read_model(str(FD_XML))
lm_model = core.read_model(str(LM_XML))
reid_model = core.read_model(str(REID_XML))

fd = core.compile_model(fd_model, "CPU")
lm = core.compile_model(lm_model, "CPU")
reid = core.compile_model(reid_model, "CPU")

fd_input = fd.input(0)
fd_output = fd.output(0)

lm_input = lm.input(0)
lm_output = lm.output(0)

reid_input = reid.input(0)
reid_output = reid.output(0)

print("Face detector :", FD_XML.name)
print("Landmarks     :", LM_XML.name)
print("Face verifier :", REID_XML.name)
print()
print("FD input      :", list(fd_input.shape))
print("LM input      :", list(lm_input.shape))
print("REID input    :", list(reid_input.shape))
print("REID output   :", list(reid_output.shape))
print()
print("Official distance threshold:", ID_DISTANCE_THRESHOLD)


# ============================================================
# IMAGE HELPERS
# ============================================================

def to_nchw_bgr(image, width, height):
    resized = cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    blob = resized.astype(
        np.float32
    )

    blob = np.transpose(
        blob,
        (2, 0, 1),
    )

    blob = np.expand_dims(
        blob,
        axis=0,
    )

    return np.ascontiguousarray(
        blob,
        dtype=np.float32,
    )


def expand_bbox(
    bbox,
    frame_shape,
    ratio=BBOX_EXPAND_RATIO,
):
    h, w = frame_shape[:2]

    x1, y1, x2, y2 = map(
        float,
        bbox,
    )

    bw = x2 - x1
    bh = y2 - y1

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    bw *= ratio
    bh *= ratio

    nx1 = max(
        0,
        int(round(cx - bw / 2.0)),
    )

    ny1 = max(
        0,
        int(round(cy - bh / 2.0)),
    )

    nx2 = min(
        w,
        int(round(cx + bw / 2.0)),
    )

    ny2 = min(
        h,
        int(round(cy + bh / 2.0)),
    )

    return (
        nx1,
        ny1,
        nx2,
        ny2,
    )


# ============================================================
# FACE DETECTION
# ============================================================

def detect_faces(frame):
    h, w = frame.shape[:2]

    blob = to_nchw_bgr(
        frame,
        300,
        300,
    )

    result = fd(
        [blob]
    )[fd_output]

    boxes = []

    detections = np.asarray(
        result
    ).reshape(-1, 7)

    for det in detections:
        image_id = det[0]

        if image_id < 0:
            continue

        confidence = float(
            det[2]
        )

        if confidence < FACE_DET_THRESHOLD:
            continue

        x1 = int(
            np.clip(
                det[3] * w,
                0,
                w - 1,
            )
        )

        y1 = int(
            np.clip(
                det[4] * h,
                0,
                h - 1,
            )
        )

        x2 = int(
            np.clip(
                det[5] * w,
                x1 + 1,
                w,
            )
        )

        y2 = int(
            np.clip(
                det[6] * h,
                y1 + 1,
                h,
            )
        )

        boxes.append(
            (
                x1,
                y1,
                x2,
                y2,
                confidence,
            )
        )

    return boxes


# ============================================================
# LANDMARKS
# ============================================================

def get_landmarks(
    frame,
    bbox,
):
    x1, y1, x2, y2 = expand_bbox(
        bbox,
        frame.shape,
    )

    face_crop = frame[
        y1:y2,
        x1:x2,
    ]

    if face_crop.size == 0:
        return None

    blob = to_nchw_bgr(
        face_crop,
        48,
        48,
    )

    output = lm(
        [blob]
    )[lm_output]

    points = np.asarray(
        output,
        dtype=np.float32,
    ).reshape(-1)

    if points.size < 10:
        return None

    # normalized positions within expanded ROI
    landmarks = points[:10].reshape(
        5,
        2,
    )

    return (
        face_crop,
        landmarks,
        (x1, y1, x2, y2),
    )


# ============================================================
# OFFICIAL-LIKE ALIGNMENT
# ============================================================

def normalize_points(
    array,
    axis,
):
    arr = array.astype(
        np.float64
    ).copy()

    mean = arr.mean(
        axis=axis
    )

    arr -= mean

    std = arr.std()

    if std < 1e-12:
        raise ValueError(
            "Landmark geometry degenerate."
        )

    arr /= std

    return (
        arr,
        mean,
        std,
    )


def get_transform(
    src,
    dst,
):
    src_n, src_mean, src_std = normalize_points(
        src,
        axis=0,
    )

    dst_n, dst_mean, dst_std = normalize_points(
        dst,
        axis=0,
    )

    u, _, vt = np.linalg.svd(
        np.matmul(
            src_n.T,
            dst_n,
        )
    )

    r = np.matmul(
        u,
        vt,
    ).T

    transform = np.empty(
        (2, 3),
        dtype=np.float64,
    )

    transform[:, 0:2] = (
        r
        * (
            dst_std
            / src_std
        )
    )

    transform[:, 2] = (
        dst_mean.T
        - np.matmul(
            transform[:, 0:2],
            src_mean.T,
        )
    )

    return transform


def align_face(
    face_crop,
    landmarks,
):
    h, w = face_crop.shape[:2]

    scale = np.array(
        (w, h),
        dtype=np.float32,
    )

    desired = (
        REFERENCE_LANDMARKS
        * scale
    )

    actual = (
        landmarks
        * scale
    )

    transform = get_transform(
        desired,
        actual,
    )

    aligned = cv2.warpAffine(
        face_crop,
        transform,
        (w, h),
        flags=cv2.WARP_INVERSE_MAP,
    )

    return aligned


# ============================================================
# EMBEDDING
# ============================================================

def get_embedding(
    frame,
    bbox,
):
    lm_result = get_landmarks(
        frame,
        bbox,
    )

    if lm_result is None:
        return None

    (
        face_crop,
        landmarks,
        expanded_bbox,
    ) = lm_result

    aligned = align_face(
        face_crop,
        landmarks,
    )

    blob = to_nchw_bgr(
        aligned,
        128,
        128,
    )

    output = reid(
        [blob]
    )[reid_output]

    embedding = np.asarray(
        output,
        dtype=np.float32,
    ).reshape(-1)

    norm = np.linalg.norm(
        embedding
    )

    if norm < 1e-12:
        return None

    embedding = (
        embedding
        / norm
    )

    return (
        embedding,
        landmarks,
        expanded_bbox,
    )


def compare_embeddings(
    emb_a,
    emb_b,
):
    cosine_similarity = float(
        np.dot(
            emb_a,
            emb_b,
        )
    )

    cosine_similarity = float(
        np.clip(
            cosine_similarity,
            -1.0,
            1.0,
        )
    )

    # Exactly the scaling used by Open Model Zoo
    # FacesDatabase.Identity.cosine_dist:
    scaled_distance = (
        1.0
        - cosine_similarity
    ) * 0.5

    return (
        cosine_similarity,
        scaled_distance,
    )


# ============================================================
# BUILD REFERENCE EMBEDDING
# ============================================================

print()
print("=" * 76)
print("BUILDING REFERENCE EMBEDDING")
print("=" * 76)

reference_image = cv2.imread(
    str(REFERENCE_PATH)
)

if reference_image is None:
    raise RuntimeError(
        "Reference image gagal dibaca."
    )

reference_faces = detect_faces(
    reference_image
)

if len(reference_faces) == 0:
    raise RuntimeError(
        "Tidak ada wajah terdeteksi pada reference image."
    )

if len(reference_faces) > 1:
    raise RuntimeError(
        "Reference image harus berisi tepat satu wajah."
    )

reference_bbox = reference_faces[0][:4]

reference_result = get_embedding(
    reference_image,
    reference_bbox,
)

if reference_result is None:
    raise RuntimeError(
        "Gagal membuat reference embedding."
    )

reference_embedding = (
    reference_result[0]
)

print("Reference :", REFERENCE_PATH)
print("Embedding :", reference_embedding.shape)
print("Reference face OK.")


# ============================================================
# UI COLORS
# ============================================================

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)


def draw_text(
    image,
    text,
    position,
    color=WHITE,
    scale=0.60,
    thickness=2,
):
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        BLACK,
        thickness + 3,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    args.camera,
    cv2.CAP_DSHOW,
)

if not cap.isOpened():
    cap = cv2.VideoCapture(
        args.camera
    )

if not cap.isOpened():
    raise RuntimeError(
        "Webcam tidak dapat dibuka."
    )

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH,
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT,
)

similarity_history = deque(
    maxlen=SMOOTHING_WINDOW
)

distance_history = deque(
    maxlen=SMOOTHING_WINDOW
)

fps_history = deque(
    maxlen=30
)

previous_time = time.perf_counter()


print()
print("=" * 76)
print("REALTIME 1:1 FACE VERIFICATION STARTED")
print("=" * 76)
print("Model : face-reidentification-retail-0095")
print("R     : reset smoothing")
print("Q/ESC : exit")
print()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ok, frame = cap.read()

    if not ok:
        break

    start = time.perf_counter()

    faces = detect_faces(
        frame
    )

    raw_similarity = 0.0
    raw_distance = 1.0

    median_similarity = 0.0
    median_distance = 1.0

    state = "NO FACE"

    bbox_to_draw = None


    # ========================================================
    # EXACTLY ONE FACE
    # ========================================================

    if len(faces) == 1:

        bbox = faces[0][:4]

        result = get_embedding(
            frame,
            bbox,
        )

        if result is not None:

            current_embedding = (
                result[0]
            )

            (
                raw_similarity,
                raw_distance,
            ) = compare_embeddings(
                reference_embedding,
                current_embedding,
            )

            similarity_history.append(
                raw_similarity
            )

            distance_history.append(
                raw_distance
            )

            median_similarity = float(
                np.median(
                    list(similarity_history)
                )
            )

            median_distance = float(
                np.median(
                    list(distance_history)
                )
            )

            state = (
                "MATCH"
                if median_distance <= ID_DISTANCE_THRESHOLD
                else "NOT MATCH"
            )

            bbox_to_draw = bbox

        else:
            state = "ALIGNMENT ERROR"


    # ========================================================
    # MULTIPLE / NO FACE
    # ========================================================

    elif len(faces) > 1:

        state = "MULTIPLE FACES"

        similarity_history.clear()
        distance_history.clear()

    else:

        state = "NO FACE"

        similarity_history.clear()
        distance_history.clear()


    # ========================================================
    # FPS
    # ========================================================

    now = time.perf_counter()

    processing_ms = (
        now - start
    ) * 1000.0

    dt = (
        now - previous_time
    )

    previous_time = now

    if dt > 0:
        fps_history.append(
            1.0 / dt
        )

    fps = (
        sum(fps_history)
        / len(fps_history)
        if fps_history
        else 0.0
    )


    # ========================================================
    # DRAW
    # ========================================================

    if state == "MATCH":
        state_color = GREEN

    elif state == "NOT MATCH":
        state_color = RED

    else:
        state_color = YELLOW


    if bbox_to_draw is not None:

        x1, y1, x2, y2 = map(
            int,
            bbox_to_draw,
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            state_color,
            3,
        )


    cv2.rectangle(
        frame,
        (0, 0),
        (CAMERA_WIDTH, 220),
        BLACK,
        -1,
    )

    draw_text(
        frame,
        "COMMERCIAL-FRIENDLY 1:1 FACE VERIFICATION",
        (18, 30),
        WHITE,
        0.57,
    )

    draw_text(
        frame,
        "OpenVINO / Open Model Zoo",
        (18, 58),
        CYAN,
        0.50,
    )

    draw_text(
        frame,
        f"RESULT: {state}",
        (18, 100),
        state_color,
        0.82,
        2,
    )

    draw_text(
        frame,
        (
            f"RAW cosine similarity: "
            f"{raw_similarity:.4f}"
        ),
        (18, 133),
        WHITE,
        0.52,
    )

    draw_text(
        frame,
        (
            f"MEDIAN({len(similarity_history)}): "
            f"similarity={median_similarity:.4f}  "
            f"distance={median_distance:.4f}"
        ),
        (18, 162),
        WHITE,
        0.50,
    )

    draw_text(
        frame,
        (
            f"Official distance threshold: "
            f"{ID_DISTANCE_THRESHOLD:.2f}"
        ),
        (18, 190),
        CYAN,
        0.48,
    )

    draw_text(
        frame,
        (
            f"FPS={fps:.1f}  "
            f"Processing={processing_ms:.1f} ms  "
            f"Faces={len(faces)}"
        ),
        (18, 214),
        WHITE,
        0.45,
    )


    cv2.imshow(
        WINDOW_NAME,
        frame,
    )

    key = (
        cv2.waitKey(1)
        & 0xFF
    )

    if key == ord("q") or key == 27:
        break

    if key == ord("r"):
        similarity_history.clear()
        distance_history.clear()


cap.release()
cv2.destroyAllWindows()
