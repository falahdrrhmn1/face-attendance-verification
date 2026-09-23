from pathlib import Path
import argparse
import sys
import time
import threading
import queue
from collections import deque

import cv2
import numpy as np
import openvino as ov
import torch
import torch.nn.functional as F


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent

# ----------------------------
# Stage 1: OpenVINO 1:1 FR
# ----------------------------

FR_MODEL_DIR = (
    ROOT
    / "models"
    / "commercial_test"
    / "openvino_fr"
)

FD_XML = (
    FR_MODEL_DIR
    / "face-detection-retail-0004"
    / "FP16"
    / "face-detection-retail-0004.xml"
)

LM_XML = (
    FR_MODEL_DIR
    / "landmarks-regression-retail-0009"
    / "FP16"
    / "landmarks-regression-retail-0009.xml"
)

REID_XML = (
    FR_MODEL_DIR
    / "face-reidentification-retail-0095"
    / "FP16"
    / "face-reidentification-retail-0095.xml"
)

FACE_DET_THRESHOLD = 0.60
BBOX_EXPAND_RATIO = 1.15

# Open Model Zoo face-recognition demo default threshold
# distance = (1 - cosine_similarity) * 0.5
FR_DISTANCE_THRESHOLD = 0.30

FR_SMOOTHING_WINDOW = 10
FR_MIN_SAMPLES = 5


# ----------------------------
# Stage 2: CVPR2024 Swin-V2
# ----------------------------

FAS_REPO_DIR = (
    ROOT
    / "models"
    / "commercial_test"
    / "cvpr2024_fas"
)

FAS_WEIGHT_PATH = (
    FAS_REPO_DIR
    / "weights"
    / "face_swin_v2_base.pth"
)

FAS_SWIN_SOURCE = (
    FAS_REPO_DIR
    / "nets"
    / "swin_transformer_v2.py"
)

FAS_INPUT_SIZE = 224
FAS_LIVE_THRESHOLD = 0.50

# Hanya butuh sedikit sampel karena 1 inference CPU cukup berat.
FAS_SMOOTHING_WINDOW = 3
FAS_MIN_SAMPLES = 2

# Main loop tidak menunggu Swin.
# Frame terbaru dikirim ke worker secara periodik.
FAS_SUBMIT_INTERVAL = 0.15

# Maximum time allowed for one verification attempt.
# Timer starts when exactly one face is first detected after reset/retry.
MAX_DECISION_SECONDS = 4.0

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
).reshape(1, 1, 3)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
).reshape(1, 1, 3)


# ----------------------------
# Camera / UI
# ----------------------------

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

WINDOW_NAME = "ATTENDANCE - OPENVINO 1:1 + CVPR2024 LIVENESS"

VIDEO_W = 800
VIDEO_H = 600

PANEL_W = 430
STATUS_H = 75

CANVAS_W = VIDEO_W + PANEL_W
CANVAS_H = VIDEO_H + STATUS_H

BUTTON_X1 = VIDEO_W + 45
BUTTON_Y1 = 475
BUTTON_X2 = CANVAS_W - 45
BUTTON_Y2 = 545


# ============================================================
# ARGUMENT
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Attendance candidate stack: OpenVINO 1:1 verification "
        "+ CVPR2024 Swin-V2 liveness."
    )
)

parser.add_argument(
    "--reference",
    required=True,
    help="Path foto reference, contoh: .\\FALAH.jpg",
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
# CHECK FILES
# ============================================================

required_files = [
    FD_XML,
    LM_XML,
    REID_XML,
    FAS_WEIGHT_PATH,
    FAS_SWIN_SOURCE,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"File/model belum ditemukan:\n{path}"
        )


# ============================================================
# COLORS
# ============================================================

BLACK = (0, 0, 0)
DARK = (25, 25, 25)
WHITE = (255, 255, 255)
GRAY = (180, 180, 180)

GREEN = (0, 235, 0)
RED = (0, 0, 255)
YELLOW = (0, 230, 255)
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


def state_color(state):
    if state in ("MATCH", "LIVE", "VERIFIED"):
        return GREEN

    if state in (
        "NOT MATCH",
        "FAKE",
        "SPOOF ATTACK",
        "WRONG PERSON",
        "WRONG PERSON + SPOOF",
    ):
        return RED

    return YELLOW


# ============================================================
# STAGE 1 — OPENVINO FACE VERIFICATION
# ============================================================

print("=" * 78)
print("LOADING STAGE 1 - OPENVINO 1:1 FACE VERIFICATION")
print("=" * 78)

ov_core = ov.Core()

fd_model = ov_core.read_model(str(FD_XML))
lm_model = ov_core.read_model(str(LM_XML))
reid_model = ov_core.read_model(str(REID_XML))

fd = ov_core.compile_model(fd_model, "CPU")
lm = ov_core.compile_model(lm_model, "CPU")
reid = ov_core.compile_model(reid_model, "CPU")

fd_output = fd.output(0)
lm_output = lm.output(0)
reid_output = reid.output(0)

print("Face detector :", FD_XML.name)
print("Landmarks     :", LM_XML.name)
print("Face verifier :", REID_XML.name)
print("FR threshold  :", FR_DISTANCE_THRESHOLD)


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


def to_nchw_bgr(image, width, height):
    resized = cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    blob = resized.astype(np.float32)
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, axis=0)

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

    x1, y1, x2, y2 = map(float, bbox)

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

    return nx1, ny1, nx2, ny2


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

    detections = np.asarray(
        result
    ).reshape(-1, 7)

    boxes = []

    for det in detections:
        if det[0] < 0:
            continue

        confidence = float(det[2])

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

    landmarks = points[:10].reshape(
        5,
        2,
    )

    return (
        face_crop,
        landmarks,
        (x1, y1, x2, y2),
    )


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

    embedding /= norm

    return embedding


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

    distance = (
        1.0
        - cosine_similarity
    ) * 0.5

    return (
        cosine_similarity,
        distance,
    )


# ============================================================
# REFERENCE EMBEDDING
# ============================================================

print()
print("=" * 78)
print("BUILDING REFERENCE EMBEDDING")
print("=" * 78)

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
        "Tidak ada wajah pada reference image."
    )

if len(reference_faces) > 1:
    raise RuntimeError(
        "Reference image harus berisi tepat satu wajah."
    )

reference_bbox = reference_faces[0][:4]

reference_embedding = get_embedding(
    reference_image,
    reference_bbox,
)

if reference_embedding is None:
    raise RuntimeError(
        "Gagal membuat reference embedding."
    )

print("Reference :", REFERENCE_PATH)
print("Embedding :", reference_embedding.shape)
print("Reference face OK.")


# ============================================================
# STAGE 2 — LOAD CVPR2024 SWIN-V2
# ============================================================

print()
print("=" * 78)
print("LOADING STAGE 2 - CVPR2024 FACE SWIN-V2 LIVENESS")
print("=" * 78)

# Source resmi punya satu .cuda() hardcoded.
source_text = FAS_SWIN_SOURCE.read_text(
    encoding="utf-8"
)

old_text = "torch.tensor(1. / 0.01).cuda()"
new_text = "torch.tensor(1. / 0.01, device=x.device)"

if old_text in source_text:
    FAS_SWIN_SOURCE.write_text(
        source_text.replace(
            old_text,
            new_text,
        ),
        encoding="utf-8",
    )

    print(
        "CPU compatibility patch applied."
    )
else:
    print(
        "CPU compatibility patch not needed."
    )


sys.path.insert(
    0,
    str(FAS_REPO_DIR),
)

try:
    from nets.utils import get_model
except ModuleNotFoundError as exc:
    if exc.name == "timm":
        raise RuntimeError(
            r"Install dulu: .\.venv\Scripts\python.exe -m pip install timm"
        ) from exc

    if exc.name == "thop":
        raise RuntimeError(
            r"Install dulu: .\.venv\Scripts\python.exe -m pip install thop"
        ) from exc

    raise


FAS_DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# Agar Swin CPU tidak mengambil semua thread dan membuat webcam macet.
if FAS_DEVICE.type == "cpu":
    try:
        torch.set_num_threads(2)
    except RuntimeError:
        pass

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


fas_model = get_model(
    "swin_v2_b",
    num_classes=2,
)

checkpoint = torch.load(
    str(FAS_WEIGHT_PATH),
    map_location="cpu",
    weights_only=False,
)

state_dict = checkpoint["state_dict"]

clean_state_dict = {}

for key, value in state_dict.items():
    new_key = key

    while new_key.startswith(
        "module."
    ):
        new_key = new_key[
            len("module.") :
        ]

    clean_state_dict[
        new_key
    ] = value


load_result = fas_model.load_state_dict(
    clean_state_dict,
    strict=False,
)

print("Device          :", FAS_DEVICE)
print("Checkpoint arch :", checkpoint.get("arch"))
print("Missing keys    :", len(load_result.missing_keys))
print("Unexpected keys :", len(load_result.unexpected_keys))

fas_model.to(
    FAS_DEVICE
)

fas_model.eval()

print("Swin-V2 liveness loaded.")


# ============================================================
# FAS HELPERS
# ============================================================

def crop_for_liveness(
    frame,
    bbox,
    padding_ratio=0.08,
):
    h, w = frame.shape[:2]

    x1, y1, x2, y2 = map(
        float,
        bbox,
    )

    bw = x2 - x1
    bh = y2 - y1

    pad_x = bw * padding_ratio
    pad_y = bh * padding_ratio

    x1 = int(
        max(
            0,
            x1 - pad_x,
        )
    )

    y1 = int(
        max(
            0,
            y1 - pad_y,
        )
    )

    x2 = int(
        min(
            w,
            x2 + pad_x,
        )
    )

    y2 = int(
        min(
            h,
            y2 + pad_y,
        )
    )

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[
        y1:y2,
        x1:x2,
    ]

    if crop.size == 0:
        return None

    return crop.copy()


def preprocess_fas(
    face_crop,
):
    image = cv2.resize(
        face_crop,
        (
            FAS_INPUT_SIZE,
            FAS_INPUT_SIZE,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    image = (
        image.astype(
            np.float32
        )
        / 255.0
    )

    image = (
        image
        - IMAGENET_MEAN
    ) / IMAGENET_STD

    image = np.transpose(
        image,
        (2, 0, 1),
    )

    tensor = torch.from_numpy(
        np.ascontiguousarray(
            image
        )
    ).unsqueeze(0)

    return tensor.to(
        FAS_DEVICE,
        non_blocking=True,
    )


@torch.inference_mode()
def run_fas(
    face_crop,
):
    tensor = preprocess_fas(
        face_crop
    )

    output = fas_model(
        tensor
    )

    if isinstance(
        output,
        (tuple, list),
    ):
        logits = output[-1]
    else:
        logits = output

    probs = F.softmax(
        logits,
        dim=1,
    )[0]

    p_live = float(
        probs[0]
        .detach()
        .cpu()
    )

    p_spoof = float(
        probs[1]
        .detach()
        .cpu()
    )

    state = (
        "LIVE"
        if p_live >= FAS_LIVE_THRESHOLD
        else "FAKE"
    )

    return (
        state,
        p_live,
        p_spoof,
    )


# ============================================================
# ASYNC LIVENESS WORKER
# ============================================================

fas_queue = queue.Queue(
    maxsize=1
)

fas_lock = threading.Lock()

fas_worker_state = {
    "session_id": -1,
    "sample_id": 0,
    "state": "WAITING",
    "p_live": 0.0,
    "p_spoof": 0.0,
    "infer_ms": 0.0,
    "busy": False,
}

fas_stop_event = threading.Event()


def fas_worker():
    while not fas_stop_event.is_set():

        try:
            item = fas_queue.get(
                timeout=0.1
            )
        except queue.Empty:
            continue

        if item is None:
            break

        (
            item_session_id,
            crop,
        ) = item

        with fas_lock:
            fas_worker_state[
                "busy"
            ] = True

        infer_start = time.perf_counter()

        try:
            (
                pred_state,
                p_live,
                p_spoof,
            ) = run_fas(
                crop
            )

            infer_ms = (
                time.perf_counter()
                - infer_start
            ) * 1000.0

            with fas_lock:
                fas_worker_state[
                    "session_id"
                ] = item_session_id

                fas_worker_state[
                    "sample_id"
                ] += 1

                fas_worker_state[
                    "state"
                ] = pred_state

                fas_worker_state[
                    "p_live"
                ] = p_live

                fas_worker_state[
                    "p_spoof"
                ] = p_spoof

                fas_worker_state[
                    "infer_ms"
                ] = infer_ms

        finally:
            with fas_lock:
                fas_worker_state[
                    "busy"
                ] = False

            fas_queue.task_done()


fas_thread = threading.Thread(
    target=fas_worker,
    daemon=True,
)

fas_thread.start()


def submit_latest_fas_crop(
    session_id,
    crop,
):
    """
    Queue size = 1.
    Kalau masih ada crop lama yang belum diproses,
    crop lama dibuang dan diganti frame terbaru.
    """

    try:
        while True:
            fas_queue.get_nowait()
            fas_queue.task_done()
    except queue.Empty:
        pass

    try:
        fas_queue.put_nowait(
            (
                session_id,
                crop,
            )
        )
    except queue.Full:
        pass


# ============================================================
# DECISION STATE
# ============================================================

fr_similarity_history = deque(
    maxlen=FR_SMOOTHING_WINDOW
)

fr_distance_history = deque(
    maxlen=FR_SMOOTHING_WINDOW
)

fas_live_history = deque(
    maxlen=FAS_SMOOTHING_WINDOW
)

session_id = 0

last_consumed_fas_sample = 0
last_fas_submit_time = 0.0

raw_fr_similarity = 0.0
raw_fr_distance = 1.0

median_fr_similarity = 0.0
median_fr_distance = 1.0

fr_state = "CHECKING"

raw_fas_state = "WAITING"
raw_fas_live = 0.0
raw_fas_spoof = 0.0
raw_fas_infer_ms = 0.0

fas_state = "CHECKING"
median_fas_live = 0.0

final_state = None

# Timing for one attendance attempt.
# Starts only when exactly one face is detected.
session_started_at = None
decision_time_sec = None
session_elapsed_sec = 0.0

latest_bbox = None

retry_requested = False
fullscreen = False


def purge_fas_queue():
    try:
        while True:
            fas_queue.get_nowait()
            fas_queue.task_done()
    except queue.Empty:
        pass


def reset_session():
    global session_id
    global last_consumed_fas_sample
    global last_fas_submit_time

    global raw_fr_similarity
    global raw_fr_distance
    global median_fr_similarity
    global median_fr_distance
    global fr_state

    global raw_fas_state
    global raw_fas_live
    global raw_fas_spoof
    global raw_fas_infer_ms
    global fas_state
    global median_fas_live

    global final_state
    global session_started_at
    global decision_time_sec
    global session_elapsed_sec
    global latest_bbox

    session_id += 1

    fr_similarity_history.clear()
    fr_distance_history.clear()
    fas_live_history.clear()

    purge_fas_queue()

    with fas_lock:
        last_consumed_fas_sample = (
            fas_worker_state[
                "sample_id"
            ]
        )

    last_fas_submit_time = 0.0

    raw_fr_similarity = 0.0
    raw_fr_distance = 1.0

    median_fr_similarity = 0.0
    median_fr_distance = 1.0

    fr_state = "CHECKING"

    raw_fas_state = "WAITING"
    raw_fas_live = 0.0
    raw_fas_spoof = 0.0
    raw_fas_infer_ms = 0.0

    fas_state = "CHECKING"
    median_fas_live = 0.0

    final_state = None

    session_started_at = None
    decision_time_sec = None
    session_elapsed_sec = 0.0

    latest_bbox = None


def get_final_state(
    identity_state,
    liveness_state,
):
    if (
        identity_state == "MATCH"
        and liveness_state == "LIVE"
    ):
        return "VERIFIED"

    if (
        identity_state == "MATCH"
        and liveness_state == "FAKE"
    ):
        return "SPOOF ATTACK"

    if (
        identity_state == "NOT MATCH"
        and liveness_state == "LIVE"
    ):
        return "WRONG PERSON"

    if (
        identity_state == "NOT MATCH"
        and liveness_state == "FAKE"
    ):
        return "WRONG PERSON + SPOOF"

    return None


# ============================================================
# RETRY BUTTON
# ============================================================

def mouse_callback(
    event,
    x,
    y,
    flags,
    param,
):
    global retry_requested

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if (
        BUTTON_X1 <= x <= BUTTON_X2
        and BUTTON_Y1 <= y <= BUTTON_Y2
    ):
        retry_requested = True


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


cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL,
)

cv2.resizeWindow(
    WINDOW_NAME,
    CANVAS_W,
    CANVAS_H,
)

cv2.setMouseCallback(
    WINDOW_NAME,
    mouse_callback,
)


fps_history = deque(
    maxlen=30
)

previous_time = time.perf_counter()


print()
print("=" * 78)
print("FULL ATTENDANCE CANDIDATE STARTED")
print("=" * 78)
print("Stage 1 : OpenVINO 1:1 Face Verification")
print("Stage 2 : CVPR2024 Swin-V2 Liveness (ASYNC)")
print("Decision timer: starts when exactly one face is detected")
print("Max decision time:", MAX_DECISION_SECONDS, "seconds")
print()
print("R       : Retry / reset")
print("F       : Fullscreen")
print("Q / ESC : Exit")
print()


# ============================================================
# MAIN LOOP
# ============================================================

try:
    while True:

        if retry_requested:
            reset_session()
            retry_requested = False

        ok, frame = cap.read()

        if not ok:
            break

        main_start = time.perf_counter()

        faces = detect_faces(
            frame
        )

        now = time.perf_counter()


        # ====================================================
        # EXACTLY ONE FACE
        # ====================================================

        if len(faces) == 1:

            bbox = faces[0][:4]

            latest_bbox = bbox

            # Start timer only when one valid face is actually present.
            if (
                final_state is None
                and session_started_at is None
            ):
                session_started_at = now

            if (
                final_state is None
                and session_started_at is not None
            ):
                session_elapsed_sec = (
                    now - session_started_at
                )

            # --------------------------------------------
            # Stage 1 runs in main loop: light & fast
            # --------------------------------------------

            fr_result = get_embedding(
                frame,
                bbox,
            )

            if fr_result is not None:

                (
                    raw_fr_similarity,
                    raw_fr_distance,
                ) = compare_embeddings(
                    reference_embedding,
                    fr_result,
                )

                if final_state is None:
                    fr_similarity_history.append(
                        raw_fr_similarity
                    )

                    fr_distance_history.append(
                        raw_fr_distance
                    )


                if fr_distance_history:

                    median_fr_similarity = float(
                        np.median(
                            list(
                                fr_similarity_history
                            )
                        )
                    )

                    median_fr_distance = float(
                        np.median(
                            list(
                                fr_distance_history
                            )
                        )
                    )

                    if (
                        len(fr_distance_history)
                        >= FR_MIN_SAMPLES
                    ):
                        fr_state = (
                            "MATCH"
                            if (
                                median_fr_distance
                                <= FR_DISTANCE_THRESHOLD
                            )
                            else "NOT MATCH"
                        )

                    else:
                        fr_state = "CHECKING"


            # --------------------------------------------
            # Submit liveness crop to BACKGROUND worker
            # --------------------------------------------

            if final_state is None:

                if (
                    now - last_fas_submit_time
                    >= FAS_SUBMIT_INTERVAL
                ):
                    fas_crop = crop_for_liveness(
                        frame,
                        bbox,
                    )

                    if fas_crop is not None:
                        submit_latest_fas_crop(
                            session_id,
                            fas_crop,
                        )

                        last_fas_submit_time = now


            # --------------------------------------------
            # Consume NEW liveness result if available
            # --------------------------------------------

            with fas_lock:
                worker_snapshot = dict(
                    fas_worker_state
                )

            if (
                worker_snapshot[
                    "sample_id"
                ]
                != last_consumed_fas_sample
                and worker_snapshot[
                    "session_id"
                ] == session_id
            ):
                last_consumed_fas_sample = (
                    worker_snapshot[
                        "sample_id"
                    ]
                )

                raw_fas_state = (
                    worker_snapshot[
                        "state"
                    ]
                )

                raw_fas_live = float(
                    worker_snapshot[
                        "p_live"
                    ]
                )

                raw_fas_spoof = float(
                    worker_snapshot[
                        "p_spoof"
                    ]
                )

                raw_fas_infer_ms = float(
                    worker_snapshot[
                        "infer_ms"
                    ]
                )

                if final_state is None:
                    fas_live_history.append(
                        raw_fas_live
                    )


            # --------------------------------------------
            # Aggregate liveness
            # --------------------------------------------

            if fas_live_history:

                median_fas_live = float(
                    np.median(
                        list(
                            fas_live_history
                        )
                    )
                )

                if (
                    len(fas_live_history)
                    >= FAS_MIN_SAMPLES
                ):
                    fas_state = (
                        "LIVE"
                        if (
                            median_fas_live
                            >= FAS_LIVE_THRESHOLD
                        )
                        else "FAKE"
                    )

                else:
                    fas_state = "CHECKING"

            else:
                fas_state = "CHECKING"


            # --------------------------------------------
            # Final decision
            # --------------------------------------------

            if final_state is None:

                candidate = get_final_state(
                    fr_state,
                    fas_state,
                )

                if candidate is not None:
                    final_state = candidate

                    if session_started_at is not None:
                        decision_time_sec = (
                            now - session_started_at
                        )

                # Hard upper bound for one attempt.
                # If usable evidence is not ready in time, return RETRY.
                if (
                    final_state is None
                    and session_started_at is not None
                    and session_elapsed_sec
                    >= MAX_DECISION_SECONDS
                ):
                    final_state = "RETRY"
                    decision_time_sec = session_elapsed_sec


        # ====================================================
        # MULTIPLE / NO FACE
        # ====================================================

        elif len(faces) > 1:

            latest_bbox = None

            if final_state is None:
                reset_session()

            fr_state = "MULTIPLE FACES"
            fas_state = "MULTIPLE FACES"

        else:

            latest_bbox = None

            if final_state is None:
                reset_session()

            fr_state = "NO FACE"
            fas_state = "NO FACE"


        # ====================================================
        # MAIN LOOP FPS
        # ====================================================

        main_end = time.perf_counter()

        main_processing_ms = (
            main_end - main_start
        ) * 1000.0

        dt = (
            main_end - previous_time
        )

        previous_time = main_end

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


        # ====================================================
        # VIDEO AREA
        # ====================================================

        display_frame = cv2.resize(
            frame,
            (
                VIDEO_W,
                VIDEO_H,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        if latest_bbox is not None:

            sx = (
                VIDEO_W
                / CAMERA_WIDTH
            )

            sy = (
                VIDEO_H
                / CAMERA_HEIGHT
            )

            x1, y1, x2, y2 = map(
                int,
                latest_bbox,
            )

            dx1 = int(
                x1 * sx
            )

            dy1 = int(
                y1 * sy
            )

            dx2 = int(
                x2 * sx
            )

            dy2 = int(
                y2 * sy
            )

            if final_state is not None:
                box_color = state_color(
                    final_state
                )

            elif (
                fr_state == "MATCH"
                and fas_state == "LIVE"
            ):
                box_color = GREEN

            elif (
                fr_state in (
                    "NOT MATCH",
                )
                or fas_state == "FAKE"
            ):
                box_color = RED

            else:
                box_color = YELLOW


            cv2.rectangle(
                display_frame,
                (dx1, dy1),
                (dx2, dy2),
                box_color,
                4,
            )


        # ====================================================
        # CANVAS
        # ====================================================

        canvas = np.zeros(
            (
                CANVAS_H,
                CANVAS_W,
                3,
            ),
            dtype=np.uint8,
        )

        canvas[
            0:VIDEO_H,
            0:VIDEO_W,
        ] = display_frame

        cv2.rectangle(
            canvas,
            (
                VIDEO_W,
                0,
            ),
            (
                CANVAS_W,
                VIDEO_H,
            ),
            DARK,
            -1,
        )


        panel_x = VIDEO_W + 25


        # ====================================================
        # STAGE 1 PANEL
        # ====================================================

        draw_text(
            canvas,
            "1. FACE VERIFICATION",
            (
                panel_x,
                45,
            ),
            WHITE,
            0.58,
        )

        draw_text(
            canvas,
            fr_state,
            (
                panel_x,
                85,
            ),
            state_color(
                fr_state
            ),
            0.82,
        )

        draw_text(
            canvas,
            (
                f"median distance: "
                f"{median_fr_distance:.4f}"
            ),
            (
                panel_x,
                120,
            ),
            GRAY,
            0.47,
        )

        draw_text(
            canvas,
            (
                f"cosine: "
                f"{median_fr_similarity:.4f}"
            ),
            (
                panel_x,
                148,
            ),
            CYAN,
            0.47,
        )

        draw_text(
            canvas,
            (
                f"threshold <= "
                f"{FR_DISTANCE_THRESHOLD:.2f}"
            ),
            (
                panel_x,
                176,
            ),
            GRAY,
            0.43,
        )


        # ====================================================
        # STAGE 2 PANEL
        # ====================================================

        draw_text(
            canvas,
            "2. LIVENESS",
            (
                panel_x,
                230,
            ),
            WHITE,
            0.58,
        )

        draw_text(
            canvas,
            fas_state,
            (
                panel_x,
                270,
            ),
            state_color(
                fas_state
            ),
            0.82,
        )

        draw_text(
            canvas,
            (
                f"median LIVE: "
                f"{median_fas_live:.3f}"
            ),
            (
                panel_x,
                305,
            ),
            GRAY,
            0.47,
        )

        draw_text(
            canvas,
            (
                f"raw: {raw_fas_state} "
                f"L={raw_fas_live:.3f} "
                f"S={raw_fas_spoof:.3f}"
            ),
            (
                panel_x,
                333,
            ),
            CYAN,
            0.42,
        )

        with fas_lock:
            is_fas_busy = bool(
                fas_worker_state[
                    "busy"
                ]
            )

        draw_text(
            canvas,
            (
                f"Swin: "
                f"{raw_fas_infer_ms:.0f} ms "
                f"({'BUSY' if is_fas_busy else 'IDLE'})"
            ),
            (
                panel_x,
                362,
            ),
            GRAY,
            0.43,
        )

        draw_text(
            canvas,
            (
                f"samples: "
                f"{len(fas_live_history)}/"
                f"{FAS_MIN_SAMPLES}+"
            ),
            (
                panel_x,
                390,
            ),
            GRAY,
            0.43,
        )


        # ====================================================
        # PERFORMANCE
        # ====================================================

        draw_text(
            canvas,
            (
                f"Main FPS: {fps:.1f}  "
                f"Main: {main_processing_ms:.0f} ms"
            ),
            (
                panel_x,
                430,
            ),
            WHITE,
            0.45,
        )

        if decision_time_sec is not None:
            timing_text = (
                f"Decision time: "
                f"{decision_time_sec:.2f} s"
            )
        elif session_started_at is not None:
            timing_text = (
                f"Elapsed: "
                f"{session_elapsed_sec:.2f} / "
                f"{MAX_DECISION_SECONDS:.1f} s"
            )
        else:
            timing_text = (
                "Decision timer: waiting for face"
            )

        draw_text(
            canvas,
            timing_text,
            (
                panel_x,
                458,
            ),
            CYAN,
            0.43,
        )


        # ====================================================
        # RETRY BUTTON
        # ====================================================

        button_text = (
            "RETRY"
            if final_state is not None
            else "RESET"
        )

        cv2.rectangle(
            canvas,
            (
                BUTTON_X1,
                BUTTON_Y1,
            ),
            (
                BUTTON_X2,
                BUTTON_Y2,
            ),
            (
                80,
                80,
                180,
            ),
            -1,
        )

        cv2.rectangle(
            canvas,
            (
                BUTTON_X1,
                BUTTON_Y1,
            ),
            (
                BUTTON_X2,
                BUTTON_Y2,
            ),
            WHITE,
            2,
        )

        text_size = cv2.getTextSize(
            button_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            2,
        )[0]

        text_x = int(
            (
                BUTTON_X1
                + BUTTON_X2
                - text_size[0]
            )
            / 2
        )

        text_y = int(
            (
                BUTTON_Y1
                + BUTTON_Y2
                + text_size[1]
            )
            / 2
        )

        draw_text(
            canvas,
            button_text,
            (
                text_x,
                text_y,
            ),
            WHITE,
            0.85,
            2,
        )

        draw_text(
            canvas,
            "R = Retry/Reset",
            (
                panel_x,
                575,
            ),
            GRAY,
            0.40,
        )

        draw_text(
            canvas,
            "F = Fullscreen",
            (
                panel_x,
                597,
            ),
            GRAY,
            0.40,
        )


        # ====================================================
        # FINAL BAR
        # ====================================================

        if final_state is None:

            if (
                fr_state
                in (
                    "NO FACE",
                    "MULTIPLE FACES",
                )
            ):
                display_final = "POSITION ONE FACE"
            else:
                display_final = "ANALYZING..."

            final_color = YELLOW

        else:
            display_final = final_state
            final_color = state_color(
                final_state
            )


        cv2.rectangle(
            canvas,
            (
                0,
                VIDEO_H,
            ),
            (
                CANVAS_W,
                CANVAS_H,
            ),
            final_color,
            -1,
        )

        if decision_time_sec is not None:
            final_text = (
                f"FINAL: {display_final}  "
                f"({decision_time_sec:.2f} s)"
            )
        else:
            final_text = (
                f"FINAL: {display_final}"
            )

        draw_text(
            canvas,
            final_text,
            (
                30,
                VIDEO_H + 50,
            ),
            WHITE,
            1.05,
            3,
        )


        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(
            WINDOW_NAME,
            canvas,
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if (
            key == ord("q")
            or key == 27
        ):
            break

        if key == ord("r"):
            reset_session()

        if key == ord("f"):

            fullscreen = not fullscreen

            cv2.setWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_FULLSCREEN,
                (
                    cv2.WINDOW_FULLSCREEN
                    if fullscreen
                    else cv2.WINDOW_NORMAL
                ),
            )

            if not fullscreen:
                cv2.resizeWindow(
                    WINDOW_NAME,
                    CANVAS_W,
                    CANVAS_H,
                )

finally:
    fas_stop_event.set()

    purge_fas_queue()

    try:
        fas_queue.put_nowait(
            None
        )
    except queue.Full:
        pass

    fas_thread.join(
        timeout=1.0
    )

    cap.release()
    cv2.destroyAllWindows()
