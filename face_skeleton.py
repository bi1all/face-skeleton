"""
Face Skeleton — Dense Mesh
mediapipe 0.10.35 Tasks API | Python 3.12 | RGB camera 0
"""

import cv2
import numpy as np
import mediapipe as mp
import importlib.util
import urllib.request
import os
import time

# ── LOAD CONNECTION CONSTANTS ─────────────────────────────────────────────────
# mediapipe 0.10.x broke the normal import path for face_mesh_connections.
# Load the .py file directly from disk using importlib — bypasses module registry.

def _load_connections():
    mp_dir    = os.path.dirname(mp.__file__)
    conn_file = os.path.join(mp_dir, 'python', 'solutions', 'face_mesh_connections.py')
    if os.path.exists(conn_file):
        spec = importlib.util.spec_from_file_location('_fmc', conn_file)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return None

_fmc = _load_connections()

if _fmc:
    FACEMESH_TESSELATION   = _fmc.FACEMESH_TESSELATION
    FACEMESH_FACE_OVAL     = _fmc.FACEMESH_FACE_OVAL
    FACEMESH_LEFT_EYE      = _fmc.FACEMESH_LEFT_EYE
    FACEMESH_RIGHT_EYE     = _fmc.FACEMESH_RIGHT_EYE
    FACEMESH_LEFT_EYEBROW  = _fmc.FACEMESH_LEFT_EYEBROW
    FACEMESH_RIGHT_EYEBROW = _fmc.FACEMESH_RIGHT_EYEBROW
    FACEMESH_LIPS          = _fmc.FACEMESH_LIPS
    FACEMESH_IRISES        = getattr(_fmc, 'FACEMESH_IRISES', frozenset())
    print("[INFO] Connection constants loaded from mediapipe package files.")
else:
    # Hardcoded fallback — contour skeleton, no tesselation fill
    print("[INFO] face_mesh_connections.py not found — using hardcoded contour skeleton.")
    FACEMESH_TESSELATION   = frozenset()
    FACEMESH_FACE_OVAL = frozenset([
        (10,338),(338,297),(297,332),(332,284),(284,251),(251,389),(389,356),
        (356,454),(454,323),(323,361),(361,288),(288,397),(397,365),(365,379),
        (379,378),(378,400),(400,377),(377,152),(152,148),(148,176),(176,149),
        (149,150),(150,136),(136,172),(172,58),(58,132),(132,93),(93,234),
        (234,127),(127,162),(162,21),(21,54),(54,103),(103,67),(67,109),(109,10)
    ])
    FACEMESH_LEFT_EYE = frozenset([
        (263,249),(249,390),(390,373),(373,374),(374,380),(380,381),(381,382),
        (382,362),(362,398),(398,384),(384,385),(385,386),(386,387),(387,388),
        (388,466),(466,263)
    ])
    FACEMESH_RIGHT_EYE = frozenset([
        (33,7),(7,163),(163,144),(144,145),(145,153),(153,154),(154,155),
        (155,133),(133,173),(173,157),(157,158),(158,159),(159,160),(160,161),
        (161,246),(246,33)
    ])
    FACEMESH_LEFT_EYEBROW = frozenset([
        (276,283),(283,282),(282,295),(295,285),(300,293),(293,334),(334,296),(296,336)
    ])
    FACEMESH_RIGHT_EYEBROW = frozenset([
        (46,53),(53,52),(52,65),(65,55),(70,63),(63,105),(105,66),(66,107)
    ])
    FACEMESH_LIPS = frozenset([
        (61,146),(146,91),(91,181),(181,84),(84,17),(17,314),(314,405),(405,321),
        (321,375),(375,291),(61,185),(185,40),(40,39),(39,37),(37,0),(0,267),
        (267,269),(269,270),(270,409),(409,291),(78,95),(95,88),(88,178),(178,87),
        (87,14),(14,317),(317,402),(402,318),(318,324),(324,308),(78,191),(191,80),
        (80,81),(81,82),(82,13),(13,312),(312,311),(311,310),(310,415),(415,308)
    ])
    FACEMESH_IRISES = frozenset([
        (468,469),(469,470),(470,471),(471,472),(472,468),
        (473,474),(474,475),(475,476),(476,477),(477,473)
    ])

# ── Tasks API aliases ─────────────────────────────────────────────────────────
BaseOptions           = mp.tasks.BaseOptions
FaceLandmarker        = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

# ── CONFIG ────────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
CANVAS_W     = 1280
CANVAS_H     = 960
MODEL_PATH   = "face_landmarker.task"
MODEL_URL    = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# ── COLORS (BGR) ──────────────────────────────────────────────────────────────
C_MESH = (20,  20,  20 )
C_OVAL = (0,   210, 90 )
C_EYE  = (210, 155, 0  )
C_BROW = (0,   130, 255)
C_LIPS = (30,  50,  240)
C_IRIS = (255, 255, 255)

# ── HELPERS ───────────────────────────────────────────────────────────────────

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("[SETUP] Downloading face_landmarker.task (~30 MB) — one time only...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[SETUP] Done.")

def to_pixels(landmarks, w, h):
    return [(int((1.0 - lm.x) * w), int(lm.y * h), lm.z) for lm in landmarks]

def z_range(pts):
    zs = [z for _, _, z in pts]
    return min(zs), max(zs)

def draw_connections(canvas, pts, connections, color, thickness=1):
    for a, b in connections:
        if a < len(pts) and b < len(pts):
            cv2.line(canvas,
                     (pts[a][0], pts[a][1]),
                     (pts[b][0], pts[b][1]),
                     color, thickness, cv2.LINE_AA)

def draw_dots(canvas, pts, z_min, z_max):
    span = z_max - z_min + 1e-9
    for x, y, z in pts:
        t          = (z - z_min) / span
        brightness = int(255 * (1.0 - t * 0.75))
        radius     = max(1, int(3 * (1.0 - t)))
        cv2.circle(canvas, (x, y), radius,
                   (brightness, brightness, brightness), -1, cv2.LINE_AA)

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    download_model()

    options = FaceLandmarkerOptions(
        base_options                          = BaseOptions(model_asset_path=MODEL_PATH),
        running_mode                          = VisionRunningMode.VIDEO,
        num_faces                             = 1,
        min_face_detection_confidence         = 0.5,
        min_face_presence_confidence          = 0.5,
        min_tracking_confidence               = 0.5,
        output_face_blendshapes               = False,
        output_facial_transformation_matrixes = False,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera 0.")
        return

    cap.set(cv2.CAP_PROP_FPS, 30)
    print("[INFO] Running. ESC = quit | S = save landmarks")

    last_result = None
    t_prev      = time.perf_counter()

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            timestamp_ms = int(time.time() * 1000)

            # ── PRIVACY BARRIER ──────────────────────────────────────────────
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            del frame, rgb
            result      = landmarker.detect_for_video(mp_image, timestamp_ms)
            last_result = result
            del mp_image
            # ─────────────────────────────────────────────────────────────────

            canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

            if result.face_landmarks:
                for face in result.face_landmarks:
                    pts    = to_pixels(face, CANVAS_W, CANVAS_H)
                    zm, zx = z_range(pts)

                    draw_connections(canvas, pts, FACEMESH_TESSELATION,   C_MESH, 1)
                    draw_connections(canvas, pts, FACEMESH_FACE_OVAL,     C_OVAL, 2)
                    draw_connections(canvas, pts, FACEMESH_LEFT_EYE,      C_EYE,  1)
                    draw_connections(canvas, pts, FACEMESH_RIGHT_EYE,     C_EYE,  1)
                    draw_connections(canvas, pts, FACEMESH_LEFT_EYEBROW,  C_BROW, 1)
                    draw_connections(canvas, pts, FACEMESH_RIGHT_EYEBROW, C_BROW, 1)
                    draw_connections(canvas, pts, FACEMESH_LIPS,          C_LIPS, 1)
                    draw_connections(canvas, pts, FACEMESH_IRISES,        C_IRIS, 1)
                    draw_dots(canvas, pts, zm, zx)

            now    = time.perf_counter()
            fps    = 1.0 / (now - t_prev + 1e-9)
            t_prev = now
            cv2.putText(canvas, f"FPS {fps:.1f}", (10, 28),
                        cv2.FONT_HERSHEY_PLAIN, 1.4, (55, 55, 55), 1, cv2.LINE_AA)

            cv2.imshow("Face Skeleton", canvas)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break
            if key == ord('s') and last_result and last_result.face_landmarks:
                with open("face_landmarks.txt", "w") as f:
                    f.write("id,x,y,z\n")
                    for i, lm in enumerate(last_result.face_landmarks[0]):
                        f.write(f"{i},{lm.x:.6f},{lm.y:.6f},{lm.z:.6f}\n")
                print("[SAVED] face_landmarks.txt")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
