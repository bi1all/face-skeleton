"""
Face Skeleton — Dense Mesh
mediapipe 0.10.35 Tasks API | Python 3.12 | RGB camera 0
"""

import cv2
import numpy as np
import mediapipe as mp
import urllib.request
import os
import hashlib
import time

# ── LOAD CONNECTION CONSTANTS ─────────────────────────────────────────────────
# Securely load connection constants from the Mediapipe Tasks API.

def _convert_connections(connections_list):
    if not connections_list:
        return np.zeros((0, 2), dtype=np.int32)
    return np.array([(conn.start, conn.end) for conn in connections_list], dtype=np.int32)

try:
    from mediapipe.tasks.python.vision import face_landmarker
    fmc = face_landmarker.FaceLandmarksConnections

    FACEMESH_TESSELATION   = _convert_connections(fmc.FACE_LANDMARKS_TESSELATION)
    FACEMESH_FACE_OVAL     = _convert_connections(fmc.FACE_LANDMARKS_FACE_OVAL)
    FACEMESH_LEFT_EYE      = _convert_connections(fmc.FACE_LANDMARKS_LEFT_EYE)
    FACEMESH_RIGHT_EYE     = _convert_connections(fmc.FACE_LANDMARKS_RIGHT_EYE)
    FACEMESH_LEFT_EYEBROW  = _convert_connections(fmc.FACE_LANDMARKS_LEFT_EYEBROW)
    FACEMESH_RIGHT_EYEBROW = _convert_connections(fmc.FACE_LANDMARKS_RIGHT_EYEBROW)
    FACEMESH_LIPS          = _convert_connections(fmc.FACE_LANDMARKS_LIPS)

    # Irises are split in the modern API, combine them
    left_iris = _convert_connections(getattr(fmc, 'FACE_LANDMARKS_LEFT_IRIS', []))
    right_iris = _convert_connections(getattr(fmc, 'FACE_LANDMARKS_RIGHT_IRIS', []))
    FACEMESH_IRISES = np.vstack((left_iris, right_iris)) if left_iris.size and right_iris.size else left_iris if left_iris.size else right_iris

    print("[INFO] Connection constants loaded securely from FaceLandmarksConnections.")
except (ImportError, AttributeError):
    # Hardcoded fallback — contour skeleton, no tesselation fill
    print("[INFO] FaceLandmarksConnections not found — using hardcoded contour skeleton.")
    FACEMESH_TESSELATION   = np.zeros((0, 2), dtype=np.int32)
    FACEMESH_FACE_OVAL = np.array([
        (10,338),(338,297),(297,332),(332,284),(284,251),(251,389),(389,356),
        (356,454),(454,323),(323,361),(361,288),(288,397),(397,365),(365,379),
        (379,378),(378,400),(400,377),(377,152),(152,148),(148,176),(176,149),
        (149,150),(150,136),(136,172),(172,58),(58,132),(132,93),(93,234),
        (234,127),(127,162),(162,21),(21,54),(54,103),(103,67),(67,109),(109,10)
    ], dtype=np.int32)
    FACEMESH_LEFT_EYE = np.array([
        (263,249),(249,390),(390,373),(373,374),(374,380),(380,381),(381,382),
        (382,362),(362,398),(398,384),(384,385),(385,386),(386,387),(387,388),
        (388,466),(466,263)
    ], dtype=np.int32)
    FACEMESH_RIGHT_EYE = np.array([
        (33,7),(7,163),(163,144),(144,145),(145,153),(153,154),(154,155),
        (155,133),(133,173),(173,157),(157,158),(158,159),(159,160),(160,161),
        (161,246),(246,33)
    ], dtype=np.int32)
    FACEMESH_LEFT_EYEBROW = np.array([
        (276,283),(283,282),(282,295),(295,285),(300,293),(293,334),(334,296),(296,336)
    ], dtype=np.int32)
    FACEMESH_RIGHT_EYEBROW = np.array([
        (46,53),(53,52),(52,65),(65,55),(70,63),(63,105),(105,66),(66,107)
    ], dtype=np.int32)
    FACEMESH_LIPS = np.array([
        (61,146),(146,91),(91,181),(181,84),(84,17),(17,314),(314,405),(405,321),
        (321,375),(375,291),(61,185),(185,40),(40,39),(39,37),(37,0),(0,267),
        (267,269),(269,270),(270,409),(409,291),(78,95),(95,88),(88,178),(178,87),
        (87,14),(14,317),(317,402),(402,318),(318,324),(324,308),(78,191),(191,80),
        (80,81),(81,82),(82,13),(13,312),(312,311),(311,310),(310,415),(415,308)
    ], dtype=np.int32)
    FACEMESH_IRISES = np.array([
        (468,469),(469,470),(470,471),(471,472),(472,468),
        (473,474),(474,475),(475,476),(476,477),(477,473)
    ], dtype=np.int32)

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
MODEL_URL    = os.environ.get(
    "MODEL_URL",
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
EXPECTED_MODEL_HASH = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"

# ── COLORS (BGR) ──────────────────────────────────────────────────────────────
C_MESH = (20,  20,  20 )
C_OVAL = (0,   210, 90 )
C_EYE  = (210, 155, 0  )
C_BROW = (0,   130, 255)
C_LIPS = (30,  50,  240)
C_IRIS = (255, 255, 255)

# ── CONNECTION SPECS ──────────────────────────────────────────────────────────
# Define connection specifications once to avoid recreating the list inside the loop
CONNECTION_SPECS = [
    (FACEMESH_TESSELATION,   C_MESH, 1),
    (FACEMESH_FACE_OVAL,     C_OVAL, 2),
    (FACEMESH_LEFT_EYE,      C_EYE,  1),
    (FACEMESH_RIGHT_EYE,     C_EYE,  1),
    (FACEMESH_LEFT_EYEBROW,  C_BROW, 1),
    (FACEMESH_RIGHT_EYEBROW, C_BROW, 1),
    (FACEMESH_LIPS,          C_LIPS, 1),
    (FACEMESH_IRISES,        C_IRIS, 1)
]

# ── HELPERS ───────────────────────────────────────────────────────────────────

class SmoothedLandmark:
    __slots__ = ['x', 'y', 'z']
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class LandmarkSmoother:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.smoothed = None

    def update(self, landmarks):
        if self.smoothed is None or len(self.smoothed) != len(landmarks):
            self.smoothed = [SmoothedLandmark(lm.x, lm.y, lm.z) for lm in landmarks]
        else:
            alpha = self.alpha
            inv_alpha = 1.0 - alpha
            for i, lm in enumerate(landmarks):
                s = self.smoothed[i]
                s.x = alpha * lm.x + inv_alpha * s.x
                s.y = alpha * lm.y + inv_alpha * s.y
                s.z = alpha * lm.z + inv_alpha * s.z

        return self.smoothed

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("[SETUP] Downloading face_landmarker.task (~30 MB) — one time only...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[SETUP] Done.")
    with open(MODEL_PATH, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    if file_hash != EXPECTED_MODEL_HASH:
        os.remove(MODEL_PATH)
        raise RuntimeError("Hash mismatch for downloaded model.")

def to_pixels(landmarks, w, h):
    x_scale = w - 1 if w > 0 else w
    y_scale = h - 1 if h > 0 else h
    return [(int((1.0 - lm.x) * x_scale), int(lm.y * y_scale), lm.z) for lm in landmarks]

def z_range(pts):
    if not pts:
        return 0.0, 0.0
    return min(pt[2] for pt in pts), max(pt[2] for pt in pts)

def draw_connections(canvas, pts, connections, color, thickness=1, pts_arr=None):
    n = len(pts)
    valid_connections = [(a, b) for a, b in connections if a < n and b < n]
    if not valid_connections:
        return
    if pts_arr is None:
        pts_arr = np.array(pts, dtype=np.int32)[:, :2]
    segments = pts_arr[valid_connections]
    cv2.polylines(canvas, segments, False, color, thickness, cv2.LINE_AA)

def draw_dots(canvas, pts, z_min, z_max):
    span = z_max - z_min + 1e-9
    for x, y, z in pts:
        t          = (z - z_min) / span
        brightness = int(255 * (1.0 - t * 0.75))
        radius     = max(1, int(3 * (1.0 - t)))
        cv2.circle(canvas, (x, y), radius,
                   (brightness, brightness, brightness), -1, cv2.LINE_AA)

# ── MAIN ──────────────────────────────────────────────────────────────────────

def setup_landmarker_options():
    return FaceLandmarkerOptions(
        base_options                          = BaseOptions(model_asset_path=MODEL_PATH),
        running_mode                          = VisionRunningMode.VIDEO,
        num_faces                             = 1,
        min_face_detection_confidence         = 0.5,
        min_face_presence_confidence          = 0.5,
        min_tracking_confidence               = 0.5,
        output_face_blendshapes               = False,
        output_facial_transformation_matrixes = False,
    )

def init_camera(camera_index, fps=30):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_index}.")
        return None

    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap

def process_frame(landmarker, frame):
    timestamp_ms = int(time.time() * 1000)

    # ── PRIVACY BARRIER ──────────────────────────────────────────────
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = landmarker.detect_for_video(mp_image, timestamp_ms)
    # ─────────────────────────────────────────────────────────────────
    return result

def render_result(canvas, result, smoother):
    if result and result.face_landmarks:
        for face in result.face_landmarks:
            smoothed_face = smoother.update(face)
            pts    = to_pixels(smoothed_face, CANVAS_W, CANVAS_H)
            zm, zx = z_range(pts)

            pts_arr = np.array(pts, dtype=np.int32)[:, :2]

            for conn, color, thickness in CONNECTION_SPECS:
                draw_connections(canvas, pts, conn, color, thickness, pts_arr)
            draw_dots(canvas, pts, zm, zx)
    else:
        smoother.smoothed = None

def save_landmarks(smoother, filename="face_landmarks.txt"):
    if smoother.smoothed is not None:
        with open(filename, "w") as f:
            f.write("id,x,y,z\n")
            for i, lm in enumerate(smoother.smoothed):
                f.write(f"{i},{lm.x:.6f},{lm.y:.6f},{lm.z:.6f}\n")
        print(f"[SAVED] {filename}")

def run_tracking_loop(cap, options, smoother):
    last_result = None
    t_prev      = time.perf_counter()
    paused      = False
    canvas      = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                result      = process_frame(landmarker, frame)
                last_result = result
            else:
                # If paused, we keep using the `last_result`
                result = last_result
                # We need to simulate time passing for fps calculation, though fps might not make as much sense when paused
                time.sleep(0.01)

            canvas.fill(0)

            render_result(canvas, result, smoother)

            now    = time.perf_counter()
            fps    = 1.0 / (now - t_prev + 1e-9)
            t_prev = now
            cv2.putText(canvas, f"FPS {fps:.1f}", (10, 28),
                        cv2.FONT_HERSHEY_PLAIN, 1.4, (55, 55, 55), 1, cv2.LINE_AA)

            cv2.imshow("Face Skeleton", canvas)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break
            if key == ord(' '):
                paused = not paused
            if key == ord('s'):
                save_landmarks(smoother)

def main():
    download_model()

    options = setup_landmarker_options()

    cap = init_camera(CAMERA_INDEX)
    if cap is None:
        return

    print("[INFO] Running. ESC = quit | S = save landmarks")
    smoother = LandmarkSmoother(alpha=0.5)

    try:
        run_tracking_loop(cap, options, smoother)
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
