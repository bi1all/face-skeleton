import hashlib
import sys
import types
from unittest.mock import mock_open

import numpy as np
import pytest
import os


def _install_test_stubs():
    if "cv2" not in sys.modules:
        cv2_stub = types.ModuleType("cv2")
        cv2_stub.LINE_AA = 16
        cv2_stub.polylines = lambda *args, **kwargs: None
        sys.modules["cv2"] = cv2_stub

    if "mediapipe" not in sys.modules:
        mp_stub = types.ModuleType("mediapipe")
        mp_stub.__file__ = "/tmp/mediapipe/__init__.py"
        mp_stub.tasks = types.SimpleNamespace(
            BaseOptions=object,
            vision=types.SimpleNamespace(
                FaceLandmarker=object,
                FaceLandmarkerOptions=object,
                RunningMode=types.SimpleNamespace(VIDEO="VIDEO"),
            ),
        )
        sys.modules["mediapipe"] = mp_stub


_install_test_stubs()

import cv2
import face_skeleton
from face_skeleton import to_pixels, z_range, SmoothedLandmark


def test_to_pixels_happy_path():
    landmarks = [
        SmoothedLandmark(0.0, 0.0, 0.0),
        SmoothedLandmark(0.5, 0.5, 0.5),
        SmoothedLandmark(1.0, 1.0, 1.0)
    ]
    w, h = 100, 200
    expected = [
        (99, 0, 0.0),    # (1 - 0) * (100 - 1) = 99
        (49, 99, 0.5),   # (1 - 0.5) * 99 = 49.5 -> 49, 0.5 * 199 = 99.5 -> 99
        (0, 199, 1.0)    # (1 - 1) * (100 - 1) = 0, 1 * (200 - 1) = 199
    ]
    assert to_pixels(landmarks, w, h) == expected


def test_to_pixels_empty_landmarks():
    assert to_pixels([], 100, 200) == []


def test_to_pixels_zero_dimensions():
    landmarks = [
        SmoothedLandmark(0.5, 0.5, 0.5)
    ]
    w, h = 0, 0
    expected = [
        (0, 0, 0.5)
    ]
    assert to_pixels(landmarks, w, h) == expected


def test_to_pixels_negative_dimensions():
    landmarks = [
        SmoothedLandmark(0.5, 0.5, 0.5)
    ]
    w, h = -100, -200
    expected = [
        (-50, -100, 0.5)
    ]
    assert to_pixels(landmarks, w, h) == expected


def test_to_pixels_negative_coordinates():
    landmarks = [
        SmoothedLandmark(-0.5, -0.5, -0.5)
    ]
    w, h = 100, 200
    expected = [
        (148, -99, -0.5) # (1 - (-0.5)) * 99 = 148.5 -> 148, -0.5 * 199 = -99.5 -> -99
    ]
    assert to_pixels(landmarks, w, h) == expected


def test_to_pixels_type_casting():
    landmarks = [
        SmoothedLandmark(0.123, 0.456, 0.789)
    ]
    w, h = 100, 200

    # int((1 - 0.123) * 99) = int(0.877 * 99) = int(86.823) = 86
    # int(0.456 * 199) = int(90.744) = 90
    expected = [
        (86, 90, 0.789)
    ]

    res = to_pixels(landmarks, w, h)
    assert res == expected
    # verify that the x and y are indeed ints
    assert isinstance(res[0][0], int)
    assert isinstance(res[0][1], int)


def test_download_model_success(mocker):
    import urllib.request
    download_model = face_skeleton.download_model
    model_path = face_skeleton.MODEL_PATH
    model_url = face_skeleton.MODEL_URL

    m_exists = mocker.patch("face_skeleton.os.path.exists", return_value=False)
    m_urlretrieve = mocker.patch("urllib.request.urlretrieve")

    mock_file_content = b"fake_model_data"
    mock_hash = hashlib.sha256(mock_file_content).hexdigest()
    mocker.patch("face_skeleton.EXPECTED_MODEL_HASH", mock_hash)

    m_open = mocker.patch("builtins.open", mock_open(read_data=mock_file_content))

    download_model()

    m_exists.assert_called()
    m_urlretrieve.assert_called_once_with(model_url, model_path)
    m_open.assert_called_once_with(model_path, "rb")


def test_download_model_hash_mismatch(mocker):
    import urllib.request
    download_model = face_skeleton.download_model
    model_path = face_skeleton.MODEL_PATH

    mocker.patch("face_skeleton.os.path.exists", return_value=False)
    mocker.patch("urllib.request.urlretrieve")

    mock_file_content = b"corrupted_model_data"
    m_open = mocker.patch("builtins.open", mock_open(read_data=mock_file_content))
    m_remove = mocker.patch("face_skeleton.os.remove")

    with pytest.raises(RuntimeError, match="Hash mismatch for downloaded model"):
        download_model()

    m_open.assert_called_once_with(model_path, "rb")
    m_remove.assert_called_once_with(model_path)


def test_draw_connections_polylines(mocker):
    from face_skeleton import draw_connections

    mock_polylines = mocker.patch("cv2.polylines")

    mock_canvas = mocker.Mock()

    pts = [
        (10, 20, 0),
        (30, 40, 0),
        (50, 60, 0),
        (70, 80, 0)
    ]
    pts_arr = np.array([
        [10, 20],
        [30, 40],
        [50, 60],
        [70, 80]
    ], dtype=np.int32)
    connections = np.array([(0, 1), (1, 2), (0, 3)], dtype=np.int32)
    color = (255, 255, 255)
    thickness = 2

    draw_connections(mock_canvas, pts, connections, color, thickness, pts_arr)

    mock_polylines.assert_called_once()

    args, _ = mock_polylines.call_args

    assert args[0] is mock_canvas

    segments = args[1]
    expected_segments = np.array([
        [[10, 20], [30, 40]],
        [[30, 40], [50, 60]],
        [[10, 20], [70, 80]]
    ], dtype=np.int32)

    np.testing.assert_array_equal(segments, expected_segments)

    assert args[2] is False
    assert args[3] == color
    assert args[4] == thickness
    assert args[5] == cv2.LINE_AA


def test_z_range_empty():
    assert z_range([]) == (0.0, 0.0)


def test_z_range_single_point():
    pts = [(10, 20, 5.5)]
    assert z_range(pts) == (5.5, 5.5)


def test_z_range_multiple_points():
    pts = [
        (10, 20, 5.5),
        (30, 40, -2.1),
        (50, 60, 8.9),
        (70, 80, 0.0)
    ]
    assert z_range(pts) == (-2.1, 8.9)


def test_z_range_generator():
    pts = ((x, x + 1, z) for x, z in enumerate([5.5, -2.1, 8.9, 0.0]))
    assert z_range(pts) == (-2.1, 8.9)


def test_landmark_smoother_initial_update():
    from face_skeleton import LandmarkSmoother
    smoother = LandmarkSmoother(alpha=0.5)
    landmarks = [
        SmoothedLandmark(1.0, 2.0, 3.0),
        SmoothedLandmark(4.0, 5.0, 6.0)
    ]

    smoothed = smoother.update(landmarks)

    assert len(smoothed) == 2
    assert smoothed[0].x == 1.0
    assert smoothed[0].y == 2.0
    assert smoothed[0].z == 3.0
    assert smoothed[1].x == 4.0
    assert smoothed[1].y == 5.0
    assert smoothed[1].z == 6.0


def test_landmark_smoother_subsequent_update():
    from face_skeleton import LandmarkSmoother
    smoother = LandmarkSmoother(alpha=0.5)
    landmarks1 = [
        SmoothedLandmark(10.0, 20.0, 30.0)
    ]
    smoother.update(landmarks1)

    landmarks2 = [
        SmoothedLandmark(20.0, 30.0, 40.0)
    ]
    smoothed = smoother.update(landmarks2)

    assert len(smoothed) == 1
    # alpha = 0.5
    # new_val = 0.5 * lm + 0.5 * old_val
    # new_x = 0.5 * 20.0 + 0.5 * 10.0 = 15.0
    assert smoothed[0].x == 15.0
    assert smoothed[0].y == 25.0
    assert smoothed[0].z == 35.0


def test_landmark_smoother_length_change():
    from face_skeleton import LandmarkSmoother
    smoother = LandmarkSmoother(alpha=0.5)
    landmarks1 = [
        SmoothedLandmark(1.0, 2.0, 3.0)
    ]
    smoother.update(landmarks1)

    landmarks2 = [
        SmoothedLandmark(10.0, 20.0, 30.0),
        SmoothedLandmark(40.0, 50.0, 60.0)
    ]
    # Length changed, so it should reset and copy exactly
    smoothed = smoother.update(landmarks2)

    assert len(smoothed) == 2
    assert smoothed[0].x == 10.0
    assert smoothed[0].y == 20.0
    assert smoothed[0].z == 30.0
    assert smoothed[1].x == 40.0
    assert smoothed[1].y == 50.0
    assert smoothed[1].z == 60.0

def test__convert_connections_empty():
    from face_skeleton import _convert_connections
    res = _convert_connections([])
    assert isinstance(res, np.ndarray)
    assert res.shape == (0, 2)
    assert res.dtype == np.int32


def test__convert_connections_valid():
    from face_skeleton import _convert_connections
    class MockConnection:
        def __init__(self, start, end):
            self.start = start
            self.end = end

    connections = [
        MockConnection(0, 1),
        MockConnection(1, 2),
        MockConnection(2, 3)
    ]

    res = _convert_connections(connections)
    assert isinstance(res, np.ndarray)
    assert res.shape == (3, 2)
    assert res.dtype == np.int32
    np.testing.assert_array_equal(res, np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32))
