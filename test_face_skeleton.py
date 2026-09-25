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
from face_skeleton import to_pixels, z_range


class MockLandmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


def test_to_pixels_happy_path():
    landmarks = [
        MockLandmark(0.0, 0.0, 0.0),
        MockLandmark(0.5, 0.5, 0.5),
        MockLandmark(1.0, 1.0, 1.0)
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
        MockLandmark(0.5, 0.5, 0.5)
    ]
    w, h = 0, 0
    expected = [
        (0, 0, 0.5)
    ]
    assert to_pixels(landmarks, w, h) == expected


def test_to_pixels_negative_dimensions():
    landmarks = [
        MockLandmark(0.5, 0.5, 0.5)
    ]
    w, h = -100, -200
    expected = [
        (-50, -100, 0.5)
    ]
    assert to_pixels(landmarks, w, h) == expected


def test_to_pixels_negative_coordinates():
    landmarks = [
        MockLandmark(-0.5, -0.5, -0.5)
    ]
    w, h = 100, 200
    expected = [
        (148, -99, -0.5) # (1 - (-0.5)) * 99 = 148.5 -> 148, -0.5 * 199 = -99.5 -> -99
    ]
    assert to_pixels(landmarks, w, h) == expected


def test_to_pixels_type_casting():
    landmarks = [
        MockLandmark(0.123, 0.456, 0.789)
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


@pytest.mark.skipif(not hasattr(face_skeleton, "EXPECTED_MODEL_HASH"), reason="model integrity verification is not implemented in this branch")
def test_download_model_success(mocker):
    download_model = face_skeleton.download_model
    model_path = face_skeleton.MODEL_PATH
    model_url = face_skeleton.MODEL_URL

    m_exists = mocker.patch("face_skeleton.os.path.exists", return_value=False)
    m_urlretrieve = mocker.patch("face_skeleton.urllib.request.urlretrieve")

    mock_file_content = b"fake_model_data"
    mock_hash = hashlib.sha256(mock_file_content).hexdigest()
    mocker.patch("face_skeleton.EXPECTED_MODEL_HASH", mock_hash)

    m_open = mocker.patch("builtins.open", mock_open(read_data=mock_file_content))

    download_model()

    m_exists.assert_called()
    m_urlretrieve.assert_called_once_with(model_url, model_path)
    m_open.assert_called_once_with(model_path, "rb")


@pytest.mark.skipif(not hasattr(face_skeleton, "EXPECTED_MODEL_HASH"), reason="model integrity verification is not implemented in this branch")
def test_download_model_hash_mismatch(mocker):
    download_model = face_skeleton.download_model
    model_path = face_skeleton.MODEL_PATH

    mocker.patch("face_skeleton.os.path.exists", return_value=False)
    mocker.patch("face_skeleton.urllib.request.urlretrieve")

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

    pts_arr = np.array([
        [10, 20],
        [30, 40],
        [50, 60],
        [70, 80]
    ], dtype=np.int32)
    connections = np.array([(0, 1), (1, 2), (0, 3)], dtype=np.int32)
    color = (255, 255, 255)
    thickness = 2

    draw_connections(mock_canvas, pts_arr, connections, color, thickness)

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
