import pytest
from unittest.mock import patch, call
import math
import face_skeleton

# Import the classes and functions to test
from face_skeleton import (
    SmoothedLandmark,
    LandmarkSmoother,
    to_pixels,
    z_range,
    download_model,
    MODEL_PATH,
    MODEL_URL
)


class DummyLandmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


def test_smoothed_landmark():
    lm = SmoothedLandmark(0.1, 0.2, 0.3)
    assert lm.x == 0.1
    assert lm.y == 0.2
    assert lm.z == 0.3


def test_landmark_smoother_initialization():
    smoother = LandmarkSmoother(alpha=0.6)
    assert smoother.alpha == 0.6
    assert smoother.smoothed is None


def test_landmark_smoother_update():
    smoother = LandmarkSmoother(alpha=0.5)

    # First update should just copy the coordinates
    landmarks = [DummyLandmark(1.0, 2.0, 3.0), DummyLandmark(0.5, 0.5, 0.5)]
    res1 = smoother.update(landmarks)

    assert len(res1) == 2
    assert res1[0].x == 1.0
    assert res1[0].y == 2.0
    assert res1[0].z == 3.0

    assert smoother.smoothed == [[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]]

    # Second update should blend
    landmarks2 = [DummyLandmark(2.0, 4.0, 6.0), DummyLandmark(0.0, 0.0, 0.0)]
    res2 = smoother.update(landmarks2)

    # For index 0: alpha=0.5. new_x = 0.5 * 2.0 + 0.5 * 1.0 = 1.5
    assert math.isclose(res2[0].x, 1.5)
    assert math.isclose(res2[0].y, 3.0)
    assert math.isclose(res2[0].z, 4.5)

    assert math.isclose(res2[1].x, 0.25)
    assert math.isclose(res2[1].y, 0.25)
    assert math.isclose(res2[1].z, 0.25)


def test_to_pixels():
    landmarks = [
        DummyLandmark(0.0, 0.0, 0.1),
        DummyLandmark(1.0, 1.0, 0.5),
        DummyLandmark(0.5, 0.25, -0.1)
    ]
    w = 100
    h = 200

    res = to_pixels(landmarks, w, h)

    # x = int((1.0 - lm.x) * w)
    # y = int(lm.y * h)
    # z = lm.z
    assert res == [
        (100, 0, 0.1),
        (0, 200, 0.5),
        (50, 50, -0.1)
    ]


def test_z_range():
    pts = [
        (10, 10, -0.5),
        (20, 20, 1.5),
        (30, 30, 0.0)
    ]
    z_min, z_max = z_range(pts)
    assert z_min == -0.5
    assert z_max == 1.5


@patch("face_skeleton.os.path.exists")
@patch("face_skeleton.urllib.request.urlretrieve")
def test_download_model_not_exists(mock_urlretrieve, mock_exists):
    mock_exists.return_value = False

    download_model()

    mock_exists.assert_called_once_with(MODEL_PATH)
    mock_urlretrieve.assert_called_once_with(MODEL_URL, MODEL_PATH)


@patch("face_skeleton.os.path.exists")
@patch("face_skeleton.urllib.request.urlretrieve")
def test_download_model_exists(mock_urlretrieve, mock_exists):
    mock_exists.return_value = True

    download_model()

    mock_exists.assert_called_once_with(MODEL_PATH)
    mock_urlretrieve.assert_not_called()
