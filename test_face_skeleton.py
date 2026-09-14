import pytest

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
        (100, 0, 0.0),   # (1 - 0) * 100 = 100
        (50, 100, 0.5),  # (1 - 0.5) * 100 = 50, 0.5 * 200 = 100
        (0, 200, 1.0)    # (1 - 1) * 100 = 0, 1 * 200 = 200
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
        (150, -100, -0.5) # (1 - (-0.5)) * 100 = 150, -0.5 * 200 = -100
    ]
    assert to_pixels(landmarks, w, h) == expected

def test_to_pixels_type_casting():
    landmarks = [
        MockLandmark(0.123, 0.456, 0.789)
    ]
    w, h = 100, 200

    # int((1 - 0.123) * 100) = int(0.877 * 100) = int(87.7) = 87
    # int(0.456 * 200) = int(91.2) = 91
    expected = [
        (87, 91, 0.789)
    ]

    res = to_pixels(landmarks, w, h)
    assert res == expected
    # verify that the x and y are indeed ints
    assert isinstance(res[0][0], int)
    assert isinstance(res[0][1], int)

import os
import hashlib
from unittest.mock import patch, mock_open

def test_download_model_success(mocker):
    from face_skeleton import download_model, EXPECTED_MODEL_HASH
    mocker.patch('os.path.exists', return_value=False)
    mocker.patch('urllib.request.urlretrieve')

    # Mock file contents to match expected hash
    mock_file_content = b"fake_model_data"
    mock_hash = hashlib.sha256(mock_file_content).hexdigest()

    # We patch EXPECTED_MODEL_HASH just for this test to match our fake content
    mocker.patch('face_skeleton.EXPECTED_MODEL_HASH', mock_hash)

    m_open = mocker.patch('builtins.open', mock_open(read_data=mock_file_content))

    # This shouldn't raise any exception
    download_model()
    m_open.assert_called_once_with('face_landmarker.task', 'rb')


def test_download_model_hash_mismatch(mocker):
    from face_skeleton import download_model
    mocker.patch('os.path.exists', return_value=False)
    mocker.patch('urllib.request.urlretrieve')

    mock_file_content = b"corrupted_model_data"

    m_open = mocker.patch('builtins.open', mock_open(read_data=mock_file_content))
    m_remove = mocker.patch('os.remove')

    # Should raise RuntimeError because the hash of 'corrupted_model_data' won't match EXPECTED_MODEL_HASH
    with pytest.raises(RuntimeError, match="Hash mismatch for downloaded model"):
        download_model()

    m_open.assert_called_once_with('face_landmarker.task', 'rb')
    m_remove.assert_called_once_with('face_landmarker.task')
