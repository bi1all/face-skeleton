import pytest
import os

from face_skeleton import to_pixels, z_range, save_landmarks, LandmarkSmoother, SmoothedLandmark

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

def test_save_landmarks(tmp_path):
    filename = tmp_path / "test_face_landmarks.txt"

    # Setup dummy smoother and landmarks
    smoother = LandmarkSmoother()
    smoother.smoothed = [
        SmoothedLandmark(0.1, 0.2, 0.3),
        SmoothedLandmark(0.4, 0.5, 0.6)
    ]

    # Save the landmarks
    save_landmarks(smoother, filename=filename)

    # Verify the file was created and contains the expected CSV content
    assert os.path.exists(filename)

    with open(filename, "r") as f:
        content = f.read()

    expected_content = "id,x,y,z\n0,0.100000,0.200000,0.300000\n1,0.400000,0.500000,0.600000\n"

    assert content == expected_content

    # Clean up the test file
    os.remove(filename)

def test_save_landmarks_no_data():
    filename = "test_empty_landmarks.txt"

    smoother = LandmarkSmoother()
    smoother.smoothed = None

    save_landmarks(smoother, filename=filename)

    assert not os.path.exists(filename)
