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

def test_to_pixels_inversion_logic():
    # specifically verify the 1.0 - lm.x logic
    landmarks = [MockLandmark(0.25, 0.5, 0.0)]
    w, h = 100, 100
    res = to_pixels(landmarks, w, h)
    # x should be int((1.0 - 0.25) * 100) = 75
    assert res[0][0] == 75

def test_to_pixels_scaling():
    # specifically verify the scaling logic for x and y
    landmarks = [MockLandmark(0.1, 0.2, 0.5)]
    w, h = 1000, 2000
    res = to_pixels(landmarks, w, h)
    # x should be int((1.0 - 0.1) * 1000) = 900
    # y should be int(0.2 * 2000) = 400
    assert res[0][0] == 900
    assert res[0][1] == 400
    assert res[0][2] == 0.5 # z is unmodified

def test_to_pixels_boundary_values():
    landmarks = [
        MockLandmark(-1.0, -1.0, -1.0),
        MockLandmark(2.0, 2.0, 2.0)
    ]
    w, h = 100, 100
    res = to_pixels(landmarks, w, h)

    # 1st point: x = (1 - (-1.0)) * 100 = 200, y = -1.0 * 100 = -100
    assert res[0][0] == 200
    assert res[0][1] == -100

    # 2nd point: x = (1 - 2.0) * 100 = -100, y = 2.0 * 100 = 200
    assert res[1][0] == -100
    assert res[1][1] == 200
