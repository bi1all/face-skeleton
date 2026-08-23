import pytest
from face_skeleton import to_pixels

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

from face_skeleton import z_range

def test_z_range_happy_path():
    pts = [(1, 2, 3), (4, 5, 10), (7, 8, -5)]
    z_min, z_max = z_range(pts)
    assert z_min == -5
    assert z_max == 10

def test_z_range_single_point():
    pts = [(1, 2, 5)]
    z_min, z_max = z_range(pts)
    assert z_min == 5
    assert z_max == 5

def test_z_range_empty_list():
    # z_range now explicitly handles empty lists and returns 0.0, 0.0
    z_min, z_max = z_range([])
    assert z_min == 0.0
    assert z_max == 0.0
