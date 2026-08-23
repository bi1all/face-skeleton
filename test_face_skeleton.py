import pytest

from face_skeleton import to_pixels, z_range, SmoothedLandmark, LandmarkSmoother

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

def test_z_range_happy_path():
    pts = [
        (10, 20, 0.5),
        (30, 40, -1.0),
        (50, 60, 2.5)
    ]
    min_z, max_z = z_range(pts)
    assert min_z == -1.0
    assert max_z == 2.5

def test_z_range_empty():
    min_z, max_z = z_range([])
    assert min_z == 0.0
    assert max_z == 0.0

def test_z_range_single_element():
    pts = [(10, 20, 3.14)]
    min_z, max_z = z_range(pts)
    assert min_z == 3.14
    assert max_z == 3.14

def test_smoothed_landmark_init():
    lm = SmoothedLandmark(1.0, 2.0, 3.0)
    assert lm.x == 1.0
    assert lm.y == 2.0
    assert lm.z == 3.0

def test_landmark_smoother_initial_update():
    smoother = LandmarkSmoother(alpha=0.5)
    assert smoother.smoothed is None

    landmarks = [MockLandmark(1.0, 2.0, 3.0), MockLandmark(4.0, 5.0, 6.0)]
    result = smoother.update(landmarks)

    assert len(result) == 2
    assert result[0].x == 1.0
    assert result[0].y == 2.0
    assert result[0].z == 3.0
    assert result[1].x == 4.0
    assert result[1].y == 5.0
    assert result[1].z == 6.0

def test_landmark_smoother_subsequent_update():
    smoother = LandmarkSmoother(alpha=0.5)

    # First update sets initial state
    landmarks1 = [MockLandmark(1.0, 2.0, 3.0)]
    smoother.update(landmarks1)

    # Second update applies EMA
    landmarks2 = [MockLandmark(3.0, 4.0, 5.0)]
    result = smoother.update(landmarks2)

    # Expected: alpha * new_val + (1 - alpha) * old_val
    # Expected x = 0.5 * 3.0 + 0.5 * 1.0 = 1.5 + 0.5 = 2.0
    assert len(result) == 1
    assert result[0].x == 2.0
    assert result[0].y == 3.0
    assert result[0].z == 4.0

def test_landmark_smoother_length_mismatch():
    smoother = LandmarkSmoother(alpha=0.5)

    landmarks1 = [MockLandmark(1.0, 2.0, 3.0)]
    smoother.update(landmarks1)

    # Different number of landmarks should reset smoothing
    landmarks2 = [MockLandmark(2.0, 2.0, 2.0), MockLandmark(3.0, 3.0, 3.0)]
    result = smoother.update(landmarks2)

    assert len(result) == 2
    # Should be exact values from landmarks2, not smoothed
    assert result[0].x == 2.0
    assert result[1].x == 3.0
