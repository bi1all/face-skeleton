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

from face_skeleton import LandmarkSmoother

def test_landmark_smoother_initial_state():
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

def test_landmark_smoother_subsequent_updates():
    smoother = LandmarkSmoother(alpha=0.5)
    landmarks1 = [MockLandmark(1.0, 2.0, 3.0)]
    smoother.update(landmarks1)

    landmarks2 = [MockLandmark(3.0, 6.0, 9.0)]
    result = smoother.update(landmarks2)

    # new_val = 0.5 * new + 0.5 * old
    # x: 0.5*3 + 0.5*1 = 1.5 + 0.5 = 2.0
    # y: 0.5*6 + 0.5*2 = 3.0 + 1.0 = 4.0
    # z: 0.5*9 + 0.5*3 = 4.5 + 1.5 = 6.0
    assert len(result) == 1
    assert result[0].x == 2.0
    assert result[0].y == 4.0
    assert result[0].z == 6.0

def test_landmark_smoother_length_change():
    smoother = LandmarkSmoother(alpha=0.5)
    landmarks1 = [MockLandmark(1.0, 2.0, 3.0)]
    smoother.update(landmarks1)

    landmarks2 = [MockLandmark(10.0, 20.0, 30.0), MockLandmark(40.0, 50.0, 60.0)]
    result = smoother.update(landmarks2)

    # Should reset entirely because length changed
    assert len(result) == 2
    assert result[0].x == 10.0
    assert result[1].x == 40.0

def test_landmark_smoother_alpha_one():
    smoother = LandmarkSmoother(alpha=1.0)
    smoother.update([MockLandmark(1.0, 2.0, 3.0)])

    result = smoother.update([MockLandmark(5.0, 6.0, 7.0)])

    # Should adopt new entirely
    assert result[0].x == 5.0
    assert result[0].y == 6.0
    assert result[0].z == 7.0

def test_landmark_smoother_alpha_zero():
    smoother = LandmarkSmoother(alpha=0.0)
    smoother.update([MockLandmark(1.0, 2.0, 3.0)])

    result = smoother.update([MockLandmark(5.0, 6.0, 7.0)])

    # Should retain old entirely
    assert result[0].x == 1.0
    assert result[0].y == 2.0
    assert result[0].z == 3.0

def test_landmark_smoother_empty_landmarks():
    smoother = LandmarkSmoother(alpha=0.5)
    result1 = smoother.update([])
    assert result1 == []

    result2 = smoother.update([])
    assert result2 == []
