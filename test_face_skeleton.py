import pytest
from face_skeleton import SmoothedLandmark, LandmarkSmoother, to_pixels, z_range

class MockLandmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

def test_smoothed_landmark_initialization():
    lm = SmoothedLandmark(1.0, 2.0, 3.0)
    assert lm.x == 1.0
    assert lm.y == 2.0
    assert lm.z == 3.0

def test_smoothed_landmark_slots():
    lm = SmoothedLandmark(1.0, 2.0, 3.0)
    with pytest.raises(AttributeError):
        lm.w = 4.0

def test_landmark_smoother_initialization():
    smoother = LandmarkSmoother()
    assert smoother.alpha == 0.5
    assert smoother.smoothed is None

    smoother_custom = LandmarkSmoother(alpha=0.8)
    assert smoother_custom.alpha == 0.8
    assert smoother_custom.smoothed is None

def test_landmark_smoother_update():
    smoother = LandmarkSmoother(alpha=0.5)

    # First update: initializes self.smoothed directly
    landmarks1 = [MockLandmark(1.0, 2.0, 3.0), MockLandmark(4.0, 5.0, 6.0)]
    smoothed1 = smoother.update(landmarks1)

    assert len(smoothed1) == 2
    assert smoother.smoothed == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert smoothed1[0].x == 1.0
    assert smoothed1[0].y == 2.0
    assert smoothed1[0].z == 3.0

    # Second update: applies exponential smoothing
    # new_val = alpha * new + (1 - alpha) * old
    # For index 0: 0.5 * 3.0 + 0.5 * 1.0 = 2.0
    landmarks2 = [MockLandmark(3.0, 4.0, 5.0), MockLandmark(6.0, 7.0, 8.0)]
    smoothed2 = smoother.update(landmarks2)

    assert smoother.smoothed == [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]
    assert smoothed2[0].x == 2.0
    assert smoothed2[0].y == 3.0
    assert smoothed2[0].z == 4.0

def test_landmark_smoother_length_change():
    smoother = LandmarkSmoother(alpha=0.5)

    landmarks1 = [MockLandmark(1.0, 2.0, 3.0)]
    smoother.update(landmarks1)

    # Change length, should reset
    landmarks2 = [MockLandmark(3.0, 4.0, 5.0), MockLandmark(6.0, 7.0, 8.0)]
    smoothed2 = smoother.update(landmarks2)

    assert len(smoother.smoothed) == 2
    assert smoother.smoothed == [[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]

def test_to_pixels():
    landmarks = [MockLandmark(0.2, 0.4, 0.1)]
    # x = int((1.0 - 0.2) * 100) = 80
    # y = int(0.4 * 200) = 80
    # z = 0.1
    w = 100
    h = 200
    pixels = to_pixels(landmarks, w, h)

    assert len(pixels) == 1
    assert pixels[0] == (80, 80, 0.1)

def test_z_range():
    pts = [(10, 20, 0.5), (30, 40, -0.2), (50, 60, 1.5)]
    z_min, z_max = z_range(pts)

    assert z_min == -0.2
    assert z_max == 1.5
