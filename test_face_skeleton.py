import pytest
from face_skeleton import LandmarkSmoother

class MockLandmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

def test_landmark_smoother_initialization():
    smoother_default = LandmarkSmoother()
    assert smoother_default.alpha == 0.5
    assert smoother_default.smoothed is None

    smoother_custom = LandmarkSmoother(alpha=0.8)
    assert smoother_custom.alpha == 0.8
    assert smoother_custom.smoothed is None

def test_landmark_smoother_first_update():
    smoother = LandmarkSmoother(alpha=0.5)
    landmarks = [MockLandmark(1.0, 2.0, 3.0), MockLandmark(4.0, 5.0, 6.0)]

    result = smoother.update(landmarks)

    assert smoother.smoothed == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
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
    landmarks1 = [MockLandmark(0.0, 0.0, 0.0)]
    smoother.update(landmarks1)

    # Second update should smooth: new_val = alpha * lm + (1-alpha) * old_val
    # new_x = 0.5 * 10.0 + 0.5 * 0.0 = 5.0
    landmarks2 = [MockLandmark(10.0, 10.0, 10.0)]
    result = smoother.update(landmarks2)

    assert smoother.smoothed == [[5.0, 5.0, 5.0]]
    assert len(result) == 1
    assert result[0].x == 5.0
    assert result[0].y == 5.0
    assert result[0].z == 5.0

def test_landmark_smoother_subsequent_update_different_alpha():
    smoother = LandmarkSmoother(alpha=0.8)

    # First update sets initial state
    landmarks1 = [MockLandmark(0.0, 0.0, 0.0)]
    smoother.update(landmarks1)

    # Second update should smooth: new_val = alpha * lm + (1-alpha) * old_val
    # new_x = 0.8 * 10.0 + 0.2 * 0.0 = 8.0
    landmarks2 = [MockLandmark(10.0, 10.0, 10.0)]
    result = smoother.update(landmarks2)

    assert smoother.smoothed == [[8.0, 8.0, 8.0]]
    assert len(result) == 1
    assert result[0].x == 8.0
    assert result[0].y == 8.0
    assert result[0].z == 8.0

def test_landmark_smoother_length_change():
    smoother = LandmarkSmoother(alpha=0.5)

    # First update sets initial state length 2
    landmarks1 = [MockLandmark(1.0, 2.0, 3.0), MockLandmark(4.0, 5.0, 6.0)]
    smoother.update(landmarks1)
    assert len(smoother.smoothed) == 2

    # Second update with length 1 should reset the state
    landmarks2 = [MockLandmark(10.0, 20.0, 30.0)]
    result = smoother.update(landmarks2)

    assert smoother.smoothed == [[10.0, 20.0, 30.0]]
    assert len(result) == 1
    assert result[0].x == 10.0
    assert result[0].y == 20.0
    assert result[0].z == 30.0
