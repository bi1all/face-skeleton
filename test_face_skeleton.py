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

def test_draw_connections_polylines(mocker):
    from face_skeleton import draw_connections
    import numpy as np
    import cv2

    mock_polylines = mocker.patch('cv2.polylines')

    # We create a mock canvas to prevent numpy equality errors in call comparisons
    mock_canvas = mocker.Mock()

    pts = [
        (10, 20, 0.1),
        (30, 40, 0.2),
        (50, 60, 0.3)
    ]
    connections = [(0, 1), (1, 2), (0, 3)] # (0, 3) is invalid
    color = (255, 255, 255)
    thickness = 2

    draw_connections(mock_canvas, pts, connections, color, thickness)

    # Verify polylines was called exactly once
    mock_polylines.assert_called_once()

    # Extract arguments from the call
    args, kwargs = mock_polylines.call_args

    # Canvas should be the mock object
    assert args[0] is mock_canvas

    # The segments array should contain points for valid connections (0,1) and (1,2)
    segments = args[1]
    expected_segments = np.array([
        [[10, 20], [30, 40]],
        [[30, 40], [50, 60]]
    ], dtype=np.int32)

    np.testing.assert_array_equal(segments, expected_segments)

    # Verify other arguments
    assert args[2] is False # isClosed
    assert args[3] == color # color
    assert args[4] == thickness # thickness
    assert args[5] == cv2.LINE_AA # lineType
