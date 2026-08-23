import pytest

import numpy as np

from face_skeleton import to_pixels, z_range, draw_connections
import face_skeleton

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


def test_draw_connections_happy_path(mocker):
    mock_line = mocker.patch('face_skeleton.cv2.line')
    canvas = mocker.Mock()
    pts = [(10, 20, 0), (30, 40, 1), (50, 60, 2)]
    connections = frozenset([(0, 1), (1, 2)])
    color = (255, 0, 0)
    thickness = 2

    draw_connections(canvas, pts, connections, color, thickness)

    assert mock_line.call_count == 2
    # Verify the calls. Order in frozenset is not guaranteed, so we check if all expected calls are made
    expected_calls = [
        mocker.call(canvas, (10, 20), (30, 40), color, thickness, face_skeleton.cv2.LINE_AA),
        mocker.call(canvas, (30, 40), (50, 60), color, thickness, face_skeleton.cv2.LINE_AA)
    ]
    mock_line.assert_has_calls(expected_calls, any_order=True)

def test_draw_connections_out_of_bounds(mocker):
    mock_line = mocker.patch('face_skeleton.cv2.line')
    canvas = mocker.Mock()
    pts = [(10, 20, 0), (30, 40, 1)]
    # 2 is out of bounds for pts (len 2)
    connections = frozenset([(0, 1), (1, 2), (3, 0)])
    color = (0, 255, 0)

    draw_connections(canvas, pts, connections, color)

    assert mock_line.call_count == 1
    mock_line.assert_called_once_with(
        canvas, (10, 20), (30, 40), color, 1, face_skeleton.cv2.LINE_AA
    )

def test_draw_connections_empty(mocker):
    mock_line = mocker.patch('face_skeleton.cv2.line')
    canvas = mocker.Mock()
    color = (0, 0, 255)

    # Empty points and connections
    draw_connections(canvas, [], frozenset(), color)
    assert mock_line.call_count == 0

    # Empty connections only
    pts = [(10, 20, 0)]
    draw_connections(canvas, pts, frozenset(), color)
    assert mock_line.call_count == 0

    # Empty points, non-empty connections
    connections = frozenset([(0, 1)])
    draw_connections(canvas, [], connections, color)
    assert mock_line.call_count == 0
