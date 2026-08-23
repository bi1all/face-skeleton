import pytest
import numpy as np
import cv2

from face_skeleton import to_pixels, z_range, draw_dots

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

def test_draw_dots(mocker):
    # Mock cv2.circle to intercept calls and verify arguments
    mock_circle = mocker.patch('cv2.circle')

    canvas = np.zeros((200, 200, 3), dtype=np.uint8)
    pts = [
        (10, 20, 0.0),   # z = z_min
        (30, 40, 1.0),   # z = z_max
        (50, 60, 0.5)    # z = mid
    ]
    z_min = 0.0
    z_max = 1.0

    draw_dots(canvas, pts, z_min, z_max)

    assert mock_circle.call_count == 3

    # For pt 1 (10, 20, 0.0): t = 0
    # brightness = int(255 * (1.0 - 0)) = 255
    # radius = max(1, int(3 * (1.0 - 0))) = 3
    args1, kwargs1 = mock_circle.call_args_list[0]
    assert args1[0] is canvas
    assert args1[1] == (10, 20)
    assert args1[2] == 3
    assert args1[3] == (255, 255, 255)
    assert args1[4] == -1
    assert args1[5] == cv2.LINE_AA

    # For pt 2 (30, 40, 1.0): t = 1.0 (approx)
    # brightness = int(255 * (1.0 - 0.75)) = int(255 * 0.25) = 63
    # radius = max(1, int(3 * (1.0 - 1.0))) = max(1, 0) = 1
    args2, kwargs2 = mock_circle.call_args_list[1]
    assert args2[0] is canvas
    assert args2[1] == (30, 40)
    assert args2[2] == 1
    assert args2[3] == (63, 63, 63)
    assert args2[4] == -1
    assert args2[5] == cv2.LINE_AA

    # For pt 3 (50, 60, 0.5): t = 0.5 (approx)
    # brightness = int(255 * (1.0 - 0.5 * 0.75)) = int(255 * 0.625) = 159
    # radius = max(1, int(3 * (1.0 - 0.5))) = max(1, int(1.5)) = 1
    args3, kwargs3 = mock_circle.call_args_list[2]
    assert args3[0] is canvas
    assert args3[1] == (50, 60)
    assert args3[2] == 1
    assert args3[3] == (159, 159, 159)
    assert args3[4] == -1
    assert args3[5] == cv2.LINE_AA

def test_draw_dots_empty(mocker):
    mock_circle = mocker.patch('cv2.circle')
    canvas = np.zeros((200, 200, 3), dtype=np.uint8)
    draw_dots(canvas, [], 0.0, 1.0)
    mock_circle.assert_not_called()
