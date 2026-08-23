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

def test_main_happy_path(mocker):
    # Mock download_model to avoid network calls
    mock_download = mocker.patch("face_skeleton.download_model")

    # Mock init_camera to return a dummy MagicMock that we can configure
    mock_init_camera = mocker.patch("face_skeleton.init_camera")
    mock_cap = mocker.MagicMock()
    # Configure read() to return (True, dummy_frame) when not paused,
    # but let's just say it always returns a frame
    import numpy as np
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, mock_frame)
    mock_init_camera.return_value = mock_cap

    # Mock FaceLandmarker
    mock_landmarker_cls = mocker.patch("face_skeleton.FaceLandmarker")
    mock_landmarker_ctx = mocker.MagicMock()
    # The with block returns something, we need mock_landmarker_cls.create_from_options().return_value.__enter__.return_value = ...
    mock_landmarker_inst = mocker.MagicMock()
    mock_landmarker_ctx.__enter__.return_value = mock_landmarker_inst
    mock_landmarker_cls.create_from_options.return_value = mock_landmarker_ctx

    # Mock process_frame
    mock_process = mocker.patch("face_skeleton.process_frame")
    # Return a dummy result that has face_landmarks = [] to avoid needing a real result object
    mock_result = mocker.MagicMock()
    mock_result.face_landmarks = []
    mock_process.return_value = mock_result

    # Mock save_landmarks so we can verify it's called
    mock_save = mocker.patch("face_skeleton.save_landmarks")

    # Mock cv2 UI functions
    mocker.patch("cv2.imshow")
    mocker.patch("cv2.destroyAllWindows")

    # Mock waitKey to simulate a sequence of key presses:
    # 1. -1 (no key, just loop)
    # 2. ord(' ') (pause)
    # 3. -1 (loop while paused)
    # 4. ord(' ') (unpause)
    # 5. ord('s') (save)
    # 6. 27 (ESC to quit)
    mock_waitkey = mocker.patch("cv2.waitKey")
    mock_waitkey.side_effect = [-1, ord(' '), -1, ord(' '), ord('s'), 27]

    # Import and run main
    import face_skeleton
    face_skeleton.main()

    # Assertions
    mock_download.assert_called_once()
    mock_init_camera.assert_called_once_with(face_skeleton.CAMERA_INDEX)
    # Cap should have been read multiple times (when unpaused)
    assert mock_cap.read.call_count >= 1
    # Cap should be released at the end
    mock_cap.release.assert_called_once()
    # save_landmarks should have been called once
    mock_save.assert_called_once()
    # verify waitkey was called 6 times
    assert mock_waitkey.call_count == 6

def test_main_no_camera(mocker):
    # Mock download_model
    mock_download = mocker.patch("face_skeleton.download_model")

    # Mock init_camera to return None (e.g. camera not found)
    mock_init_camera = mocker.patch("face_skeleton.init_camera")
    mock_init_camera.return_value = None

    import face_skeleton
    face_skeleton.main()

    # Assertions
    mock_download.assert_called_once()
    mock_init_camera.assert_called_once_with(face_skeleton.CAMERA_INDEX)
    # The rest of the function shouldn't execute
    # Let's mock something from later in main to ensure it wasn't called
    # (actually we don't need to mock it, it just won't be called, but we can't assert on it unless mocked,
    # however, we know it returns early if cap is None).
