import os
import pytest
from unittest.mock import patch
import hashlib
from face_skeleton import download_model, check_model_hash, MODEL_PATH, MODEL_HASH

@pytest.fixture(autouse=True)
def cleanup():
    # Remove file before and after test
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    yield
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

def create_dummy_model(valid=True):
    # If valid, create a file that matches MODEL_HASH (just a dummy for testing)
    # Wait, check_model_hash checks the actual hash. So we should mock hashlib or create a real hash.
    pass

@patch('urllib.request.urlretrieve')
@patch('face_skeleton.check_model_hash')
def test_download_success(mock_check_hash, mock_urlretrieve):
    # Setup
    mock_check_hash.return_value = True

    # Run
    download_model()

    # Assert
    mock_urlretrieve.assert_called_once()
    mock_check_hash.assert_called()

@patch('urllib.request.urlretrieve')
@patch('face_skeleton.check_model_hash')
def test_download_failure(mock_check_hash, mock_urlretrieve):
    # Setup
    mock_check_hash.return_value = False

    # Run
    with pytest.raises(RuntimeError, match="Security Error: Downloaded model hash does not match expected hash!"):
        download_model()

    # Assert
    mock_urlretrieve.assert_called_once()
    mock_check_hash.assert_called()
