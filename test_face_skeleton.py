import pytest
from face_skeleton import z_range

def test_z_range_standard():
    pts = [(10, 20, 5), (30, 40, 15), (50, 60, 2)]
    assert z_range(pts) == (2, 15)

def test_z_range_single_element():
    pts = [(10, 20, 5)]
    assert z_range(pts) == (5, 5)

def test_z_range_negative_values():
    pts = [(10, 20, -5), (30, 40, -15), (50, 60, -2)]
    assert z_range(pts) == (-15, -2)

def test_z_range_mixed_values():
    pts = [(10, 20, -5), (30, 40, 15), (50, 60, 0)]
    assert z_range(pts) == (-5, 15)

def test_z_range_empty_list():
    pts = []
    with pytest.raises(ValueError):
        z_range(pts)
