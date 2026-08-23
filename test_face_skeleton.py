import pytest
from face_skeleton import z_range

def test_z_range_happy_path():
    pts = [(1, 2, 3), (4, 5, 10), (7, 8, -5)]
    z_min, z_max = z_range(pts)
    assert z_min == -5
    assert z_max == 10

def test_z_range_single_point():
    pts = [(1, 2, 5)]
    z_min, z_max = z_range(pts)
    assert z_min == 5
    assert z_max == 5

def test_z_range_empty_list():
    with pytest.raises(ValueError):
        z_range([])
