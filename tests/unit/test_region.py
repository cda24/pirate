import pytest
import numpy as np

from pirate.region import ROI


@pytest.fixture
def test_image():
    nx, ny = 10, 10
    _m = np.ones((nx, ny))

    _m[: nx // 2, : ny // 2] = 5
    _m[nx // 2 :, : ny // 2] = 20
    _m[nx // 2 :, ny // 2 :] = 100
    return _m


def test_rectangular_measurement(test_image):
    test_measurements = [
        ROI(x=7, y=2, h=2, w=2, idx=1),
        ROI(x=2, y=2, h=2, w=2, idx=5),
        ROI(x=2, y=7, h=2, w=2, idx=20),
        ROI(x=7, y=7, h=2, w=2, idx=100),
    ]
    valid = [1, 5, 20, 100]
    for v, M in zip(valid, test_measurements):
        assert v == M.mean(test_image)
        assert v == M.min(test_image)
        assert v == M.max(test_image)


def test_circular_measurement(test_image):
    test_measurements = [
        ROI(x=7, y=2, r=2, idx=1, kind="circle"),
        ROI(x=2, y=2, r=2, idx=5, kind="circle"),
        ROI(x=2, y=7, r=2, idx=20, kind="circle"),
        ROI(x=7, y=7, r=2, idx=100, kind="circle"),
    ]
    valid = [1, 5, 20, 100]
    for v, M in zip(valid, test_measurements):
        assert v == M.mean(test_image)
        assert v == M.min(test_image)
        assert v == M.max(test_image)


def test_poly_measurement(test_image):
    vertices = np.array([[1, 1], [1, 4], [4, 3], [3, 1]])
    test_measurements = [
        ROI(vertices=vertices + [4, 0], idx=1, kind="poly"),
        ROI(vertices=vertices, idx=5, kind="poly"),
        ROI(vertices=vertices + [0, 4], idx=20, kind="poly"),
        ROI(vertices=vertices + [4, 4], idx=100, kind="poly"),
    ]
    valid = [1, 5, 20, 100]
    for v, M in zip(valid, test_measurements):
        assert v == M.mean(test_image)
        assert v == M.min(test_image)
        assert v == M.max(test_image)
