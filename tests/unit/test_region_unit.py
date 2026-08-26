import numpy as np
import pytest

from pirate.region import ROI


@pytest.fixture
def test_image():
    nx, ny = 10, 10
    _m = np.ones((nx, ny))

    _m[: nx // 2, : ny // 2] = 5
    _m[nx // 2 :, : ny // 2] = 20
    _m[nx // 2 :, ny // 2 :] = 100
    return _m


@pytest.fixture
def test_image_large():
    nx, ny = 100, 100
    _m = np.ones((nx, ny))

    _m[: nx // 2, : ny // 2] = 5
    _m[nx // 2 :, : ny // 2] = 20
    _m[nx // 2 :, ny // 2 :] = 100
    return _m


def test_rectangular_measurement(test_image):
    test_measurements = [
        ROI(x=7, y=2, h=2, w=2, idx=1, kind="rectangle"),
        ROI(x=2, y=2, h=2, w=2, idx=5, kind="rectangle"),
        ROI(x=2, y=7, h=2, w=2, idx=20, kind="rectangle"),
        ROI(x=7, y=7, h=2, w=2, idx=100, kind="rectangle"),
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


def test_line_measurement(test_image_large):
    test_measurements = [
        ROI(idx=0, xo=25, yo=25, xp=75, yp=25, width=5, kind="lineout"),
        ROI(idx=1, xo=25, yo=25, xp=25, yp=75, width=5, kind="lineout"),
        ROI(idx=2, xo=25, yo=75, xp=75, yp=75, width=5, kind="lineout"),
        ROI(idx=3, xo=75, yo=25, xp=75, yp=75, width=5, kind="lineout"),
    ]

    valid_mean = [3.0, 12.8, 60.0, 52.48]
    valid_max = [5, 20, 100, 100]
    valid_min = [1, 5, 20, 1]

    for ve, va, vi, M in zip(valid_mean, valid_max, valid_min, test_measurements):
        assert ve == M.mean(test_image_large)
        assert vi == M.min(test_image_large)
        assert va == M.max(test_image_large)


# def test_image_size_persistence(test_image, test_image_large):

#     M1 = ROI(x=7, y=2, h=2, w=2, idx=1, kind="rectangle")
#     M2 = ROI(x=7, y=2, h=2, w=2, idx=1, kind="rectangle")
#     M3 = ROI(x=7, y=2, h=2, w=2, idx=1, kind="rectangle", image_size=test_image_large)

#     assert M1.image_size == M2.image_size
#     assert M1.image_size == test_image.shape
#     assert M1.image_size != M3.image_size
#     assert M2.image_size != M3.image_size
