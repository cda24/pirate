## Timing Tests
import time

import numpy as np

from pirate.region import ROI


def test_rectangle_timing(N=100, regions=100):
    NX = N
    IMAGE = np.random.randn(NX, NX)

    IMAGE[NX // 4 : NX // 2, NX // 4 : NX // 2] = 1
    IMAGE[NX // 2 : 3 * NX // 4, NX // 4 : NX // 2] = 2
    IMAGE[NX // 4 : NX // 2, NX // 2 : 3 * NX // 4] = 3
    IMAGE[NX // 2 : 3 * NX // 4, NX // 2 : 3 * NX // 4] = 4

    collection = []
    for r in range(regions):
        x = np.random.randint(10, N - 5, 1)[0]
        y = np.random.randint(10, N - 5, 1)[0]
        h = np.random.randint(1, y // 3, 1)[0]
        w = np.random.randint(1, x // 3, 1)[0]

        collection.append(ROI(x=x, y=y, h=h, w=w, idx=r, kind="rectangle"))

    catches = 0
    start_time = time.time()
    for c in collection:
        try:
            c.mean(IMAGE)

        except IndexError:
            catches += 1
            # print(f'IndexError, {c.idx}, \n coords: {c.x,c.y,c.h,c.w}')

        except ValueError:
            print(f"ValueError, {c.idx}, \n coords: {c.x, c.y, c.h, c.w}")

    end_time = time.time()

    print(
        f"Total Time for ({N, N}) images and {regions - catches} Measurements: {end_time - start_time}s "
    )
    print(
        f"Per measurement for ({N, N}) images: {(end_time - start_time) / (regions - catches)}s "
    )
    average_time = (end_time - start_time) / (regions - catches)

    assert average_time < 0.1


def test_circular_timing(N=100, regions=100):
    NX = N
    IMAGE = np.random.randn(NX, NX)

    IMAGE[NX // 4 : NX // 2, NX // 4 : NX // 2] = 1
    IMAGE[NX // 2 : 3 * NX // 4, NX // 4 : NX // 2] = 2
    IMAGE[NX // 4 : NX // 2, NX // 2 : 3 * NX // 4] = 3
    IMAGE[NX // 2 : 3 * NX // 4, NX // 2 : 3 * NX // 4] = 4

    collection = []
    for r in range(regions):
        x = np.random.randint(10, N - 5, 1)[0]
        y = np.random.randint(10, N - 5, 1)[0]
        r = np.random.randint(1, min(y // 3, x // 3), 1)[0]

        collection.append(ROI(x=x, y=y, r=r, idx=r, kind="Circular"))

    catches = 0
    start_time = time.time()
    for c in collection:
        try:
            c.mean(IMAGE)

        except IndexError:
            catches += 1
            # print(f'IndexError, {c.idx}, \n coords: {c.x,c.y,c.h,c.w}')

        except ValueError:
            print(f"ValueError, {c.idx}, \n coords: {c.x, c.y, c.h, c.w}")

    end_time = time.time()

    print(
        f"Total Time for ({N, N}) images and {regions - catches} Measurements: {end_time - start_time}s "
    )
    print(
        f"Per measurement for ({N, N}) images: {(end_time - start_time) / (regions - catches)}s "
    )
    average_time = (end_time - start_time) / (regions - catches)

    assert average_time < 0.1


def test_poly_timing(N=100, regions=100, p=12):
    NX = N
    IMAGE = np.random.randn(NX, NX)

    IMAGE[NX // 4 : NX // 2, NX // 4 : NX // 2] = 1
    IMAGE[NX // 2 : 3 * NX // 4, NX // 4 : NX // 2] = 2
    IMAGE[NX // 4 : NX // 2, NX // 2 : 3 * NX // 4] = 3
    IMAGE[NX // 2 : 3 * NX // 4, NX // 2 : 3 * NX // 4] = 4

    collection = []
    for r in range(regions):
        x = np.random.randint(10, N - 5, p)
        y = np.random.randint(10, N - 5, p)

        collection.append(ROI(vertices=np.vstack((x, y)).T, idx=r, kind="Poly"))

    catches = 0
    start_time = time.time()
    for c in collection:
        try:
            c.mean(IMAGE)

        except IndexError:
            catches += 1
            # print(f'IndexError, {c.idx}, \n coords: {c.x,c.y,c.h,c.w}')

        except ValueError:
            print(f"ValueError, {c.idx}, \n coords: {c.x, c.y, c.h, c.w}")

    end_time = time.time()

    print(
        f"Total Time for ({N, N}) images and {regions - catches} Measurements: {end_time - start_time}s "
    )
    print(
        f"Per measurement for ({N, N}) images: {(end_time - start_time) / (regions - catches)}s "
    )
    average_time = (end_time - start_time) / (regions - catches)
    assert average_time < 0.1
