from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.ndimage import rotate as _rotate

type Shape = tuple[int, int]  # (rows, cols) == (H, W)


class ROI:
    current_image_size: Shape | None = None

    def __new__(
        cls,
        kind: Literal[
            "polygon",
            "rectangle",
            "rectangular",
            "circle",
            "circular",
            "poly",
            "ellipse",
            "line",
            "lineout",
        ] = "polygon",
        idx: int | None = None,
        image_size: Shape | np.ndarray | None = None,
        **kwargs,
    ):
        if kind is None:
            if image_size is not None:
                cls.current_image_size = cls._as_shape(image_size)
            return None

        kind_lower = kind.lower()

        image_size = (
            cls._as_shape(image_size)
            if image_size is not None
            else cls.current_image_size
        )
        kwargs.setdefault("image_size", image_size)

        match kind_lower:
            case k if k in ["rectangle", "rectangular"]:
                return cls._create_rectangle(idx, **kwargs)

            case k if k in ["circle", "circular"]:
                return cls._create_circle(idx, **kwargs)

            case k if k in ["ellipse"]:
                return cls._create_ellipse(idx, **kwargs)

            case k if k in ["poly", "polygon"]:
                return cls._create_poly(idx, **kwargs)

            case k if k in ["line", "lineout"]:
                return cls._create_lineout(idx, **kwargs)

            case _:
                raise ValueError(
                    f"Unknown ROI kind: '{kind}'. "
                    f"Must be 'rectangle', 'circular', or 'polygon'"
                )

    @staticmethod
    def _as_shape(shape_or_array: Shape | np.ndarray | None) -> Shape | None:
        if shape_or_array is None:
            return None
        if isinstance(shape_or_array, np.ndarray):
            return shape_or_array.shape[:2]
        return tuple(shape_or_array)

    @staticmethod
    def _create_rectangle(idx: int | None = None, **kwargs):
        """Create a RectangularROI."""
        required = {"x", "y", "h", "w"}
        provided = set(kwargs.keys())

        if not required.issubset(provided):
            missing = required - provided
            raise ValueError(
                f"RectangularROI requires parameters: {', '.join(sorted(required))}. "
                f"Missing: {', '.join(sorted(missing))}"
            )

        return RectROI(
            x=kwargs["x"],
            y=kwargs["y"],
            h=kwargs["h"],
            w=kwargs["w"],
            idx=idx,
            image_size=kwargs["image_size"],
        )

    @staticmethod
    def _create_circle(idx: int | None = None, **kwargs):
        """Create a CircularROI."""
        required = {"x", "y", "r"}
        provided = set(kwargs.keys())

        if not required.issubset(provided):
            missing = required - provided
            raise ValueError(
                f"CircularROI requires parameters: {', '.join(sorted(required))}. "
                f"Missing: {', '.join(sorted(missing))}"
            )
        kwargs.setdefault("resolution", 100)
        return CircleROI(
            x=kwargs["x"],
            y=kwargs["y"],
            r=kwargs["r"],
            resolution=kwargs["resolution"],
            idx=idx,
            image_size=kwargs["image_size"],
        )

    @staticmethod
    def _create_ellipse(idx: int | None = None, **kwargs):
        """Create a CircularROI."""
        required = {"x", "y", "ra", "rb"}
        provided = set(kwargs.keys())

        if not required.issubset(provided):
            missing = required - provided
            raise ValueError(
                f"CircularROI requires parameters: {', '.join(sorted(required))}. "
                f"Missing: {', '.join(sorted(missing))}"
            )
        kwargs.setdefault("resolution", 100)
        return EllipseROI(
            x=kwargs["x"],
            y=kwargs["y"],
            ra=kwargs["ra"],
            rb=kwargs["rb"],
            resolution=kwargs["resolution"],
            idx=idx,
            image_size=kwargs["image_size"],
        )

    @staticmethod
    def _create_lineout(idx: int | None = None, **kwargs):
        """Create a Lineout."""
        required = {"xo", "yo", "xp", "yp"}
        provided = set(kwargs.keys())

        if not required.issubset(provided):
            missing = required - provided
            raise ValueError(
                f"LineoutROI requires parameters: {', '.join(sorted(required))}. "
                f"Missing: {', '.join(sorted(missing))}"
            )

        kwargs.setdefault("width", 1)

        return LineROI(
            xo=kwargs["xo"],
            yo=kwargs["yo"],
            xp=kwargs["xp"],
            yp=kwargs["yp"],
            width=kwargs["width"],
            idx=idx,
            image_size=kwargs["image_size"],
        )

    @staticmethod
    def _create_poly(idx: int | None = None, **kwargs):
        if "vertices" not in kwargs:
            raise ValueError("PolyROI requires 'vertices' parameter as list or array")
        extra_kwargs = {k: v for k, v in kwargs.items() if k != "vertices"}
        return PolyROI(vertices=kwargs["vertices"], idx=idx, **extra_kwargs)


class ROI_BASE(ABC):
    """Abstract base class for a single region of interest.

    Subclasses implement only `bound` and `mask`; a idx/index identifies
    each instance (e.g. for bookkeeping when you have many of these).

    This is the layer that makes `CompositeROI` possible: a union/
    difference/intersection of two regions generally isn't expressible as
    a single vertex list, only as a combination of two masks — so it
    subclasses `ROI` directly rather than `PolyROI`. Any future ROI that's
    mask-only (freehand brush, segmentation output, ...) belongs here too.
    """

    def __init__(self, idx, image_size: Shape | np.ndarray | None = None):
        self.idx = idx
        self.image_size = self._as_shape(image_size) if image_size is not None else None
        self._cached_mask = None

    def _cached_mask_for(self, shape: Shape):
        if self._cached_mask is not None and self._cached_mask.shape == shape:
            return self._cached_mask
        return None

    def _cache_mask(self, shape: Shape, mask: np.ndarray):
        if self.image_size == shape:
            self._cached_mask = mask
        return mask

    @property
    @abstractmethod
    def bound(self) -> tuple[int, int, int, int]:
        """(row_min, row_max, col_min, col_max) — outer bounding rectangle.

        Pure geometry: derived only from the ROI's own definition, never
        from an array. Can extend past an actual array's edges — clip
        against a real shape with `_clipped_bound` before slicing.
        """
        raise NotImplementedError

    @abstractmethod
    def mask(self, shape_or_array: Shape | np.ndarray) -> np.ndarray:
        """Boolean array, `shape_or_array`'s size, True inside the region."""
        raise NotImplementedError

    # ---- region processes ----

    def isolate(self, array: np.ndarray) -> np.ndarray:
        """ndarray, `array` size, product of mask * array"""
        return self.mask(array) * array

    def omit(self, array: np.ndarray) -> np.ndarray:
        """ndarray, `array` size, product of !mask * array"""
        return (self.mask(array) == 0) * array

    def crop(self, array: np.ndarray, fill=np.nan) -> np.ndarray:
        """ndarrray, `bound` extent size, invokes isolate then limits area"""
        h, w = array.shape[:2]
        r0, r1, c0, c1 = self.bound
        r0, r1 = max(r0, 0), min(r1, h)
        c0, c1 = max(c0, 0), min(c1, w)
        sub = array[r0:r1, c0:c1].copy()
        m = self.mask(array)[r0:r1, c0:c1]
        sub[~m] = fill
        return sub

    # ---- composition: additive / subtractive / intersection ----
    def __add__(self, other: ROI_BASE) -> CompositeROI:
        return CompositeROI(self, other, "union")

    def __sub__(self, other: ROI_BASE) -> CompositeROI:
        return CompositeROI(self, other, "difference")

    def __and__(self, other: ROI_BASE) -> CompositeROI:
        return CompositeROI(self, other, "intersection")

    # ---- small shared helpers ----
    @staticmethod
    def _as_shape(shape_or_array: Shape | np.ndarray) -> Shape:
        if isinstance(shape_or_array, np.ndarray):
            return shape_or_array.shape[:2]
        return tuple(shape_or_array)

    def _clipped_bound(self, shape: Shape) -> tuple[int, int, int, int]:
        """`.bound`, clamped to fit inside an array of `shape`."""
        h, w = shape
        r0, r1, c0, c1 = self.bound
        return max(r0, 0), min(r1, h), max(c0, 0), min(c1, w)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.idx!r})"

    # ---- Apply functions ----
    def apply(self, image: np.ndarray, func: Callable = np.array, **kwargs):
        """Apply functions only ever apply to the cropped region
        Functions should be nan-compatuble
        """
        kwargs.setdefault("fill", np.nan)
        extra_kwargs = {k: v for k, v in kwargs.items() if k != "fill"}
        return func(self.crop(image, fill=kwargs["fill"]), **extra_kwargs)

    ### 0D
    # Primary scalar production functions
    # All set to be nan safe and propagate the axis flag to apply.
    def mean(
        self, image: np.ndarray, axis: int | tuple | None = None
    ) -> float | np.ndarray:
        return self.apply(image, np.nanmean, axis=axis)

    def std(
        self, image: np.ndarray, axis: int | tuple | None = None
    ) -> float | np.ndarray:
        return self.apply(image, np.nanstd, axis=axis)

    def sum(
        self, image: np.ndarray, axis: int | tuple | None = None
    ) -> float | np.ndarray:
        return self.apply(image, np.nansum, axis=axis)

    def median(
        self, image: np.ndarray, axis: int | tuple | None = None
    ) -> float | np.ndarray:
        return self.apply(image, np.nanmedian, axis=axis)

    def min(
        self, image: np.ndarray, axis: int | tuple | None = None
    ) -> float | np.ndarray:
        return self.apply(image, np.nanmin, axis=axis)

    def max(
        self, image: np.ndarray, axis: int | tuple | None = None
    ) -> float | np.ndarray:
        return self.apply(image, np.nanmax, axis=axis)

    ### 1D
    # At present these are handled by declaring axis=(...) in the 0D functions
    # unclear if I need to add specifics at this point

    ### 2D
    def rebin(self, image: np.ndarray, kernal: int = 3) -> np.ndarray:

        image = self.apply(image)

        if image.ndim != 2:
            raise ValueError("im must be a 2D array")
        if kernal <= 0:
            raise ValueError("kernal must be > 0")

        h, w = image.shape
        h2, w2 = h // kernal, w // kernal
        if h2 == 0 or w2 == 0:
            raise ValueError("image is too small for the requested kernal")

        # Drop any leftover rows/columns so the output is exactly (H//k, W//k)
        image = image[: h2 * kernal, : w2 * kernal]

        return np.mean(np.reshape(image, shape=(h2, kernal, w2, kernal)), axis=(1, 3))

    # ---- Plot ----
    def plot(self, ax=None, **kwargs):
        """
        Generate coordinates for plotting polygonal ROI.
        **kwargs are for matplotlib.pyplot kwargs
        """
        # Close the polygon by appending first vertex at end
        x_coords = self.vertices[:, 0]
        y_coords = self.vertices[:, 1]

        if ax is None:
            return x_coords, y_coords
        else:
            ax.plot(x_coords, y_coords, **kwargs)


class PolyROI(ROI_BASE):
    """Polygon ROI: an ordered list of (x, y) vertices with an idx."""

    def __init__(self, vertices: Sequence[tuple[float, float]], idx, image_size=None):
        super().__init__(idx, image_size=image_size)
        self.vertices = np.asarray(vertices, dtype=float)
        self._close_vertices()

        if self.image_size is not None:
            self.mask(self.image_size)

    def _close_vertices(self):
        if (self.vertices[0, 0] != self.vertices[-1, 0]) or (
            self.vertices[0, 1] != self.vertices[-1, 1]
        ):
            self.vertices = np.vstack((self.vertices, self.vertices[0, :]))

    @property
    def bound(self):
        x, y = self.vertices[:, 0], self.vertices[:, 1]
        return (
            int(np.floor(y.min())),
            int(np.ceil(y.max())) + 1,
            int(np.floor(x.min())),
            int(np.ceil(x.max())) + 1,
        )

    def mask(self, shape_or_array):
        shape = self._as_shape(shape_or_array)
        cached = self._cached_mask_for(shape)
        if cached is not None:
            return cached
        h, w = shape
        yy, xx = np.mgrid[0:h, 0:w]
        points = np.column_stack((xx.ravel(), yy.ravel()))
        inside = MplPath(self.vertices).contains_points(points)
        return inside.reshape(h, w)


class CompositeROI(ROI_BASE):
    """Boolean combination of two ROIs: union / difference / intersection.

    Being an ROI itself, composites chain: `(a + b) - c` works, since each
    intermediate result is just another object with `.bound` and `.mask`.
    Left/right can be any ROI — PolyROI, another CompositeROI, or a future
    mask-only ROI — the op only relies on the shared interface.
    """

    _MASK_OPS = {
        "union": lambda a, b: a | b,
        "difference": lambda a, b: a & ~b,
        "intersection": lambda a, b: a & b,
    }

    def __init__(
        self, left: ROI_BASE, right: ROI_BASE, op: str, idx: str | None = None
    ):
        if op not in self._MASK_OPS:
            raise ValueError(f"op must be one of {list(self._MASK_OPS)}")
        symbol = {"union": "+", "difference": "-", "intersection": "&"}[op]
        super().__init__(idx or f"({left.idx} {symbol} {right.idx})")
        self.left, self.right, self.op = left, right, op

        # reset vertices for latter calls
        self.vertices = np.vstack((self.left.vertices, self.right.vertices))

    @property
    def bound(self):
        r0a, r1a, c0a, c1a = self.left.bound
        if self.op == "difference":
            return self.left.bound  # subtracting never grows the box
        r0b, r1b, c0b, c1b = self.right.bound
        if self.op == "union":
            return (min(r0a, r0b), max(r1a, r1b), min(c0a, c0b), max(c1a, c1b))
        return (
            max(r0a, r0b),
            min(r1a, r1b),
            max(c0a, c0b),
            min(c1a, c1b),
        )  # intersection

    def mask(self, shape_or_array):
        shape = self._as_shape(shape_or_array)
        cached = self._cached_mask_for(shape)
        if cached is not None:
            return cached
        return self._MASK_OPS[self.op](self.left.mask(shape), self.right.mask(shape))


#### Friendly ROIs for simple geometry generation


class LineROI(PolyROI):
    """
    Special instance of Poly for generating rectangles
    By default acts from centre out but can be generated from top-left corner with flag kind='corner'

    Requires x,y,h,w to be defined
    """

    def __init__(
        self,
        xo: float,
        yo: float,
        xp: float,
        yp: float,
        width: float = 1,
        idx: int | str | None = None,
        image_size=None,
    ):
        self.xo, self.yo = float(xo), float(yo)
        self.xp, self.yp = float(xp), float(yp)
        self.width = float(width)

        vertices = self._calculate_vertices()
        super().__init__(vertices, idx, image_size=image_size)

    def _calculate_vertices(self) -> np.ndarray:
        start = np.array([self.xo, self.yo])
        end = np.array([self.xp, self.yp])
        direction = end - start
        length = np.linalg.norm(direction)

        if length == 0:
            raise ValueError("Line endpoints must be different")
        if self.width <= 0:
            raise ValueError("width must be greater than zero")

        self.angle = np.arctan2(direction[1], direction[0])
        normal = np.array([-np.sin(self.angle), np.cos(self.angle)]) * (self.width / 2)

        return np.array(
            [
                start - normal,
                end - normal,
                end + normal,
                start + normal,
                start - normal,
            ]
        )

    def rotate(self, angle: float, degrees: bool = False) -> None:
        """Rotate the line around its midpoint and recalculate its corners."""
        if degrees:
            angle = np.deg2rad(angle)

        centre = np.array([(self.xo + self.xp) / 2, (self.yo + self.yp) / 2])
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )

        start = centre + rotation @ (np.array([self.xo, self.yo]) - centre)
        end = centre + rotation @ (np.array([self.xp, self.yp]) - centre)

        self.xo, self.yo = start
        self.xp, self.yp = end
        self.vertices = self._calculate_vertices()

    def mean(
        self, image: np.ndarray, axis: int | tuple | None = None
    ) -> float | np.ndarray:
        if axis is None:
            super().mean(image=image)
        else:
            print("in sub mean ")
            cropped = self.crop(image, fill=-1)
            rotated = _rotate(
                cropped,
                angle=-np.degrees(self.angle),
                reshape=False,
                mode="constant",
                cval=-1,
            )

            def swap_negatives(arr: np.ndarray):
                arr = arr.copy()
                arr[arr <= 0] = np.nan
                return arr

            return np.nanmean(swap_negatives(rotated), axis=axis)

    @property
    def bound(self):
        x, y = self.vertices[:, 0], self.vertices[:, 1]
        return (
            int(np.floor(y.min())) + 1,
            int(np.ceil(y.max())) + 1,
            int(np.floor(x.min())),
            int(np.ceil(x.max())),
        )


class RectROI(PolyROI):
    """
    Special instance of Poly for generating rectangles
    By default acts from centre out but can be generated from top-left corner with flag kind='corner'

    Requires x,y,h,w to be defined
    """

    def __init__(
        self,
        x: int,
        y: int,
        h: int,
        w: int,
        idx: int | str | None = None,
        kind: Literal["centre", "center", "origin", "corner"] = "centre",
        image_size=None,
    ):
        self.x = x
        self.y = y
        self.h = h
        self.w = w

        match kind:
            case k if k in ["centre", "center", "origin"]:
                x_min = self.x - self.w / 2
                x_max = self.x + self.w / 2
                y_min = self.y - self.h / 2
                y_max = self.y + self.h / 2

            case k if k in ["corner"]:
                x_min = self.x
                x_max = self.x + self.w
                y_min = self.y
                y_max = self.y + self.h

            case _:
                raise ValueError(
                    f"Unknown ROI kind: '{kind}'. Must be 'centre' or 'corner'"
                )

        vertices = np.array(
            [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
                [x_min, y_min],
            ]
        )

        super().__init__(vertices, idx, image_size=image_size)

    @property
    def bound(self):
        x, y = self.vertices[:, 0], self.vertices[:, 1]
        return (
            int(np.floor(y.min())) + 1,
            int(np.ceil(y.max())) + 1,
            int(np.floor(x.min())),
            int(np.ceil(x.max())),
        )


class EllipseROI(PolyROI):
    """
    Special instance of Poly for generating Circles

    Requires x,y,r to be defined

    """

    def __init__(
        self,
        x: int,
        y: int,
        ra: int,
        rb: int,
        idx: int | str | None = None,
        resolution: int = 100,
        image_size=None,
    ):
        self.x = x
        self.y = y
        self.ra = ra
        self.rb = rb

        theta = np.linspace(0, 2 * np.pi, resolution)
        x_coords = self.x + self.ra * np.cos(theta)
        y_coords = self.y + self.rb * np.sin(theta)

        vertices = np.array([x_coords, y_coords]).T
        super().__init__(vertices, idx, image_size=image_size)


class CircleROI(EllipseROI):
    """
    Special instance of Ellipse for generating Circles
    """

    def __init__(
        self,
        x: int,
        y: int,
        r: int,
        idx: int | str | None = None,
        resolution: int = 100,
        image_size=None,
    ):
        super().__init__(x, y, r, r, idx, resolution, image_size=image_size)
