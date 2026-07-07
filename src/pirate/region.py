import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Literal


class ROI:
    def __new__(
        cls,
        kind: Literal[
            "rectangle", "rectangular", "circle", "circular", "poly", "polygon"
        ] = "rectangle",
        idx: int | None = None,
        **kwargs,
    ):
        kind_lower = kind.lower()

        match kind_lower:
            case None:
                pass
            case k if k in ["rectangle", "rectangular"]:
                return cls._create_rectangle(idx, **kwargs)

            case k if k in ["circle", "circular"]:
                return cls._create_circle(idx, **kwargs)

            case k if k in ["poly", "polygon"]:
                return cls._create_polygon(idx, **kwargs)

            case _:
                raise ValueError(
                    f"Unknown ROI kind: '{kind}'. "
                    f"Must be 'rectangle', 'circular', or 'polygon'"
                )

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

        return RectangularROI(
            x=kwargs["x"], y=kwargs["y"], h=kwargs["h"], w=kwargs["w"], idx=idx
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

        return CircularROI(x=kwargs["x"], y=kwargs["y"], r=kwargs["r"], idx=idx)

    @staticmethod
    def _create_polygon(idx: int | None = None, **kwargs) -> "PolygonalROI":
        """Create a PolygonalROI."""
        if "vertices" not in kwargs:
            raise ValueError(
                "PolygonalROI requires 'vertices' parameter as list or array"
            )

        vertices = kwargs["vertices"]

        # Handle additional kwargs
        extra_kwargs = {k: v for k, v in kwargs.items() if k != "vertices"}

        return PolygonalROI(vertices=vertices, idx=idx, **extra_kwargs)


@dataclass
class ROI_base(ABC):
    """
    Abstract base class for Region of Interest (ROI) objects.

    Defines common interface for different ROI shapes used in image processing.

    Attributes
    ----------
    idx : int, optional
        Index/identifier for this ROI
    """

    idx: int | None = None

    @abstractmethod
    def plot_coords(self) -> tuple:
        """
        Generate coordinates for plotting ROI.

        Returns
        -------
        tuple
            (x_coords, y_coords) suitable for plt.plot()
        """
        pass

    @abstractmethod
    def image_coords(self) -> tuple:
        """
        Generate pixel indices for extracting ROI from an image array.

        Returns
        -------
        tuple
            Indexing structure suitable for image[...]
        """
        pass

    @abstractmethod
    def contains_point(self, x: float, y: float) -> bool:
        """
        Check if a point is contained within this ROI.

        Parameters
        ----------
        x : float
            X coordinate
        y : float
            Y coordinate

        Returns
        -------
        bool
            True if point is in ROI
        """
        pass

    def _apply_func(self, im, func=np.mean) -> float:
        """
        Applies defined function to a supplied image within ROI region.

        Parameters
        ----------
        im : np.ndarray
            Image array to process
        func : callable
            Function to apply (default: np.mean)

        Returns
        -------
        float
            Output from func
        """
        y_slice, x_slice = self.image_coords()
        return func(im[y_slice, x_slice])

    def mean(self, im):
        """Calculate mean value in ROI."""
        return self._apply_func(im, np.mean)

    def max(self, im):
        """Calculate maximum value in ROI."""
        return self._apply_func(im, np.max)

    def min(self, im):
        """Calculate minimum value in ROI."""
        return self._apply_func(im, np.min)

    def std(self, im):
        """Calculate standard deviation in ROI."""
        return self._apply_func(im, np.std)


@dataclass
class PolygonalROI(ROI_base):
    """
    Polygonal ROI defined by vertices.

    Attributes
    ----------
    vertices : np.ndarray
        Array of shape (N, 2) containing (x, y) coordinates of polygon vertices
    idx : int, optional
        Index/identifier for this ROI
    """

    vertices: np.ndarray | list = None
    idx: int = None

    def __post_init__(self):
        """Ensure vertices is a numpy array."""
        self.vertices = np.asarray(self.vertices)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 2:
            raise ValueError("vertices must be array of shape (N, 2)")

    def plot_coords(self) -> tuple:
        """
        Generate coordinates for plotting polygonal ROI.

        Returns
        -------
        tuple
            (x_coords, y_coords) forming closed polygon
        """
        # Close the polygon by appending first vertex at end
        x_coords = self.vertices[:, 0]
        y_coords = self.vertices[:, 1]
        if (self.vertices[0, 0] != self.vertices[-1, 0]) or (
            self.vertices[0, 1] != self.vertices[-1, 1]
        ):
            x_coords = np.append(x_coords, self.vertices[0, 0])
            y_coords = np.append(y_coords, self.vertices[0, 1])

        return x_coords, y_coords

    def image_coords(self, shape: tuple[int, int] | None = None) -> tuple:
        """
        Generate pixel coordinates for extracting polygonal ROI from image.
        Uses rasterization to create a mask.

        Returns
        -------
        tuple
            (y_mask, x_mask) boolean arrays for polygonal region
        """
        from matplotlib.path import Path

        if shape is None:
            # Get bounding box
            x_min, y_min = self.vertices.min(axis=0).astype(int)
            x_max, y_max = self.vertices.max(axis=0).astype(int)

            # Create coordinate grids
            y_grid, x_grid = np.mgrid[y_min + 1 : y_max, x_min + 1 : x_max]
        else:
            y_grid, x_grid = np.mgrid[: shape[1], : shape[0]]

        points = np.column_stack((x_grid.ravel(), y_grid.ravel()))

        # Create polygon path and check containment
        path = Path(self.vertices)
        mask = path.contains_points(points).reshape(x_grid.shape)

        return mask

    def region(self, im) -> np.ndarray:
        mask = self.image_coords()
        x_min, y_min = self.vertices.min(axis=0).astype(int)

        x_max = x_min + mask.shape[1]
        y_max = y_min + mask.shape[0]

        return im[y_min:y_max, x_min:x_max]

    def _apply_func(self, im, func=np.mean) -> float:
        """Override _apply_func for polygonal ROI using mask."""
        mask = self.image_coords()
        region = self.region(im)
        return func(region[mask])

    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is within polygonal ROI."""
        from matplotlib.path import Path

        path = Path(self.vertices)
        return path.contains_point((x, y))


@dataclass
class RectangularROI(PolygonalROI):
    x: int = 0
    y: int = 0
    h: int = 0
    w: int = 0
    idx: int = 0
    vertices: np.ndarray | list = field(init=False)

    def __post_init__(self):
        x_min = self.x - self.w / 2
        x_max = self.x + self.w / 2
        y_min = self.y - self.h / 2
        y_max = self.y + self.h / 2

        self.vertices = np.array(
            [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
                [x_min, y_min],
            ]
        )


@dataclass
class CircularROI(PolygonalROI):
    x: int = 0
    y: int = 0
    r: int = 0
    idx: int = 0
    vertices: np.ndarray | list = field(init=False)

    def __post_init__(self):
        theta = np.linspace(0, 2 * np.pi, 100)
        x_coords = self.x + self.r * np.cos(theta)
        y_coords = self.y + self.r * np.sin(theta)

        self.vertices = np.array([x_coords, y_coords]).T


if __name__ == "__main__":
    pass
