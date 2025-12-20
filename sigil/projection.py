"""
Coordinate projection utilities for handling geographic transformations.

This module provides tools for converting between WGS84 (lat/lon) coordinates
and metric projections (UTM) to ensure circles remain circular in physical space.
"""

from typing import Tuple
import numpy as np
import pyproj
from shapely.geometry import Polygon
from shapely.ops import transform


class MetricProjector:
    """
    Handles bidirectional conversion between WGS84 (lat/lon) and UTM coordinates.

    This ensures that circles optimized in metric space remain physically circular
    rather than appearing as ellipses due to Earth's curvature.

    Attributes:
        wgs84: WGS84 coordinate reference system (EPSG:4326)
        utm: UTM coordinate reference system (auto-detected or specified)
        project_to_meters: Transformer for WGS84 -> UTM
        project_to_gps: Transformer for UTM -> WGS84
    """

    def __init__(self, polygon: Polygon = None, utm_zone: int = None):
        """
        Initialize the projector with automatic or manual UTM zone selection.

        Args:
            polygon: Optional polygon to auto-detect UTM zone from centroid
            utm_zone: Optional manual UTM zone (e.g., 10 for Zone 10N)
        """
        self.wgs84 = pyproj.CRS("EPSG:4326")

        if utm_zone is not None:
            # Manual zone specification
            self.utm = pyproj.CRS(f"EPSG:326{utm_zone:02d}")
        elif polygon is not None:
            # Auto-detect from polygon centroid
            centroid = polygon.centroid
            lon, lat = centroid.x, centroid.y
            zone = int((lon + 180) / 6) + 1
            hemisphere = "6" if lat >= 0 else "7"
            self.utm = pyproj.CRS(f"EPSG:32{hemisphere}{zone:02d}")
        else:
            raise ValueError("Must provide either polygon or utm_zone")

        self.project_to_meters = pyproj.Transformer.from_crs(
            self.wgs84, self.utm, always_xy=True
        ).transform
        self.project_to_gps = pyproj.Transformer.from_crs(
            self.utm, self.wgs84, always_xy=True
        ).transform

    def poly_to_meters(self, polygon: Polygon) -> Polygon:
        """
        Transform a polygon from WGS84 to UTM (meters).

        Args:
            polygon: Polygon in WGS84 coordinates

        Returns:
            Polygon in UTM meter coordinates
        """
        return transform(self.project_to_meters, polygon)

    def coords_to_gps(self, x_meters: float, y_meters: float) -> Tuple[float, float]:
        """
        Convert UTM meter coordinates back to WGS84 lat/lon.

        Args:
            x_meters: X coordinate in UTM meters
            y_meters: Y coordinate in UTM meters

        Returns:
            Tuple of (longitude, latitude)
        """
        return self.project_to_gps(x_meters, y_meters)


class MetricNormalizer:
    """
    Normalizes polygon coordinates from meters to [0, 1] space for optimization.

    This class handles the transformation between real-world metric coordinates
    and the normalized [0, 1] space used by the neural optimization process.

    Attributes:
        center_x: X-coordinate of the polygon center in meters
        center_y: Y-coordinate of the polygon center in meters
        scale: Scaling factor to fit polygon into [0, 1] space
    """

    def __init__(self, polygon_meters: Polygon, padding: float = 0.8):
        """
        Initialize normalizer based on polygon bounds.

        Args:
            polygon_meters: Polygon in meter coordinates
            padding: Fraction of space to use (0.8 means 80% of [0,1] space)
        """
        min_x, min_y, max_x, max_y = polygon_meters.bounds
        self.center_x = (min_x + max_x) / 2
        self.center_y = (min_y + max_y) / 2
        self.scale = max(max_x - min_x, max_y - min_y) / padding

    def normalize(self, polygon_meters: Polygon) -> Polygon:
        """
        Transform polygon from meters to [0, 1] normalized space.

        Args:
            polygon_meters: Polygon in meter coordinates

        Returns:
            Polygon in [0, 1] normalized coordinates
        """
        coords = np.array(polygon_meters.exterior.coords)
        coords[:, 0] -= self.center_x
        coords[:, 1] -= self.center_y
        coords /= self.scale
        coords += 0.5
        return Polygon(coords)

    def denormalize(
        self, centers_norm: np.ndarray, radii_norm: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert normalized circle parameters back to meter coordinates.

        Args:
            centers_norm: Array of shape (n, 2) with normalized centers
            radii_norm: Array of shape (n,) with normalized radii

        Returns:
            Tuple of (centers_meters, radii_meters)
        """
        cx_m = (centers_norm[:, 0] - 0.5) * self.scale + self.center_x
        cy_m = (centers_norm[:, 1] - 0.5) * self.scale + self.center_y
        r_m = radii_norm * self.scale
        return np.stack([cx_m, cy_m], axis=1), r_m
