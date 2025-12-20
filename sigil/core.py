"""
Core API for circle packing.

This module provides the main user-facing function for packing circles into polygons.
"""

from typing import List, Dict, Optional, Union
import numpy as np
from shapely.geometry import Polygon, shape

from .projection import MetricProjector, MetricNormalizer
from .optimizer import optimize_circles, estimate_optimal_circles


def pack_polygon(
    polygon: Union[Polygon, List[tuple], dict],
    n: Optional[int] = None,
    resolution: int = 256,
    iterations: int = 2000,
    learning_rate: float = 0.08,
    device: Optional[str] = None,
    use_projection: bool = True,
    utm_zone: Optional[int] = None,
    verbose: bool = False,
) -> List[Dict[str, float]]:
    """
    Approximate a polygon as a set of circles using gradient-based optimization.

    This is the main API function for the circle packing package. It accepts
    a polygon in various formats and returns a list of circles that approximate it.

    Args:
        polygon: Input polygon in one of the following formats:
            - shapely.geometry.Polygon (in WGS84 lat/lon if use_projection=True)
            - List of (x, y) coordinate tuples
            - GeoJSON-like dict with 'type' and 'coordinates' keys
        n: Number of circles to fit. If None, auto-detects using elbow method.
        resolution: Grid resolution for optimization (higher = more accurate, slower)
        iterations: Number of optimization iterations
        learning_rate: Gradient descent learning rate
        device: PyTorch device ('cuda', 'cpu', or None for auto-detect)
        use_projection: Whether to project lat/lon to metric space (recommended for geo data)
        utm_zone: Manual UTM zone specification (None for auto-detect)
        verbose: Whether to print progress information

    Returns:
        List of dictionaries, each containing:
            - 'radius': Circle radius
            - 'centroid_x': Circle center X coordinate
            - 'centroid_y': Circle center Y coordinate
        Coordinates are in the same space as input (lat/lon if use_projection=True)

    Raises:
        ValueError: If polygon is invalid or produces empty mask
        TypeError: If polygon format is not recognized

    Examples:
        >>> from circle_packing import pack_polygon
        >>> polygon = [(-122.4, 37.8), (-122.3, 37.8), (-122.3, 37.7), (-122.4, 37.7)]
        >>> circles = pack_polygon(polygon, n=5)
        >>> print(circles[0])
        {'radius': 0.015, 'centroid_x': -122.35, 'centroid_y': 37.75}
    """
    # Parse input polygon
    if isinstance(polygon, dict):
        # GeoJSON-like format
        poly = shape(polygon)
    elif isinstance(polygon, list):
        # List of coordinates
        poly = Polygon(polygon)
    elif isinstance(polygon, Polygon):
        poly = polygon
    else:
        raise TypeError(
            f"polygon must be Polygon, list of tuples, or GeoJSON dict, got {type(polygon)}"
        )

    # Validate polygon
    if not poly.is_valid:
        raise ValueError("Invalid polygon geometry")
    if poly.is_empty:
        raise ValueError("Empty polygon")

    # Handle MultiPolygon by taking largest component
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda p: p.area)
        if verbose:
            print("MultiPolygon detected, using largest component")

    # Setup projection pipeline if needed
    if use_projection:
        if verbose:
            print("Projecting to metric space (UTM)...")
        projector = MetricProjector(polygon=poly, utm_zone=utm_zone)
        poly_meters = projector.poly_to_meters(poly)
    else:
        projector = None
        poly_meters = poly

    # Normalize to [0, 1] space
    normalizer = MetricNormalizer(poly_meters)
    poly_norm = normalizer.normalize(poly_meters)

    # Auto-detect optimal circle count if not specified
    if n is None:
        if verbose:
            print("Auto-detecting optimal number of circles...")
        n = estimate_optimal_circles(
            poly_norm,
            min_circles=2,
            max_circles=10,
            resolution=resolution // 2,
            iterations=iterations // 4,
            device=device,
        )
        if verbose:
            print(f"Selected {n} circles")

    # Optimize circles
    if verbose:
        print(f"Optimizing {n} circles...")
    centers_norm, radii_norm = optimize_circles(
        poly_norm,
        n_circles=n,
        resolution=resolution,
        iterations=iterations,
        learning_rate=learning_rate,
        device=device,
        verbose=verbose,
    )

    # Denormalize back to metric space
    centers_meters, radii_meters = normalizer.denormalize(centers_norm, radii_norm)

    # Project back to original coordinate system if needed
    if use_projection:
        centers_gps = np.array([projector.coords_to_gps(cx, cy) for cx, cy in centers_meters])
        # Convert radii using simple scaling (approximate)
        # For more accuracy, could compute radius in degrees at each latitude
        scale_factor = normalizer.scale / 111320.0  # meters per degree (approximate)
        radii_output = radii_meters / 111320.0
        centers_output = centers_gps
    else:
        centers_output = centers_meters
        radii_output = radii_meters

    # Format output
    results = []
    for i in range(n):
        results.append(
            {
                "radius": float(radii_output[i]),
                "centroid_x": float(centers_output[i, 0]),
                "centroid_y": float(centers_output[i, 1]),
            }
        )

    return results
