"""
Visualization utilities for circle packing results.

This module provides functions to visualize the circle packing results,
including both the optimization space and real-world map projections.
"""

from typing import List, Dict, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from shapely.geometry import Polygon, shape


def visualize_packing(
    polygon: Union[Polygon, List[tuple], dict],
    circles: List[Dict[str, float]],
    projection_mode: str = "auto",
    show_optimization_space: bool = False,
    figsize: tuple = (12, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize circle packing results with the original polygon.

    Args:
        polygon: Original polygon in same format as pack_polygon() input
        circles: List of circle dicts from pack_polygon() output
        projection_mode: How to handle coordinates:
            - 'auto': Detect if using lat/lon (geo) or Cartesian
            - 'geo': Treat as WGS84 lat/lon with aspect ratio correction
            - 'cartesian': Treat as simple x/y coordinates
        show_optimization_space: If True, show dual plot with normalized space
        figsize: Figure size as (width, height)
        title: Optional plot title
        save_path: Optional path to save figure

    Examples:
        >>> from circle_packing import pack_polygon, visualize_packing
        >>> polygon = [(-122.4, 37.8), (-122.3, 37.8), ...]
        >>> circles = pack_polygon(polygon, n=5)
        >>> visualize_packing(polygon, circles)
    """
    # Parse polygon
    if isinstance(polygon, dict):
        poly = shape(polygon)
    elif isinstance(polygon, list):
        poly = Polygon(polygon)
    elif isinstance(polygon, Polygon):
        poly = polygon
    else:
        raise TypeError(f"Invalid polygon type: {type(polygon)}")

    # Auto-detect projection mode
    if projection_mode == "auto":
        # Simple heuristic: if coordinates are in typical lat/lon range
        x_coords = [p[0] for p in poly.exterior.coords]
        y_coords = [p[1] for p in poly.exterior.coords]
        if -180 <= min(x_coords) <= 180 and -90 <= min(y_coords) <= 90:
            projection_mode = "geo"
        else:
            projection_mode = "cartesian"

    # Create figure
    if show_optimization_space:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        ax_main = axes[1]
    else:
        fig, ax_main = plt.subplots(1, 1, figsize=figsize)
        axes = [ax_main]

    # Plot main view
    _plot_geo_view(ax_main, poly, circles, projection_mode, title)

    # Plot optimization space if requested
    if show_optimization_space:
        _plot_normalized_view(axes[0], poly, circles)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")

    plt.show()


def _plot_geo_view(
    ax, polygon: Polygon, circles: List[Dict[str, float]], mode: str, title: Optional[str]
) -> None:
    """Plot the real-world view of polygon and circles."""
    # Plot polygon
    x, y = polygon.exterior.xy
    ax.fill(x, y, alpha=0.3, fc="green", ec="black", label="Target Polygon")

    # Plot circles
    colors = plt.cm.rainbow(np.linspace(0, 1, len(circles)))

    for i, (circle_data, color) in enumerate(zip(circles, colors)):
        cx = circle_data["centroid_x"]
        cy = circle_data["centroid_y"]
        r = circle_data["radius"]

        if mode == "geo":
            # For geographic coordinates, draw ellipse to represent circle
            _draw_geo_circle(ax, cx, cy, r, color, i)
        else:
            # For Cartesian coordinates, draw simple circle
            circle_patch = Circle(
                (cx, cy), r, color=color, fill=False, linewidth=2, label=f"Circle {i}"
            )
            ax.add_patch(circle_patch)

        # Label
        ax.text(cx, cy, str(i), color=color, weight="bold", ha="center", va="center")

    # Set aspect ratio
    if mode == "geo":
        mean_lat = np.mean(y)
        aspect_ratio = 1.0 / np.cos(np.radians(mean_lat))
        ax.set_aspect(aspect_ratio)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    else:
        ax.set_aspect("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    if title:
        ax.set_title(title)
    else:
        ax.set_title("Circle Packing Result")

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")


def _draw_geo_circle(ax, lon: float, lat: float, radius_deg: float, color, idx: int) -> None:
    """
    Draw a circle on a geographic plot using an ellipse.

    Since matplotlib plots in degree space but we want physical circles,
    we need to draw ellipses that account for latitude-dependent longitude spacing.
    """
    METERS_PER_DEG_LAT = 111320.0

    # Convert radius from degrees to meters (approximate)
    radius_meters = radius_deg * METERS_PER_DEG_LAT

    # Height in degrees (constant)
    height_deg = (radius_meters * 2) / METERS_PER_DEG_LAT

    # Width in degrees (latitude-dependent)
    meters_per_deg_lon = METERS_PER_DEG_LAT * np.cos(np.radians(lat))
    width_deg = (radius_meters * 2) / meters_per_deg_lon

    ellipse = Ellipse(
        xy=(lon, lat), width=width_deg, height=height_deg, color=color, fill=False, linewidth=2
    )
    ax.add_patch(ellipse)


def _plot_normalized_view(ax, polygon: Polygon, circles: List[Dict[str, float]]) -> None:
    """
    Plot a simplified normalized view (for debugging/demonstration).

    Note: This is approximate since we don't have access to the actual
    normalized coordinates used during optimization.
    """
    # Normalize polygon to [0, 1]
    x_coords = [p[0] for p in polygon.exterior.coords]
    y_coords = [p[1] for p in polygon.exterior.coords]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    scale = max(max_x - min_x, max_y - min_y)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # Normalize
    norm_coords = []
    for x, y in polygon.exterior.coords:
        nx = ((x - center_x) / scale) + 0.5
        ny = ((y - center_y) / scale) + 0.5
        norm_coords.append((nx, ny))

    nx, ny = zip(*norm_coords)
    ax.fill(nx, ny, alpha=0.3, fc="green", ec="black")

    # Normalize circles (approximate)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(circles)))

    for i, (circle_data, color) in enumerate(zip(circles, colors)):
        cx = ((circle_data["centroid_x"] - center_x) / scale) + 0.5
        cy = ((circle_data["centroid_y"] - center_y) / scale) + 0.5
        r = circle_data["radius"] / scale

        circle_patch = Circle((cx, cy), r, color=color, fill=False, linewidth=2)
        ax.add_patch(circle_patch)
        ax.text(cx, cy, str(i), color=color, weight="bold", ha="center", va="center")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title("Normalized Optimization Space")
    ax.set_xlabel("Normalized X")
    ax.set_ylabel("Normalized Y")
    ax.grid(True, linestyle="--", alpha=0.3)


def print_circle_summary(circles: List[Dict[str, float]]) -> None:
    """
    Print a formatted summary of circle packing results.

    Args:
        circles: List of circle dictionaries from pack_polygon()

    Examples:
        >>> from circle_packing import pack_polygon, print_circle_summary
        >>> circles = pack_polygon(polygon, n=5)
        >>> print_circle_summary(circles)
    """
    print("\n" + "=" * 60)
    print(f"      CIRCLE PACKING RESULTS ({len(circles)} circles)")
    print("=" * 60)
    print(f"{'ID':<4} | {'X':<12} | {'Y':<12} | {'Radius':<12}")
    print("-" * 60)

    for i, circle in enumerate(circles):
        cx = circle["centroid_x"]
        cy = circle["centroid_y"]
        r = circle["radius"]
        print(f"{i:<4} | {cx:<12.6f} | {cy:<12.6f} | {r:<12.6f}")

    print("=" * 60 + "\n")
