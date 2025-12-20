"""
Sigil - Circle Packing Package

A Python package to approximate irregular polygons as sets of circles using
differentiable rendering and gradient-based optimization.
"""

from .core import pack_polygon
from .visualization import visualize_packing, print_circle_summary

__version__ = "0.1.0"
__all__ = ["pack_polygon", "visualize_packing", "print_circle_summary"]
