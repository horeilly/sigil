# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python package for sigil optimization that uses differentiable rendering and PyTorch to approximate irregular polygons as sets of circles. The package provides a clean API for working with any polygon and includes specialized support for geographic data (e.g., ZIP code boundaries) using proper metric projections.

## Installation and Running

**Install package in development mode:**
```bash
pip install -e .
```

**Install development dependencies (includes Black, pytest, mypy, flake8):**
```bash
pip install -r requirements-dev.txt
```

**Run the basic demo:**
```bash
python main.py
```

**Run the SF ZIP code demo:**
```bash
python examples/sf_zipcodes_demo.py
```

**Quick test:**
```bash
python test_basic.py
```

## Development Tools

**Code formatting with Black:**
```bash
# Format all code
black sigil/ main.py test_basic.py examples/

# Check formatting without changes
black --check sigil/ main.py test_basic.py examples/
```

**Linting with flake8:**
```bash
flake8 sigil/ main.py test_basic.py examples/
```

**Type checking with mypy:**
```bash
mypy sigil/
```

**Run tests:**
```bash
pytest
```

Black configuration is in `pyproject.toml` with line length set to 100.

## Package Structure

```
sigil/
├── __init__.py          # Package entry point, exports main API
├── core.py              # Main API: pack_polygon() function
├── projection.py        # MetricProjector and MetricNormalizer classes
├── optimizer.py         # DifferentiableRenderer, CircleModel, optimization logic
└── visualization.py     # visualize_packing() and print_circle_summary()

examples/
└── sf_zipcodes_demo.py  # Demo for SF ZIP code boundaries

main.py                  # Simple demo script
test_basic.py           # Quick functionality test
```

## Main API Usage

```python
from sigil import pack_polygon, visualize_packing, print_circle_summary

# Define a polygon (list of tuples, Polygon object, or GeoJSON dict)
polygon = [(0, 0), (2, 0), (2, 1), (0, 1), (0, 0)]

# Pack circles into the polygon
circles = pack_polygon(
    polygon,
    n=5,                    # Number of circles (None for auto-detect)
    resolution=256,         # Grid resolution
    iterations=2000,        # Optimization iterations
    use_projection=False,   # True for geographic data (lat/lon)
    verbose=True
)

# Results are list of dicts: [{'radius': ..., 'centroid_x': ..., 'centroid_y': ...}, ...]
print_circle_summary(circles)
visualize_packing(polygon, circles)
```

## Architecture

### Core Pipeline Flow

When `use_projection=True` (for geographic data), the code follows a 4-stage pipeline:

1. **GPS Coordinates (WGS84)** → Input polygon in lat/lon
2. **Metric Space (UTM)** → Projected via `MetricProjector` to ensure circles stay circular
3. **Normalized Space [0,1]** → Scaled via `MetricNormalizer` for neural optimization
4. **Optimization** → Differentiable rendering with PyTorch
5. **Reverse Pipeline** → Denormalize → Project back to lat/lon

For Cartesian data (`use_projection=False`), steps 2 and 5 are skipped.

### Key Modules

**`sigil/core.py`**: Main API
- `pack_polygon()`: Primary function that orchestrates the entire pipeline. Handles input parsing, projection setup, normalization, optimization, and result formatting.
- Supports auto-detection of optimal circle count using elbow method
- Validates input and handles MultiPolygon geometries

**`sigil/projection.py`**: Coordinate transformations
- `MetricProjector`: Bidirectional WGS84 ↔ UTM transformation with auto-detection of UTM zone from polygon centroid
- `MetricNormalizer`: Scales metric coordinates to [0,1] normalized space and back
- Critical for ensuring circles remain physically circular on Earth's surface

**`sigil/optimizer.py`**: Optimization engine
- `DifferentiableRenderer`: Creates spatial grid and rasterizes polygons to binary masks
- `CircleModel`: PyTorch neural network with learnable circle centers and log-radii
- `optimize_circles()`: Main optimization loop using Adam optimizer with IoU loss
- `estimate_optimal_circles()`: Auto-detection using elbow method on loss curve

**`sigil/visualization.py`**: Visualization utilities
- `visualize_packing()`: Creates plots with automatic projection mode detection
- `print_circle_summary()`: Formatted text output of results
- Handles geographic ellipse rendering for lat/lon coordinates

### Optimization Strategy

**Loss Function**: 1 - IoU (Intersection over Union)
```python
intersection = (generated_mask * target_mask).sum()
union = generated_mask.sum() + target_mask.sum() - intersection
loss = 1.0 - (intersection / (union + 1e-6))
```

**Sharpness Annealing**: Sigmoid sharpness increases from 1.0 → 150.0 over iterations, starting with soft circles and gradually hardening them to prevent local minima.

**Soft Union**: Differentiable union computed using log-sum-exp:
```python
union_mask = 1.0 - exp(sum(log(1 - circle_masks)))
```

**Log-space Radii**: Radii stored as `log_radii` prevents negative values and improves gradient flow.

### Critical Implementation Details

1. **UTM Projection for Geographic Data**: Without projecting lat/lon to metric space, optimized "circles" would be ellipses on Earth's surface. `MetricProjector` auto-detects the appropriate UTM zone based on polygon centroid.

2. **Auto-detection of Circle Count**: Uses elbow method by running quick optimizations for different counts and finding the point of diminishing returns in loss reduction.

3. **Ellipse Visualization**: For geographic plots, circles are drawn as `Ellipse` patches because matplotlib plots in degree space. Width/height are computed by dividing physical radius by latitude-dependent degrees-per-meter conversion factors.

4. **Input Flexibility**: `pack_polygon()` accepts Shapely Polygon objects, lists of coordinate tuples, or GeoJSON-like dicts.

## Testing

Basic functionality test exists in `test_basic.py`. The README specifies pytest as the intended framework with requirements for:
- Input validation tests
- Algorithm output verification
- Visualization output tests
- Coverage of at least 3 typical test polygons

**TODO**: Implement comprehensive pytest test suite.

## Alignment with README/PRD

The code has been refactored to align with the README goals:

✅ **Completed:**
- Modular, well-engineered codebase with type hints and docstrings
- Main API function `pack_polygon()` matching README specification
- Automatic circle count detection using elbow method
- Separate visualization function
- Works with generic polygons (not just SF ZIP codes)
- Proper packaging structure with `setup.py`

**Still TODO (from README):**
- Formal pytest test suite with CI/CD (GitHub Actions)
- pip installation to PyPI
- Terraform/GCP deployment scripts (optional, low priority)
- Performance optimization for large-scale geospatial datasets