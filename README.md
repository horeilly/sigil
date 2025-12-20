# Sigil

## TL;DR
A Python package to approximate an irregular polygon as a set of circles, with optional auto-detection of optimal circle count. Accepts standard polygon input, produces both raw circle data and a visualization, and is engineered with DevOps hygiene suitable for portfolio/demo use.

## Goals

### Business Goals
- Demonstrate technical breadth as a Data Science leader, with publicly accessible, well-engineered projects.
- Build a reusable codebase for geometry/optimization problems.
- Showcase DevOps workflow, deployment automation, and code quality standards.

### User Goals
- Import and run the package for polygon-to-circle packing.
- Easily get both numerical results and visuals for use in demos, notebooks, or documentation.
- Optionally vary key parameters (like number of circles).

### Non-Goals
- No real-time web app for external users (yet).
- No billing, authentication, or commercial features.
- No support for highly performant large-scale geospatial datasets.

## User Stories
- As a Data Scientist, I want to fit circles to arbitrary polygons, so that I can demonstrate geometric optimization in presentations.
- As a developer, I want to import this as a Python library, so that I can integrate it into Jupyter notebooks or pipelines.
- As a hobbyist, I want to visualize the packing, so that I can experiment and tweak inputs for different shapes.

## Functional Requirements
- Core Algorithm (Priority: High)
  - Accept input polygon (list of coordinates or GeoJSON).
  - Accept optional n; otherwise autodetect using suitable metric (e.g., elbow method).
  - Grid-based differentiable rasterization for circle fitting.
  - Compute suitable initial centroids.
- API/Interface (Priority: High)
  - Python function/class interface.
  - Output: Array of dictionaries, each with radius, centroid_x, centroid_y.
- Visualization (Priority: Medium)
  - Function to generate an image (matplotlib, etc.) showing polygon outline and circles.
- Packaging (Priority: High)
  - Installable via pip (or equivalent, e.g., Poetry support).
- Testing & Code Quality (Priority: High)
  - Unit tests for inputs/outputs, edge cases, and algorithm steps.
  - Automated linting and type-checking, e.g., with flake8/black/pylint and mypy.
- DevOps & Deployment (Priority: Medium)
  - GitHub Actions to run tests/linting on commit/push.
  - (Optional) Terraform/GCP scripts for potential future hosting.

## User Experience
- User installs via pip.
- Imports in notebook/script: from sigil import pack_polygon
- Calls main function, passing polygon and options.
- Receives output (list of circles), optionally calls viz function for output.
- Views resulting image in notebook or as file.

## Technical Considerations
- Python-first; minimal non-Python dependencies.
- Codebase ready for CI (GitHub Actions).
- Focus on modularity/testability: split main algorithm from I/O and viz.
- Type hints and docstrings throughout.
- Testing: Use pytest; test basic input validation, algorithm output, and (if feasible) some image output.

## Success Metrics
- All primary features covered by automated tests and CI.
- Can be imported and run in a clean virtualenv.
- Generates accurate results for at least 3 "typical" test polygons.
- README/docs clear enough for DS/engineering peers to use and understand.
