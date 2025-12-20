"""
Optimization engine for circle packing using differentiable rendering.

This module contains the PyTorch-based optimization logic that fits circles
to polygons using gradient descent and soft rasterization.
"""

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.path import Path
from shapely.geometry import Polygon


class DifferentiableRenderer:
    """
    Renders shapes to a grid using differentiable operations.

    This class creates a spatial grid and provides methods for rasterizing
    polygons and circles in a way that supports gradient-based optimization.

    Attributes:
        resolution: Grid resolution (resolution × resolution pixels)
        device: PyTorch device ('cuda' or 'cpu')
        grid: Tensor of shape (resolution, resolution, 2) containing (x, y) coords
    """

    def __init__(self, resolution: int, device: str):
        """
        Initialize the renderer with a spatial grid.

        Args:
            resolution: Number of pixels per dimension
            device: PyTorch device string ('cuda' or 'cpu')
        """
        self.resolution = resolution
        self.device = device
        linspace = torch.linspace(0, 1, resolution, device=device)
        y, x = torch.meshgrid(linspace, linspace, indexing="ij")
        self.grid = torch.stack([x, y], dim=-1)

    def rasterize_polygon(self, polygon: Polygon) -> torch.Tensor:
        """
        Convert a polygon to a binary mask on the grid.

        Args:
            polygon: Shapely polygon in [0, 1] normalized coordinates

        Returns:
            Binary tensor of shape (resolution, resolution)
        """
        x, y = np.meshgrid(np.linspace(0, 1, self.resolution), np.linspace(0, 1, self.resolution))
        points = np.vstack((x.flatten(), y.flatten())).T
        path = Path(list(polygon.exterior.coords))
        mask = path.contains_points(points).reshape(self.resolution, self.resolution)
        return torch.tensor(mask, dtype=torch.float32, device=self.device)


class CircleModel(nn.Module):
    """
    Neural network representing a set of circles with learnable parameters.

    This model optimizes circle positions and sizes using gradient descent.
    Circle radii are stored in log-space for numerical stability.

    Attributes:
        centers: Learnable tensor of shape (n_circles, 2) for circle centers
        log_radii: Learnable tensor of shape (n_circles,) for log(radius)
    """

    def __init__(self, n_circles: int, init_scale: float = 0.2):
        """
        Initialize circle parameters with random values.

        Args:
            n_circles: Number of circles to optimize
            init_scale: Scale of initialization (centers in [0.4±init_scale])
        """
        super().__init__()
        # Initialize centers near the middle of [0, 1] space
        self.centers = nn.Parameter(torch.rand(n_circles, 2) * init_scale + (0.5 - init_scale / 2))
        # Initialize radii to small values (log space)
        self.log_radii = nn.Parameter(torch.ones(n_circles) * -3.0)

    def forward(self, grid: torch.Tensor, sharpness: float) -> torch.Tensor:
        """
        Render circles to a soft mask using differentiable operations.

        Uses sigmoid-based soft masking and log-sum-exp for differentiable union.

        Args:
            grid: Spatial grid tensor of shape (H, W, 2)
            sharpness: Sigmoid sharpness parameter (higher = harder edges)

        Returns:
            Soft mask tensor of shape (H, W) representing union of circles
        """
        radii = torch.exp(self.log_radii)

        # Compute distances from each grid point to each circle center
        dists = torch.norm(grid.unsqueeze(0) - self.centers.view(-1, 1, 1, 2), dim=-1)

        # Soft circle masks using sigmoid
        circle_masks = torch.sigmoid(sharpness * (radii.view(-1, 1, 1) - dists))

        # Clamp to avoid log(0) issues
        eps = 1e-6
        circle_masks = torch.clamp(circle_masks, min=eps, max=1 - eps)

        # Differentiable union: 1 - prod(1 - mask_i) = 1 - exp(sum(log(1 - mask_i)))
        union_mask = 1.0 - torch.exp(torch.sum(torch.log(1 - circle_masks), dim=0))

        return union_mask


def optimize_circles(
    target_polygon: Polygon,
    n_circles: int,
    resolution: int = 256,
    iterations: int = 2000,
    learning_rate: float = 0.08,
    start_sharpness: float = 1.0,
    end_sharpness: float = 150.0,
    device: str = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimize circle positions and sizes to approximate a polygon.

    Uses gradient descent with IoU loss and sharpness annealing to find
    optimal circle configurations.

    Args:
        target_polygon: Shapely polygon in [0, 1] normalized coordinates
        n_circles: Number of circles to fit
        resolution: Grid resolution for rasterization
        iterations: Number of optimization steps
        learning_rate: Adam optimizer learning rate
        start_sharpness: Initial sigmoid sharpness
        end_sharpness: Final sigmoid sharpness
        device: PyTorch device ('cuda', 'cpu', or None for auto)
        verbose: Whether to print progress

    Returns:
        Tuple of (centers, radii) in normalized [0, 1] coordinates
        - centers: np.ndarray of shape (n_circles, 2)
        - radii: np.ndarray of shape (n_circles,)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup renderer and target mask
    renderer = DifferentiableRenderer(resolution, device)
    target_mask = renderer.rasterize_polygon(target_polygon)

    if target_mask.sum() == 0:
        raise ValueError("Target polygon produced empty mask. Check polygon coordinates.")

    # Initialize model and optimizer
    model = CircleModel(n_circles).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Sharpness annealing schedule
    sharpness_schedule = np.linspace(start_sharpness, end_sharpness, iterations)

    # Optimization loop
    for i in range(iterations):
        optimizer.zero_grad()

        # Generate circle mask
        generated_mask = model(renderer.grid, sharpness_schedule[i])

        # IoU loss: 1 - (intersection / union)
        intersection = (generated_mask * target_mask).sum()
        union = generated_mask.sum() + target_mask.sum() - intersection
        loss = 1.0 - (intersection / (union + 1e-6))

        loss.backward()
        optimizer.step()

        if verbose and i % 200 == 0:
            print(f"Iteration {i:04d} | Loss: {loss.item():.4f}")

    # Extract optimized parameters
    centers = model.centers.detach().cpu().numpy()
    radii = torch.exp(model.log_radii).detach().cpu().numpy()

    return centers, radii


def estimate_optimal_circles(
    target_polygon: Polygon,
    min_circles: int = 2,
    max_circles: int = 10,
    resolution: int = 128,
    iterations: int = 500,
    device: str = None,
) -> int:
    """
    Estimate optimal number of circles using elbow method on IoU loss.

    Runs quick optimizations for different circle counts and finds the
    point of diminishing returns.

    Args:
        target_polygon: Shapely polygon in [0, 1] normalized coordinates
        min_circles: Minimum number of circles to test
        max_circles: Maximum number of circles to test
        resolution: Grid resolution (lower for speed)
        iterations: Number of iterations per test (lower for speed)
        device: PyTorch device ('cuda', 'cpu', or None for auto)

    Returns:
        Estimated optimal number of circles
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    renderer = DifferentiableRenderer(resolution, device)
    target_mask = renderer.rasterize_polygon(target_polygon)

    losses = []

    for n in range(min_circles, max_circles + 1):
        _, _ = optimize_circles(
            target_polygon,
            n_circles=n,
            resolution=resolution,
            iterations=iterations,
            device=device,
            verbose=False,
        )

        # Evaluate final loss
        model = CircleModel(n).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.08)

        for _ in range(iterations):
            optimizer.zero_grad()
            gen_mask = model(renderer.grid, sharpness=100.0)
            intersection = (gen_mask * target_mask).sum()
            union = gen_mask.sum() + target_mask.sum() - intersection
            loss = 1.0 - (intersection / (union + 1e-6))
            loss.backward()
            optimizer.step()

        losses.append(loss.item())

    # Find elbow using simple derivative threshold
    losses = np.array(losses)
    if len(losses) < 3:
        return min_circles

    # Calculate rate of improvement
    improvements = -np.diff(losses)
    # Find where improvement drops below threshold (elbow point)
    threshold = np.mean(improvements) * 0.3
    elbow_idx = np.where(improvements < threshold)[0]

    if len(elbow_idx) > 0:
        return min_circles + elbow_idx[0]
    else:
        return max_circles
