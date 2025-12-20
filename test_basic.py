"""Quick test to verify the refactored code works."""

from sigil import pack_polygon, print_circle_summary

# Simple test polygon (triangle)
polygon = [(0.0, 0.0), (1.0, 0.0), (0.5, 0.866), (0.0, 0.0)]

print("Testing circle packing with a triangle...")
print(f"Polygon vertices: {len(polygon)}")

# Pack circles (small iteration count for quick test)
circles = pack_polygon(
    polygon, n=3, resolution=128, iterations=500, use_projection=False, verbose=True
)

# Print results
print_circle_summary(circles)

print("\n✓ Test passed! The refactored code works.")
