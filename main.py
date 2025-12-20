#!/usr/bin/env python3
"""
Sigil - Circle Packing Demo

Simple demonstration of the sigil package with a basic polygon.
For the San Francisco ZIP code demo, see examples/sf_zipcodes_demo.py
"""

from sigil import pack_polygon, visualize_packing, print_circle_summary


def main():
    """Run a simple circle packing demo with a basic polygon."""

    print("=" * 70)
    print("  Circle Packing Demo")
    print("=" * 70)

    # Define a simple polygon (a rectangle)
    polygon = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0), (0.0, 0.0)]

    print("\nInput polygon: Rectangle (2.0 x 1.0)")
    print(f"Vertices: {len(polygon)}")

    # Pack circles into the polygon
    print("\nPacking 5 circles into polygon...")
    circles = pack_polygon(
        polygon,
        n=5,
        resolution=256,
        iterations=1500,
        use_projection=False,  # Simple Cartesian coordinates
        verbose=True,
    )

    # Print results
    print_circle_summary(circles)

    # Visualize
    print("Generating visualization...")
    visualize_packing(
        polygon,
        circles,
        projection_mode="cartesian",
        title="Circle Packing Demo: Rectangle",
        save_path="circle_packing_demo.png",
    )

    print("\nDemo complete!")
    print("For more advanced examples (e.g., SF ZIP codes), see examples/")


if __name__ == "__main__":
    main()
