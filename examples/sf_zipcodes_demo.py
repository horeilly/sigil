"""
San Francisco ZIP Code Circle Packing Demo

This script demonstrates the sigil package by fitting circles
to San Francisco ZIP code boundaries.
"""

import os
import json
import requests
from shapely.geometry import shape

# Import the sigil package
from sigil import pack_polygon, visualize_packing, print_circle_summary


class SFZipCodeLoader:
    """Helper class to load SF ZIP code data."""

    DATA_URL = "https://data.sfgov.org/resource/srq6-hmpi.geojson"
    CACHE_FILE = "sf_zipcodes.geojson"

    @staticmethod
    def get_zip_polygon(zip_code: str):
        """
        Load a polygon for a specific SF ZIP code.

        Args:
            zip_code: ZIP code string (e.g., "94123")

        Returns:
            shapely.geometry.Polygon
        """
        # Check local cache
        if not os.path.exists(SFZipCodeLoader.CACHE_FILE):
            print(f"Downloading SF ZIP code dataset to {SFZipCodeLoader.CACHE_FILE}...")
            try:
                response = requests.get(SFZipCodeLoader.DATA_URL)
                response.raise_for_status()
                with open(SFZipCodeLoader.CACHE_FILE, "w") as f:
                    json.dump(response.json(), f)
            except Exception as e:
                print(f"Download failed: {e}")
                print("Using hardcoded fallback for 94123...")
                return SFZipCodeLoader.get_fallback_94123()

        # Load from cache
        with open(SFZipCodeLoader.CACHE_FILE, "r") as f:
            data = json.load(f)

        # Find the target ZIP code
        target_feature = None
        for feature in data["features"]:
            props = feature["properties"]
            if props.get("zip_code") == zip_code or props.get("id") == zip_code:
                target_feature = feature
                break

        if not target_feature:
            print(f"ZIP code {zip_code} not found. Using fallback for 94123...")
            return SFZipCodeLoader.get_fallback_94123()

        # Parse geometry
        poly = shape(target_feature["geometry"])

        # Handle MultiPolygon
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda p: p.area)
            print(f"MultiPolygon detected for {zip_code}, using largest component")

        return poly

    @staticmethod
    def get_fallback_94123():
        """Hardcoded fallback polygon for Marina District (94123)."""
        coords = [
            (-122.4477, 37.8055),
            (-122.4450, 37.8058),
            (-122.4420, 37.8062),
            (-122.4380, 37.8068),
            (-122.4360, 37.8065),
            (-122.4330, 37.8055),
            (-122.4280, 37.8065),
            (-122.4270, 37.8080),
            (-122.4250, 37.8075),
            (-122.4240, 37.8055),
            (-122.4242, 37.7950),
            (-122.4265, 37.7950),
            (-122.4265, 37.7930),
            (-122.4350, 37.7930),
            (-122.4350, 37.7925),
            (-122.4475, 37.7925),
            (-122.4477, 37.8055),
        ]
        from shapely.geometry import Polygon

        return Polygon(coords)


def main():
    """Run the SF ZIP code circle packing demo."""
    # Configuration
    ZIP_CODE = "94123"  # Marina District
    N_CIRCLES = 4  # Set to None for auto-detection
    RESOLUTION = 256
    ITERATIONS = 2000

    print("=" * 70)
    print("  San Francisco ZIP Code Circle Packing Demo")
    print("=" * 70)
    print(f"Target ZIP Code: {ZIP_CODE}")
    print(f"Number of Circles: {N_CIRCLES if N_CIRCLES else 'Auto-detect'}")
    print("=" * 70)

    # Load the polygon
    print(f"\nLoading polygon for ZIP code {ZIP_CODE}...")
    polygon = SFZipCodeLoader.get_zip_polygon(ZIP_CODE)
    print(f"Polygon loaded: {len(polygon.exterior.coords)} vertices")

    # Pack circles
    print("\nPacking circles into polygon...")
    circles = pack_polygon(
        polygon,
        n=N_CIRCLES,
        resolution=RESOLUTION,
        iterations=ITERATIONS,
        use_projection=True,  # Use UTM projection for accurate circles
        verbose=True,
    )

    # Print results
    print_circle_summary(circles)

    # Visualize
    print("Generating visualization...")
    visualize_packing(
        polygon,
        circles,
        projection_mode="geo",
        title=f"Circle Packing for SF ZIP {ZIP_CODE}",
        save_path=f"circle_packing_{ZIP_CODE}.png",
    )


if __name__ == "__main__":
    main()
