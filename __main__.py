"""
Semi-Automated Clustering - Interactive Annotation Tool

Usage:
    python -m semi_automated_clustering <data_file.h5>
"""

import sys
from pathlib import Path


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python -m semi_automated_clustering <data_file.h5>")
        sys.exit(1)
    
    data_path = Path(sys.argv[1])
    if not data_path.exists():
        print(f"Error: File not found: {data_path}")
        sys.exit(1)
    
    from .clustering_app import ClusteringApp
    app = ClusteringApp(str(data_path))
    app.run()


if __name__ == "__main__":
    main()
