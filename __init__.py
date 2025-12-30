"""
Semi-Automated Clustering - Interactive Annotation Tool

A visual, interactive clustering tool for high-dimensional data using UMAP embeddings.
"""

from .clustering_app import ClusteringApp
from .data_loader import FlatData, load_data, save_to_hdf5

__version__ = "1.0.0"
__all__ = ["ClusteringApp", "FlatData", "load_data", "save_to_hdf5"]
