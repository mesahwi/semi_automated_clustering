"""
Blob detector for UMAP clustering.

Implements circular kernel convolution + thresholding for cluster detection,
matching the MATLAB FlatClust algorithm.
"""

import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Blob:
    """A detected blob/cluster in the UMAP space."""
    blob_id: int
    pixel_indices: np.ndarray  # Indices into flattened image
    segment_indices: np.ndarray  # Indices into segment array
    size: int  # Number of pixels
    
    @property
    def n_segments(self) -> int:
        return len(self.segment_indices)


class BlobDetector:
    """
    Detect clusters in UMAP space using density-based blob detection.
    
    Algorithm (matching MATLAB FlatClust):
    1. Create 2D histogram of UMAP coordinates
    2. Convolve with circular kernel
    3. Threshold to get binary image
    4. Connected component labeling
    5. Return largest N blobs
    """
    
    def __init__(
        self,
        n_pixels: int = 1200,
        radius: float = 5.0,
        theta: float = 0.1,
        max_clust: int = 20,
    ):
        """
        Args:
            n_pixels: Size of the density image (n_pixels x n_pixels)
            radius: Radius of circular convolution kernel
            theta: Threshold for blob detection
            max_clust: Maximum number of clusters to return
        """
        self.n_pixels = n_pixels
        self.radius = radius
        self.theta = theta
        self.max_clust = max_clust
        
        # Computed attributes
        self._kernel: Optional[np.ndarray] = None
        self._density_image: Optional[np.ndarray] = None
        self._convolved_image: Optional[np.ndarray] = None
        self._blob_image: Optional[np.ndarray] = None
        self._blobs: List[Blob] = []
        self._map_indices: Optional[np.ndarray] = None  # Maps segment to pixel index
        self._maprange: Optional[np.ndarray] = None
    
    @property
    def kernel(self) -> np.ndarray:
        """Get or create circular convolution kernel."""
        if self._kernel is None or self._kernel.shape[0] != 2 * int(self.radius) + 1:
            self._kernel = self._create_circular_kernel(self.radius)
        return self._kernel
    
    def _create_circular_kernel(self, radius: float) -> np.ndarray:
        """Create a circular averaging kernel."""
        r = int(radius)
        size = 2 * r + 1
        kernel = np.zeros((size, size))
        
        for i in range(size):
            for j in range(size):
                if (i - r) ** 2 + (j - r) ** 2 <= r ** 2:
                    kernel[i, j] = 1
        
        # Normalize
        kernel = kernel / kernel.sum()
        return kernel
    
    def compute_density_image(
        self,
        umap_coords: np.ndarray,
        maprange: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Create 2D density histogram of UMAP coordinates.
        
        Args:
            umap_coords: (N, 2) array of UMAP coordinates
            maprange: [[min_x, min_y], [max_x, max_y]] or None for auto
        
        Returns:
            (n_pixels, n_pixels) density image
        """
        if maprange is None:
            min_x, min_y = umap_coords.min(axis=0)
            max_x, max_y = umap_coords.max(axis=0)
            # Add margin to prevent blob cutoff at edges (5% is enough for typical kernel radii)
            margin_x = (max_x - min_x) * 0.05
            margin_y = (max_y - min_y) * 0.05
            maprange = np.array([
                [min_x - margin_x, min_y - margin_y],
                [max_x + margin_x, max_y + margin_y]
            ])
        
        self._maprange = maprange
        
        # Compute pixel indices for each point
        n = self.n_pixels
        x_scaled = (umap_coords[:, 0] - maprange[0, 0]) / (maprange[1, 0] - maprange[0, 0]) * (n - 1)
        y_scaled = (umap_coords[:, 1] - maprange[0, 1]) / (maprange[1, 1] - maprange[0, 1]) * (n - 1)
        
        x_idx = np.clip(x_scaled.astype(int), 0, n - 1)
        y_idx = np.clip(y_scaled.astype(int), 0, n - 1)
        
        # Store mapping from segment to pixel
        self._map_indices = y_idx * n + x_idx  # Flattened index
        
        # Create density image (count points per pixel)
        self._density_image = np.zeros((n, n), dtype=np.float32)
        for i in range(len(umap_coords)):
            self._density_image[y_idx[i], x_idx[i]] += 1
        
        return self._density_image
    
    def detect_blobs(
        self,
        umap_coords: np.ndarray,
        maprange: Optional[np.ndarray] = None,
    ) -> List[Blob]:
        """
        Detect blobs in UMAP space.
        
        Args:
            umap_coords: (N, 2) array of UMAP coordinates
            maprange: Optional map range [[min_x, min_y], [max_x, max_y]]
        
        Returns:
            List of Blob objects, sorted by size (largest first)
        """
        # Step 1: Compute density image
        self.compute_density_image(umap_coords, maprange)
        
        # Step 2: Convolve with circular kernel (FFT for speed)
        # Pad the density image by kernel radius to prevent edge cutoff
        kernel_radius = int(self.radius)
        
        if kernel_radius > 0:
            padded_density = np.pad(
                self._density_image, 
                kernel_radius, 
                mode='constant', 
                constant_values=0
            )
            
            convolved_padded = fftconvolve(padded_density, self.kernel, mode='same')
            
            # Remove padding to get back to original size
            self._convolved_image = convolved_padded[
                kernel_radius:kernel_radius + self.n_pixels,
                kernel_radius:kernel_radius + self.n_pixels
            ]
        else:
            self._convolved_image = fftconvolve(
                self._density_image, self.kernel, mode='same'
            )
        
        # Step 3: Threshold
        binary_image = self._convolved_image > self.theta
        
        # Step 4: Connected component labeling
        labeled_image, n_labels = ndimage.label(binary_image)
        
        # Step 5: Extract blob info and sort by size
        blob_sizes = ndimage.sum_labels(
            np.ones_like(labeled_image),
            labeled_image,
            index=range(1, n_labels + 1)
        )
        
        # Sort by size (descending)
        sorted_indices = np.argsort(blob_sizes)[::-1]
        
        # Create blob image with numbered blobs (largest = highest number)
        self._blob_image = np.zeros_like(labeled_image)
        self._blobs = []
        
        n_blobs = min(self.max_clust, len(sorted_indices))
        for rank, orig_idx in enumerate(sorted_indices[:n_blobs]):
            blob_label = orig_idx + 1  # Labels are 1-indexed
            blob_id = n_blobs - rank  # Largest blob gets highest ID
            
            # Find pixels in this blob
            pixel_mask = labeled_image == blob_label
            pixel_indices = np.where(pixel_mask.flatten())[0]
            
            # Find segments in this blob
            segment_indices = np.where(np.isin(self._map_indices, pixel_indices))[0]
            
            # Update blob image
            self._blob_image[pixel_mask] = blob_id
            
            self._blobs.append(Blob(
                blob_id=blob_id,
                pixel_indices=pixel_indices,
                segment_indices=segment_indices,
                size=int(blob_sizes[orig_idx]),
            ))
        
        return self._blobs
    
    def get_blob_at_pixel(self, x: int, y: int) -> Optional[Blob]:
        """Get the blob at a specific pixel location."""
        if self._blob_image is None:
            return None
        
        n = self.n_pixels
        if not (0 <= x < n and 0 <= y < n):
            return None
        
        blob_id = self._blob_image[y, x]
        if blob_id == 0:
            return None
        
        for blob in self._blobs:
            if blob.blob_id == blob_id:
                return blob
        return None
    
    def get_blob_at_umap_coord(self, umap_x: float, umap_y: float) -> Optional[Blob]:
        """Get the blob at a specific UMAP coordinate."""
        if self._maprange is None:
            return None
        
        n = self.n_pixels
        x = int((umap_x - self._maprange[0, 0]) / (self._maprange[1, 0] - self._maprange[0, 0]) * (n - 1))
        y = int((umap_y - self._maprange[0, 1]) / (self._maprange[1, 1] - self._maprange[0, 1]) * (n - 1))
        
        return self.get_blob_at_pixel(x, y)
    
    @property
    def blob_image(self) -> Optional[np.ndarray]:
        """Get the blob-labeled image."""
        return self._blob_image
    
    @property 
    def convolved_image(self) -> Optional[np.ndarray]:
        """Get the convolved density image."""
        return self._convolved_image
    
    @property
    def maprange(self) -> Optional[np.ndarray]:
        """Get the map range used for coordinate mapping."""
        return self._maprange
    
    def adjust_threshold(self, factor: float = 1.2) -> None:
        """Adjust threshold by a factor."""
        self.theta *= factor
    
    def adjust_radius(self, delta: float = 0.5) -> None:
        """Adjust radius by a delta."""
        self.radius = max(1.0, self.radius + delta)
        self._kernel = None  # Force kernel recomputation
    
    def update_threshold_only(self) -> List[Blob]:
        """
        Fast update when only threshold changed (no reconvolution needed).
        Reuses the cached convolved image.
        """
        if self._convolved_image is None:
            raise RuntimeError("Must call detect_blobs first")
        
        # Step 3: Threshold (using cached convolved image)
        binary_image = self._convolved_image > self.theta
        
        # Step 4: Connected component labeling
        labeled_image, n_labels = ndimage.label(binary_image)
        
        # Step 5: Extract blob info
        if n_labels > 0:
            blob_sizes = ndimage.sum_labels(
                np.ones_like(labeled_image),
                labeled_image,
                index=range(1, n_labels + 1)
            )
            sorted_indices = np.argsort(blob_sizes)[::-1]
        else:
            blob_sizes = []
            sorted_indices = []
        
        # Create blob image
        self._blob_image = np.zeros_like(labeled_image)
        self._blobs = []
        
        n_blobs = min(self.max_clust, len(sorted_indices))
        for rank, orig_idx in enumerate(sorted_indices[:n_blobs]):
            blob_label = orig_idx + 1
            blob_id = n_blobs - rank
            
            pixel_mask = labeled_image == blob_label
            pixel_indices = np.where(pixel_mask.flatten())[0]
            segment_indices = np.where(np.isin(self._map_indices, pixel_indices))[0]
            
            self._blob_image[pixel_mask] = blob_id
            
            self._blobs.append(Blob(
                blob_id=blob_id,
                pixel_indices=pixel_indices,
                segment_indices=segment_indices,
                size=int(blob_sizes[orig_idx]) if len(blob_sizes) > orig_idx else 0,
            ))
        
        return self._blobs
    
    def update_radius_only(self) -> List[Blob]:
        """
        Fast update when only radius changed (reuses density image).
        Must reconvolve but doesn't rebuild density histogram.
        """
        if self._density_image is None:
            raise RuntimeError("Must call detect_blobs first")
        
        # Reconvolve with new kernel (FFT for speed)
        self._convolved_image = fftconvolve(
            self._density_image, self.kernel, mode='same'
        )
        
        # Then threshold
        return self.update_threshold_only()
