"""Map pre-processing utilities for LOS visibility-graph generation.

This module implements the map-processing stages referenced by the planner
article:
1. Canny-style edge extraction + morphology over `C_free`.
2. Corner detection with DBSCAN clustering to obtain graph vertices.
3. EDT + gradient-based blocked-corridor reconstruction from a dynamic obstacle.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy import ndimage
from scipy.spatial import cKDTree

from .map_grid import FREE_SPACE_VALUE, GridPoint, MapGrid
from .visibility_graph import build_visibility_graph

BlockedSegment = tuple[GridPoint, GridPoint]

EPSILON: float = 1e-9


def _load_cv2() -> Any | None:
    """Loads OpenCV lazily when available, otherwise returns `None`."""
    try:
        return import_module("cv2")
    except ModuleNotFoundError:
        return None


@dataclass(frozen=True)
class MapProcessingConfig:
    """Configuration bundle for map pre-processing routines."""

    canny_low_threshold: float = 35.0
    canny_high_threshold: float = 90.0
    edge_gaussian_sigma: float = 1.0
    morphology_kernel_size: int = 3
    morphology_iterations: int = 1
    corner_response_k: float = 0.04
    corner_response_threshold_ratio: float = 0.01
    corner_nms_window_size: int = 3
    dbscan_eps: float = 2.0
    dbscan_min_samples: int = 1
    gradient_step_size: float = 1.0
    max_walk_steps: int = 512


def _validate_processing_config(config: MapProcessingConfig) -> None:
    """Validates map-processing parameters before running the pipeline."""
    if config.canny_low_threshold < 0.0 or config.canny_high_threshold < 0.0:
        raise ValueError("Canny thresholds must be non-negative.")
    if config.canny_low_threshold >= config.canny_high_threshold:
        raise ValueError(
            "canny_low_threshold must be lower than canny_high_threshold."
        )
    if config.edge_gaussian_sigma <= 0.0:
        raise ValueError("edge_gaussian_sigma must be greater than 0.")
    if config.morphology_kernel_size <= 0:
        raise ValueError("morphology_kernel_size must be greater than 0.")
    if config.morphology_iterations < 0:
        raise ValueError("morphology_iterations must be greater than or equal to 0.")
    if config.corner_response_k <= 0.0:
        raise ValueError("corner_response_k must be greater than 0.")
    if not 0.0 < config.corner_response_threshold_ratio <= 1.0:
        raise ValueError("corner_response_threshold_ratio must be in (0, 1].")
    if config.corner_nms_window_size <= 0:
        raise ValueError("corner_nms_window_size must be greater than 0.")
    if config.dbscan_eps <= 0.0:
        raise ValueError("dbscan_eps must be greater than 0.")
    if config.dbscan_min_samples <= 0:
        raise ValueError("dbscan_min_samples must be greater than 0.")
    if config.gradient_step_size <= 0.0:
        raise ValueError("gradient_step_size must be greater than 0.")
    if config.max_walk_steps <= 0:
        raise ValueError("max_walk_steps must be greater than 0.")


def _normalize_vector(vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64] | None:
    """Returns a unit vector or `None` when input length is nearly zero."""
    norm = float(np.linalg.norm(vector))
    if norm <= EPSILON:
        return None
    return vector / norm


def _in_bounds(point: GridPoint, shape: tuple[int, int]) -> bool:
    """Checks whether a `(row, col)` point is inside an array shape."""
    row, col = point
    return 0 <= row < shape[0] and 0 <= col < shape[1]


def _to_grid_point(point: npt.NDArray[np.float64]) -> GridPoint:
    """Rounds a floating point coordinate to the closest occupancy-grid cell."""
    return (int(round(float(point[0]))), int(round(float(point[1]))))


def _hysteresis_threshold(
    strong_edges: npt.NDArray[np.bool_],
    weak_edges: npt.NDArray[np.bool_],
) -> npt.NDArray[np.bool_]:
    """Links weak edges that are 8-connected to strong edges."""
    output = strong_edges.copy()
    structure = np.ones((3, 3), dtype=bool)

    while True:
        grown = ndimage.binary_dilation(output, structure=structure) & weak_edges
        if not np.any(grown & ~output):
            break
        output |= grown

    return output


def _canny_like_edges_scipy(
    image_uint8: npt.NDArray[np.uint8],
    low_threshold: float,
    high_threshold: float,
    gaussian_sigma: float,
) -> npt.NDArray[np.bool_]:
    """Computes Canny-style edges using SciPy primitives.

    The implementation uses Gaussian smoothing, Sobel gradients and hysteresis.
    """
    smoothed = ndimage.gaussian_filter(image_uint8.astype(np.float64), gaussian_sigma)
    grad_row = ndimage.sobel(smoothed, axis=0, mode="reflect")
    grad_col = ndimage.sobel(smoothed, axis=1, mode="reflect")

    magnitude = np.hypot(grad_row, grad_col)
    max_magnitude = float(np.max(magnitude))
    if max_magnitude <= EPSILON:
        return np.zeros_like(image_uint8, dtype=bool)

    scaled_magnitude = (magnitude / max_magnitude) * 255.0
    strong_edges = scaled_magnitude >= high_threshold
    weak_edges = (scaled_magnitude >= low_threshold) & ~strong_edges

    return _hysteresis_threshold(strong_edges, weak_edges)


def extract_cfree_boundaries(
    grid_obj: MapGrid,
    low_threshold: float = 35.0,
    high_threshold: float = 90.0,
    gaussian_sigma: float = 1.0,
    morphology_kernel_size: int = 3,
    morphology_iterations: int = 1,
) -> npt.NDArray[np.bool_]:
    """Extracts `C_free` boundaries with Canny-style edges and morphology.

    Args:
        grid_obj: Occupancy-grid map.
        low_threshold: Lower edge hysteresis threshold on a `[0, 255]` scale.
        high_threshold: Upper edge hysteresis threshold on a `[0, 255]` scale.
        gaussian_sigma: Gaussian smoothing sigma before edge extraction.
        morphology_kernel_size: Square morphology kernel size.
        morphology_iterations: Number of morphology iterations.

    Returns:
        Binary edge mask with `True` cells on `C_free` boundaries.
    """
    if low_threshold < 0.0 or high_threshold < 0.0:
        raise ValueError("Edge thresholds must be non-negative.")
    if low_threshold >= high_threshold:
        raise ValueError("low_threshold must be lower than high_threshold.")
    if gaussian_sigma <= 0.0:
        raise ValueError("gaussian_sigma must be greater than 0.")
    if morphology_kernel_size <= 0:
        raise ValueError("morphology_kernel_size must be greater than 0.")
    if morphology_iterations < 0:
        raise ValueError("morphology_iterations must be greater than or equal to 0.")

    free_space_mask = grid_obj.grid == FREE_SPACE_VALUE
    free_space_image = np.where(free_space_mask, 255, 0).astype(np.uint8)
    cv2_module = _load_cv2()

    if cv2_module is not None:
        blurred = cv2_module.GaussianBlur(
            free_space_image,
            ksize=(0, 0),
            sigmaX=gaussian_sigma,
            sigmaY=gaussian_sigma,
        )
        edges = cv2_module.Canny(
            blurred,
            threshold1=low_threshold,
            threshold2=high_threshold,
            L2gradient=True,
        )
        boundary_mask = edges > 0
    else:
        boundary_mask = _canny_like_edges_scipy(
            free_space_image,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            gaussian_sigma=gaussian_sigma,
        )

    if morphology_iterations > 0:
        structure = np.ones(
            (morphology_kernel_size, morphology_kernel_size),
            dtype=bool,
        )
        boundary_mask = ndimage.binary_closing(
            boundary_mask,
            structure=structure,
            iterations=morphology_iterations,
        )
        boundary_mask = ndimage.binary_opening(
            boundary_mask,
            structure=structure,
            iterations=morphology_iterations,
        )

    return boundary_mask.astype(bool)


def detect_corners(
    processed_map: npt.NDArray[np.bool_] | npt.NDArray[np.uint8],
    response_k: float = 0.04,
    response_threshold_ratio: float = 0.01,
    gaussian_sigma: float = 1.0,
    nms_window_size: int = 3,
) -> list[GridPoint]:
    """Detects corner candidates from a processed binary map.

    The corner response follows a Harris-style formulation over image gradients.
    """
    if response_k <= 0.0:
        raise ValueError("response_k must be greater than 0.")
    if not 0.0 < response_threshold_ratio <= 1.0:
        raise ValueError("response_threshold_ratio must be in (0, 1].")
    if gaussian_sigma <= 0.0:
        raise ValueError("gaussian_sigma must be greater than 0.")
    if nms_window_size <= 0:
        raise ValueError("nms_window_size must be greater than 0.")

    processed_float = processed_map.astype(np.float64)
    if float(np.max(processed_float)) <= 1.0:
        processed_float *= 255.0

    grad_row = ndimage.sobel(processed_float, axis=0, mode="reflect")
    grad_col = ndimage.sobel(processed_float, axis=1, mode="reflect")

    second_moment_xx = ndimage.gaussian_filter(grad_row * grad_row, gaussian_sigma)
    second_moment_xy = ndimage.gaussian_filter(grad_row * grad_col, gaussian_sigma)
    second_moment_yy = ndimage.gaussian_filter(grad_col * grad_col, gaussian_sigma)

    determinant = (second_moment_xx * second_moment_yy) - (second_moment_xy**2)
    trace = second_moment_xx + second_moment_yy
    corner_response = determinant - (response_k * (trace**2))

    max_response = float(np.max(corner_response))
    if max_response <= EPSILON:
        return []

    threshold = response_threshold_ratio * max_response
    local_maxima = ndimage.maximum_filter(corner_response, size=nms_window_size)

    corner_mask = (
        (corner_response == local_maxima)
        & (corner_response >= threshold)
        & (processed_float > 0.0)
    )

    points = np.argwhere(corner_mask)
    corners: list[GridPoint] = [
        (int(row_index), int(col_index)) for row_index, col_index in points
    ]
    corners.sort()
    return corners


def cluster_points_dbscan(
    points: Sequence[GridPoint],
    eps: float = 2.0,
    min_samples: int = 1,
) -> list[GridPoint]:
    """Clusters grid points with DBSCAN and returns cluster centroids."""
    if eps <= 0.0:
        raise ValueError("eps must be greater than 0.")
    if min_samples <= 0:
        raise ValueError("min_samples must be greater than 0.")

    if not points:
        return []

    point_array = np.asarray(points, dtype=np.float64)
    tree = cKDTree(point_array)

    n_points = len(point_array)
    labels = np.full(n_points, -1, dtype=int)
    visited = np.zeros(n_points, dtype=bool)

    cluster_id = 0
    for point_index in range(n_points):
        if visited[point_index]:
            continue

        visited[point_index] = True
        neighbors = tree.query_ball_point(point_array[point_index], r=eps)
        if len(neighbors) < min_samples:
            continue

        labels[point_index] = cluster_id
        search_queue: deque[int] = deque(neighbors)

        while search_queue:
            neighbor_index = search_queue.popleft()

            if not visited[neighbor_index]:
                visited[neighbor_index] = True
                expanded_neighbors = tree.query_ball_point(
                    point_array[neighbor_index],
                    r=eps,
                )
                if len(expanded_neighbors) >= min_samples:
                    search_queue.extend(expanded_neighbors)

            if labels[neighbor_index] == -1:
                labels[neighbor_index] = cluster_id

        cluster_id += 1

    centroids: list[GridPoint] = []
    for cluster_label in range(cluster_id):
        cluster_points = point_array[labels == cluster_label]
        if cluster_points.size == 0:
            continue
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(
            (int(round(float(centroid[0]))), int(round(float(centroid[1]))))
        )

    # Keep stable deterministic order and remove accidental duplicates.
    return sorted(set(centroids))


def extract_graph_vertices(
    grid_obj: MapGrid,
    low_threshold: float = 35.0,
    high_threshold: float = 90.0,
    gaussian_sigma: float = 1.0,
    morphology_kernel_size: int = 3,
    morphology_iterations: int = 1,
    corner_response_k: float = 0.04,
    corner_response_threshold_ratio: float = 0.01,
    corner_nms_window_size: int = 3,
    dbscan_eps: float = 2.0,
    dbscan_min_samples: int = 1,
) -> list[GridPoint]:
    """Runs edge extraction + corner detection + DBSCAN vertex clustering."""
    boundary_mask = extract_cfree_boundaries(
        grid_obj,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        gaussian_sigma=gaussian_sigma,
        morphology_kernel_size=morphology_kernel_size,
        morphology_iterations=morphology_iterations,
    )
    corner_points = detect_corners(
        boundary_mask,
        response_k=corner_response_k,
        response_threshold_ratio=corner_response_threshold_ratio,
        gaussian_sigma=gaussian_sigma,
        nms_window_size=corner_nms_window_size,
    )
    return cluster_points_dbscan(
        corner_points,
        eps=dbscan_eps,
        min_samples=dbscan_min_samples,
    )


def compute_edt_and_gradient(
    grid_obj: MapGrid,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
]:
    """Computes EDT over `C_free` and returns `D`, `grad_row`, `grad_col`.

    Also returns nearest-obstacle and nearest-free index maps used by corridor
    reconstruction fallback logic.
    """
    free_space_mask = grid_obj.grid == FREE_SPACE_VALUE

    distance_field, nearest_obstacle_indices = ndimage.distance_transform_edt(
        free_space_mask,
        return_indices=True,
    )
    _, nearest_free_indices = ndimage.distance_transform_edt(
        ~free_space_mask,
        return_indices=True,
    )

    grad_row_raw, grad_col_raw = np.gradient(distance_field.astype(np.float64))
    grad_row = np.asarray(grad_row_raw, dtype=np.float64)
    grad_col = np.asarray(grad_col_raw, dtype=np.float64)

    return (
        distance_field.astype(np.float64),
        grad_row.astype(np.float64),
        grad_col.astype(np.float64),
        nearest_obstacle_indices.astype(np.int64),
        nearest_free_indices.astype(np.int64),
    )


def _direction_to_nearest_index(
    point: GridPoint,
    nearest_indices: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64] | None:
    """Builds direction vector from a point to its nearest indexed target."""
    target = np.array(
        [
            float(nearest_indices[0, point[0], point[1]]),
            float(nearest_indices[1, point[0], point[1]]),
        ]
    )
    direction = target - np.array([float(point[0]), float(point[1])])
    return _normalize_vector(direction.astype(np.float64))


def _walk_to_boundary_along_negative_gradient(
    p_obs: GridPoint,
    free_space_mask: npt.NDArray[np.bool_],
    grad_row: npt.NDArray[np.float64],
    grad_col: npt.NDArray[np.float64],
    nearest_obstacle_indices: npt.NDArray[np.int64],
    nearest_free_indices: npt.NDArray[np.int64],
    step_size: float,
    max_steps: int,
) -> GridPoint:
    """Walks from `p_obs` in `-grad D` direction to find `p_c1` boundary point."""
    shape = free_space_mask.shape
    if not _in_bounds(p_obs, shape):
        raise ValueError(f"p_obs={p_obs} must be inside map bounds.")

    current = np.array([float(p_obs[0]), float(p_obs[1])], dtype=np.float64)

    if free_space_mask[p_obs]:
        last_free = p_obs
    else:
        nearest_free_direction = _direction_to_nearest_index(
            p_obs,
            nearest_free_indices,
        )
        if nearest_free_direction is None:
            return p_obs
        seeded_point = current + (nearest_free_direction * step_size)
        seeded_grid_point = _to_grid_point(seeded_point)
        if not _in_bounds(seeded_grid_point, shape):
            return p_obs
        current = seeded_point
        last_free = seeded_grid_point

    previous_direction: npt.NDArray[np.float64] | None = None

    for _ in range(max_steps):
        current_grid_point = _to_grid_point(current)
        if not _in_bounds(current_grid_point, shape):
            return last_free

        if not free_space_mask[current_grid_point]:
            return last_free

        last_free = current_grid_point

        gradient = np.array(
            [
                grad_row[current_grid_point],
                grad_col[current_grid_point],
            ],
            dtype=np.float64,
        )
        direction = _normalize_vector(-gradient)

        if direction is None:
            direction = previous_direction
        if direction is None:
            direction = _direction_to_nearest_index(
                current_grid_point,
                nearest_obstacle_indices,
            )
        if direction is None:
            return last_free

        previous_direction = direction
        current = current + (direction * step_size)

    return last_free


def _walk_along_fixed_direction(
    start: GridPoint,
    direction: npt.NDArray[np.float64],
    free_space_mask: npt.NDArray[np.bool_],
    step_size: float,
    max_steps: int,
) -> GridPoint:
    """Walks from a start point along a fixed direction while staying in `C_free`."""
    unit_direction = _normalize_vector(direction)
    if unit_direction is None:
        return start

    current = np.array([float(start[0]), float(start[1])], dtype=np.float64)
    last_free = start

    for _ in range(max_steps):
        current = current + (unit_direction * step_size)
        current_grid_point = _to_grid_point(current)

        if not _in_bounds(current_grid_point, free_space_mask.shape):
            return last_free
        if not free_space_mask[current_grid_point]:
            return last_free

        last_free = current_grid_point

    return last_free


def compute_blocked_corridor_segment(
    grid_obj: MapGrid,
    p_obs: GridPoint,
    step_size: float = 1.0,
    max_steps: int = 512,
) -> BlockedSegment:
    """Reconstructs blocked corridor segment `S_block` from a dynamic obstacle.

    The procedure follows the article description:
    1. Compute `D` (EDT) and `grad D` over the occupancy map.
    2. From `p_obs`, follow `-grad D` to find boundary point `p_c1`.
    3. From `p_c1`, walk along the gradient normal to find `p_c2`.

    Args:
        grid_obj: Occupancy grid map.
        p_obs: Dynamic obstacle coordinate `(row, col)`.
        step_size: Integration step used in grid walks.
        max_steps: Upper bound for each directional walk.

    Returns:
        Blocked corridor segment as `((p_c1_row, p_c1_col), (p_c2_row, p_c2_col))`.
    """
    if not grid_obj.in_bounds(*p_obs):
        raise ValueError(f"p_obs={p_obs} must be inside map bounds.")
    if step_size <= 0.0:
        raise ValueError("step_size must be greater than 0.")
    if max_steps <= 0:
        raise ValueError("max_steps must be greater than 0.")

    (
        _,
        grad_row,
        grad_col,
        nearest_obstacle_indices,
        nearest_free_indices,
    ) = compute_edt_and_gradient(grid_obj)
    free_space_mask = (grid_obj.grid == FREE_SPACE_VALUE)

    p_c1 = _walk_to_boundary_along_negative_gradient(
        p_obs=p_obs,
        free_space_mask=free_space_mask,
        grad_row=grad_row,
        grad_col=grad_col,
        nearest_obstacle_indices=nearest_obstacle_indices,
        nearest_free_indices=nearest_free_indices,
        step_size=step_size,
        max_steps=max_steps,
    )

    gradient_at_pc1 = np.array(
        [grad_row[p_c1], grad_col[p_c1]],
        dtype=np.float64,
    )
    normalized_gradient = _normalize_vector(gradient_at_pc1)
    if normalized_gradient is None:
        fallback_gradient = np.array(
            [float(p_c1[0] - p_obs[0]), float(p_c1[1] - p_obs[1])],
            dtype=np.float64,
        )
        normalized_gradient = _normalize_vector(fallback_gradient)
    if normalized_gradient is None:
        normalized_gradient = np.array([1.0, 0.0], dtype=np.float64)

    normal_direction = np.array(
        [-normalized_gradient[1], normalized_gradient[0]],
        dtype=np.float64,
    )

    candidate_forward = _walk_along_fixed_direction(
        p_c1,
        normal_direction,
        free_space_mask,
        step_size,
        max_steps,
    )
    candidate_backward = _walk_along_fixed_direction(
        p_c1,
        -normal_direction,
        free_space_mask,
        step_size,
        max_steps,
    )

    forward_distance = float(np.hypot(candidate_forward[0] - p_c1[0], candidate_forward[1] - p_c1[1]))
    backward_distance = float(np.hypot(candidate_backward[0] - p_c1[0], candidate_backward[1] - p_c1[1]))

    p_c2 = candidate_forward if forward_distance >= backward_distance else candidate_backward

    if p_c2 == p_c1:
        p_c2 = _walk_along_fixed_direction(
            p_c1,
            normalized_gradient,
            free_space_mask,
            step_size,
            max_steps,
        )

    return (p_c1, p_c2)


class MapProcessor:
    """High-level orchestrator for map pre-processing and corridor inference."""

    def __init__(
        self,
        grid_obj: MapGrid,
        config: MapProcessingConfig | None = None,
    ) -> None:
        self.grid_obj = grid_obj
        self.config = config if config is not None else MapProcessingConfig()
        _validate_processing_config(self.config)

    def extract_free_space_boundaries(self) -> npt.NDArray[np.bool_]:
        """Extracts `C_free` boundaries with Canny-style + morphology filtering."""
        return extract_cfree_boundaries(
            self.grid_obj,
            low_threshold=self.config.canny_low_threshold,
            high_threshold=self.config.canny_high_threshold,
            gaussian_sigma=self.config.edge_gaussian_sigma,
            morphology_kernel_size=self.config.morphology_kernel_size,
            morphology_iterations=self.config.morphology_iterations,
        )

    def detect_corner_candidates(
        self,
        processed_map: npt.NDArray[np.bool_] | npt.NDArray[np.uint8] | None = None,
    ) -> list[GridPoint]:
        """Detects corner candidates on the supplied map or extracted boundaries."""
        map_to_process = (
            self.extract_free_space_boundaries()
            if processed_map is None
            else processed_map
        )
        return detect_corners(
            map_to_process,
            response_k=self.config.corner_response_k,
            response_threshold_ratio=self.config.corner_response_threshold_ratio,
            gaussian_sigma=self.config.edge_gaussian_sigma,
            nms_window_size=self.config.corner_nms_window_size,
        )

    def cluster_corner_candidates(self, corners: Sequence[GridPoint]) -> list[GridPoint]:
        """Clusters corner candidates with DBSCAN and returns centroids."""
        return cluster_points_dbscan(
            corners,
            eps=self.config.dbscan_eps,
            min_samples=self.config.dbscan_min_samples,
        )

    def extract_graph_vertices(self) -> list[GridPoint]:
        """Runs full vertex extraction pipeline and returns clustered centroids."""
        corners = self.detect_corner_candidates()
        return self.cluster_corner_candidates(corners)

    def build_initial_visibility_graph(self) -> nx.Graph[GridPoint]:
        """Builds the initial LOS visibility graph from extracted centroids."""
        vertices = self.extract_graph_vertices()
        return build_visibility_graph(self.grid_obj, vertices)

    def compute_blocked_segment(self, p_obs: GridPoint) -> BlockedSegment:
        """Computes blocked corridor segment from dynamic obstacle coordinate."""
        return compute_blocked_corridor_segment(
            self.grid_obj,
            p_obs,
            step_size=self.config.gradient_step_size,
            max_steps=self.config.max_walk_steps,
        )
