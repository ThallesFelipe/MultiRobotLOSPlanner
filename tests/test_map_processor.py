"""Tests for map pre-processing and blocked-corridor reconstruction."""

import numpy as np

from core.map_grid import MapGrid
from core.map_processor import (
    MapProcessingConfig,
    MapProcessor,
    compute_blocked_corridor_segment,
    extract_cfree_boundaries,
)
from presets.map_catalog import create_map_from_catalog

GridPoint = tuple[int, int]


def test_extract_cfree_boundaries_detects_obstacle_interfaces() -> None:
    """Ensures edge extraction detects boundaries near obstacle contours."""
    grid = MapGrid(20, 20)
    grid.add_obstacle_rect(6, 14, 9, 11)

    boundary_mask = extract_cfree_boundaries(
        grid,
        low_threshold=20.0,
        high_threshold=60.0,
        gaussian_sigma=1.0,
        morphology_kernel_size=3,
        morphology_iterations=1,
    )

    assert boundary_mask.shape == (20, 20)
    assert boundary_mask.dtype == np.bool_
    assert int(np.sum(boundary_mask)) > 0
    assert bool(boundary_mask[5:15, 8:12].any())


def test_map_processor_corner_detection_and_dbscan_returns_centroids() -> None:
    """Checks corner extraction and DBSCAN centroid reduction on a rectangle."""
    grid = MapGrid(30, 30)
    grid.add_obstacle_rect(8, 22, 10, 20)

    processor = MapProcessor(
        grid,
        config=MapProcessingConfig(
            canny_low_threshold=20.0,
            canny_high_threshold=60.0,
            edge_gaussian_sigma=1.0,
            corner_response_threshold_ratio=0.01,
            dbscan_eps=2.5,
            dbscan_min_samples=1,
        ),
    )

    vertices = processor.extract_graph_vertices()

    expected_corners: list[GridPoint] = [
        (8, 10),
        (8, 19),
        (21, 10),
        (21, 19),
    ]

    assert len(vertices) >= 4
    for expected_corner in expected_corners:
        assert any(
            abs(vertex[0] - expected_corner[0]) <= 3
            and abs(vertex[1] - expected_corner[1]) <= 3
            for vertex in vertices
        )
    assert all(grid.is_free(*vertex) for vertex in vertices)


def test_map_processor_projects_bidas_vertices_to_free_space() -> None:
    """Corner centroids used as graph vertices must not remain in obstacles."""
    grid = create_map_from_catalog("bidas")
    vertices = MapProcessor(grid).extract_graph_vertices()

    assert vertices
    assert all(grid.is_free(*vertex) for vertex in vertices)


def test_map_processor_build_initial_visibility_graph_has_weighted_edges() -> None:
    """Verifies graph build uses extracted centroid vertices and edge weights."""
    grid = MapGrid(24, 24)
    grid.add_obstacle_rect(9, 15, 9, 15)

    processor = MapProcessor(
        grid,
        config=MapProcessingConfig(
            canny_low_threshold=20.0,
            canny_high_threshold=60.0,
            edge_gaussian_sigma=1.0,
            corner_response_threshold_ratio=0.01,
            dbscan_eps=2.0,
            dbscan_min_samples=1,
        ),
    )

    graph = processor.build_initial_visibility_graph()

    assert graph.number_of_nodes() > 0
    assert set(graph.nodes) == set(processor.extract_graph_vertices())
    for source_vertex, target_vertex, edge_data in graph.edges(data=True):
        assert source_vertex != target_vertex
        assert float(edge_data["weight"]) > 0.0


def test_compute_blocked_corridor_segment_returns_in_bounds_segment() -> None:
    """Ensures EDT/gradient blocked-corridor reconstruction returns valid segment."""
    grid = MapGrid(30, 30)
    grid.add_obstacle_rect(0, 30, 10, 11)
    grid.add_obstacle_rect(0, 30, 18, 19)

    p_c1, p_c2 = compute_blocked_corridor_segment(
        grid,
        p_obs=(15, 12),
        step_size=1.0,
        max_steps=200,
    )

    assert grid.in_bounds(*p_c1)
    assert grid.in_bounds(*p_c2)
    assert p_c1 != p_c2
    assert abs(p_c1[1] - p_c2[1]) <= 2
    assert abs(p_c1[0] - p_c2[0]) >= 6
