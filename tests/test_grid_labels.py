"""Unit tests for the training-label grid assignment core."""

from __future__ import annotations

import itertools
import random

import numpy as np
import pytest

from app import grid_labels as gl
from app.errors import InvalidInput  # noqa: F401  (ensures error module imports)


def single_pixel(height, width, y, x):
    mask = np.zeros((height, width), dtype=bool)
    mask[y, x] = True
    return mask


# --------------------------------------------------------------------------- #
# Grid geometry
# --------------------------------------------------------------------------- #


def test_grid_dimensions_count_partial_edge_cells():
    assert gl.grid_dimensions(8, 8, 4) == (2, 2)
    assert gl.grid_dimensions(9, 6, 4) == (3, 2)
    assert gl.grid_dimensions(1, 1, 16) == (1, 1)
    assert gl.grid_dimensions(512, 512, 16) == (32, 32)


def test_edge_cell_keeps_full_cell_center():
    # 6x6 image, stride 4: the last cell (1, 1) is a 2x2 partial cell but its
    # center stays at the full-cell position (6, 6).
    mask = single_pixel(6, 6, 5, 5)
    cells, target = gl.candidate_cells(mask, 4, 2.0)
    assert target == (5.5, 5.5)
    by_coord = {(c.gx, c.gy): c for c in cells}
    assert (1, 1) in by_coord
    assert (by_coord[1, 1].center_x, by_coord[1, 1].center_y) == (6.0, 6.0)


def test_pixel_centers_start_at_half_pixel():
    mask = single_pixel(8, 8, 0, 0)
    assert gl.mask_target(mask) == (0.5, 0.5)
    mask = single_pixel(8, 8, 3, 5)
    assert gl.mask_target(mask) == (5.5, 3.5)


def test_target_is_mean_of_foreground_pixel_centers():
    mask = np.zeros((8, 8), dtype=bool)
    mask[0, 1] = True
    mask[0, 3] = True
    # x centers 1.5 and 3.5 -> mean 2.5; y center 0.5.
    assert gl.candidate_cells(mask, 4, 2.0)[1] == (2.5, 0.5)


def test_cells_must_contain_own_foreground():
    # Foreground only in cell (1, 1); cell (0, 0) must never be offered even
    # though its center is within the bound of a distant target.
    mask = single_pixel(8, 8, 6, 6)
    cells, _ = gl.candidate_cells(mask, 4, 2.0)
    assert [(c.gx, c.gy) for c in cells] == [(1, 1)]


def test_offset_bound_filters_cells_on_both_axes():
    # Mask straddling the x boundary between cells 0 and 1 at y=0.
    mask = np.zeros((8, 8), dtype=bool)
    mask[0, 3] = True
    mask[0, 4] = True
    # Target (3.5, 0.5). Cell (1,0) center (6,2): x offset -0.625 strides.
    # A tight bound therefore excludes it; the cell's y offset is the same as
    # cell 0's, so this exercises the x-axis cut specifically.
    cells, _ = gl.candidate_cells(mask, 4, 0.25)
    assert [(c.gx, c.gy) for c in cells] == []
    cells, _ = gl.candidate_cells(mask, 4, 0.75)
    assert [(c.gx, c.gy) for c in cells] == [(0, 0), (1, 0)]


def test_y_axis_cut_is_independent():
    # Foreground in cell (0,0) only, but far enough down that a tight y bound
    # rejects the containing cell.
    mask = single_pixel(8, 8, 3, 1)  # target (1.5, 3.5); cell center (2, 2)
    cells, _ = gl.candidate_cells(mask, 4, 0.25)
    assert cells == ()
    cells, _ = gl.candidate_cells(mask, 4, 0.5)
    assert [(c.gx, c.gy) for c in cells] == [(0, 0)]


def test_boundary_offset_equal_to_bound_is_allowed():
    mask = np.zeros((8, 8), dtype=bool)
    mask[0, 3] = True
    mask[0, 4] = True
    # Symmetric target on the boundary between the two cells: |offset| = 0.5.
    cells, _ = gl.candidate_cells(mask, 4, 0.5)
    # Cell 1 has offset exactly -0.5 on x, which the inclusive bound accepts.
    assert [(c.gx, c.gy) for c in cells] == [(0, 0), (1, 0)]


# --------------------------------------------------------------------------- #
# Assignment outcomes
# --------------------------------------------------------------------------- #


def test_single_mask_zero_offset_when_target_is_cell_center():
    # 2x2 block centered exactly on cell (1,1)'s full-cell center at (6, 6).
    mask = np.zeros((8, 8), dtype=bool)
    mask[5:7, 5:7] = True
    result = gl.solve_labels([mask], 4, 1.0)
    assert result.feasible
    assignment = result.assignments[0]
    assert (assignment.cell.gx, assignment.cell.gy) == (1, 1)
    assert assignment.offset_x == pytest.approx(0.0)
    assert assignment.offset_y == pytest.approx(0.0)


def test_contention_moves_one_product_to_its_second_cell():
    # Product A straddles the boundary: eligible for cells (0,0) and (1,0).
    mask_a = np.zeros((8, 8), dtype=bool)
    mask_a[0, 3] = True
    mask_a[0, 4] = True
    # Product B can only use cell (0,0).
    mask_b = single_pixel(8, 8, 0, 0)
    result = gl.solve_labels([mask_a, mask_b], 4, 0.5)
    assert result.feasible
    chosen = {a.mask_index: (a.cell.gx, a.cell.gy) for a in result.assignments}
    assert chosen == {0: (1, 0), 1: (0, 0)}
    # Cells are pairwise distinct.
    cells = [chosen[i] for i in range(2)]
    assert len(set(cells)) == 2


def test_solution_minimises_sum_of_squared_offsets():
    # Both masks span cells (0,0) and (1,0) and, with bound 1.0, both cells are
    # eligible for both products; they only differ in where their lean sits.
    mask_left = np.zeros((8, 8), dtype=bool)
    mask_left[0, 0] = True
    mask_left[0, 4] = True  # target x = 2.0, i.e. the cell-0 center line
    mask_right = np.zeros((8, 8), dtype=bool)
    mask_right[0, 3] = True
    mask_right[0, 7] = True  # target x = 5.0, i.e. the cell-1 center line
    result = gl.solve_labels([mask_left, mask_right], 4, 1.0)
    assert result.feasible
    chosen = {a.mask_index: (a.cell.gx, a.cell.gy) for a in result.assignments}
    # Each product takes its zero-x-offset cell; crossing would be costlier.
    assert chosen == {0: (0, 0), 1: (1, 0)}
    total = sum(a.offset_x**2 + a.offset_y**2 for a in result.assignments)
    costs = {}
    for i, mask in enumerate((mask_left, mask_right)):
        cells, (tx, ty) = gl.candidate_cells(mask, 4, 1.0)
        costs[i] = {
            (c.gx, c.gy): ((tx - c.center_x) / 4) ** 2 + ((ty - c.center_y) / 4) ** 2
            for c in cells
        }
    crossed = costs[0][(1, 0)] + costs[1][(0, 0)]
    natural = costs[0][(0, 0)] + costs[1][(1, 0)]
    assert total == pytest.approx(natural)
    assert total < crossed


def test_conflict_reports_hall_witness_with_fewer_cells():
    mask = single_pixel(8, 8, 0, 0)
    result = gl.solve_labels([mask, mask.copy()], 4, 0.5)
    assert not result.feasible
    assert sorted(result.conflict_mask_indices) == [0, 1]
    assert result.conflict_cell_count == 1
    assert result.conflict_cell_count < len(result.conflict_mask_indices)


def test_conflict_witness_subset_can_be_smaller_than_all_products():
    # Products 0 and 1 deadlock on one cell; product 2 has its own cell.
    deadlock = single_pixel(8, 8, 0, 0)
    independent = single_pixel(8, 8, 0, 7)
    result = gl.solve_labels([deadlock, deadlock.copy(), independent], 4, 0.5)
    assert not result.feasible
    witness = set(result.conflict_mask_indices)
    # The witness only needs to cover the deadlocking pair.
    assert witness <= {0, 1, 2}
    assert result.conflict_cell_count < len(witness)
    # The reported cells are exactly the union of the witness's candidates.
    witness_cells = set()
    for index in witness:
        witness_cells.update((c.gx, c.gy) for c in result.candidates[index])
    assert result.conflict_cell_count == len(witness_cells)


def test_product_with_no_eligible_cell_is_its_own_witness():
    # A single foreground pixel whose only containing cell breaks the bound.
    mask = single_pixel(8, 8, 0, 0)
    result = gl.solve_labels([mask], 4, 0.25)
    assert not result.feasible
    assert result.conflict_mask_indices == (0,)
    assert result.conflict_cell_count == 0


def test_twelve_masks_each_get_distinct_cells():
    # One pixel per column pair in a 48x8 image, stride 4 -> 12 columns.
    masks = []
    for i in range(12):
        masks.append(single_pixel(8, 48, 4, i * 4 + 2))
    result = gl.solve_labels(masks, 4, 1.0)
    assert result.feasible
    cells = [(a.cell.gx, a.cell.gy) for a in result.assignments]
    assert len(set(cells)) == 12
    assert [a.mask_index for a in result.assignments] == list(range(12))


def test_empty_and_mismatched_masks_raise():
    with pytest.raises(ValueError, match="no pixels"):
        gl.solve_labels([np.zeros((4, 4), dtype=bool)], 4, 1.0)
    with pytest.raises(ValueError, match="share one size"):
        gl.solve_labels(
            [single_pixel(4, 4, 0, 0), single_pixel(5, 5, 0, 0)], 4, 1.0
        )


# --------------------------------------------------------------------------- #
# Differential fuzzing against an independent brute force
# --------------------------------------------------------------------------- #


def _brute_force(masks, stride, bound):
    """Enumerate every injective assignment; return feasibility and min cost."""

    result = gl.solve_labels(masks, stride, bound)
    options = []
    for cells in result.candidates:
        options.append([(c.gx, c.gy) for c in cells])
    if any(not option for option in options):
        assert not result.feasible
        return
    best = None
    for combo in itertools.product(*options):
        if len(set(combo)) != len(combo):
            continue
        total = 0.0
        for index, coord in enumerate(combo):
            cells = result.candidates[index]
            cell = next(c for c in cells if (c.gx, c.gy) == coord)
            target_x, target_y = gl.mask_target(masks[index])
            ox = (target_x - cell.center_x) / stride
            oy = (target_y - cell.center_y) / stride
            total += ox * ox + oy * oy
        if best is None or total < best:
            best = total
    if best is None:
        assert not result.feasible
        # Hall check: the witness's neighbour union really is too small.
        witness = result.conflict_mask_indices
        union = set()
        for index in witness:
            union.update(options[index])
        assert len(union) < len(witness)
        return
    assert result.feasible
    got = sum(a.offset_x**2 + a.offset_y**2 for a in result.assignments)
    assert got == pytest.approx(best, rel=1e-12, abs=1e-12)
    assert len({(a.cell.gx, a.cell.gy) for a in result.assignments}) == len(masks)


def test_matches_brute_force_on_random_cases():
    rng = random.Random(20261122)
    for _ in range(400):
        height = rng.randint(1, 14)
        width = rng.randint(1, 14)
        stride = rng.choice((4, 8, 16))
        bound = rng.choice((0.5, 0.75, 1.0, 1.5, 2.0))
        n_masks = rng.randint(1, 6)
        masks = []
        for _ in range(n_masks):
            mask = np.zeros((height, width), dtype=bool)
            for y in range(height):
                for x in range(width):
                    if rng.random() < 0.35:
                        mask[y, x] = True
            if not mask.any():
                mask[rng.randrange(height), rng.randrange(width)] = True
            masks.append(mask)
        _brute_force(masks, stride, bound)
