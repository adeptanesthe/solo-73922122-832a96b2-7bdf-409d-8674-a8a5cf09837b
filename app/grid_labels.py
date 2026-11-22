"""Training-label generation: assign one feature-grid cell to each mask.

Semantics
---------
The image is divided into cells of side ``stride`` starting at the top-left
corner. A cell is identified by zero-based coordinates ``(gx, gy)``; its
*center* is the center of a full cell even when the image ends mid-cell, so a
cell near the right or bottom edge has the same center it would have on an
infinite grid::

    center_x = (gx + 0.5) * stride
    center_y = (gy + 0.5) * stride

For every non-empty mask the target point is the unweighted mean of its
foreground pixel *centers*. Pixel centers start at ``(0.5, 0.5)``, so a
foreground pixel at integer index ``(x, y)`` contributes ``(x + 0.5, y + 0.5)``.

A product may only be placed in a cell that (a) contains at least one of its
own foreground pixels and (b) whose center is within the shared offset bound
on both axes, i.e. the absolute per-axis offset between cell center and target
is at most ``offset_bound`` strides. Each cell can be assigned to at most one
product. Among all one-to-one assignments we minimise the sum of squared
signed offsets of all products (offsets measured in strides). When no
assignment covers every product, we report a Hall-witness subset of products
whose combined candidate cells are fewer than the products themselves.

Algorithms
----------
* Existence is checked with Kuhn's augmenting-path algorithm for maximum
  bipartite matching. When the matching is not perfect, an alternating BFS
  from every unmatched product yields the standard Hall-violation witness:
  the reached products' neighbour set is strictly smaller than the set
  itself.
* The optimum is the rectangular Hungarian algorithm (successive-shortest
  augmenting paths on potentials), run on the union of all candidate cells.
  Infeasible product/cell pairs get a sentinel cost far above any feasible
  total, so a perfect feasible matching is chosen whenever one exists.

There are at most 12 products and each mask reaches at most a handful of
cells on a grid of side <= 512, so these small graph algorithms are ample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# Infeasible pairs in the Hungarian cost matrix. Feasible squared offsets are
# in [0, 8] per pair and the total over <= 12 products is below 96, so any
# value above that works; the large constant also keeps potentials stable.
_INFEASIBLE_COST = 1e12

# Absolute tolerance (in strides) when comparing an offset with the bound, to
# absorb float64 round-off from dividing pixel means by the stride.
_BOUND_EPS = 1e-9


@dataclass(frozen=True)
class Cell:
    """One feature-grid cell and its pixel-space center."""

    gx: int
    gy: int
    center_x: float
    center_y: float


@dataclass(frozen=True)
class Assignment:
    """One assigned product: its cell and the signed offset (in strides)."""

    mask_index: int
    cell: Cell
    offset_x: float  # (target_x - center_x) / stride
    offset_y: float


@dataclass(frozen=True)
class LabelResult:
    """Outcome of the assignment for one request."""

    feasible: bool
    stride: int
    grid_width: int
    grid_height: int
    assignments: tuple[Assignment, ...] = ()
    # Hall witness, only populated when feasible is False.
    conflict_mask_indices: tuple[int, ...] = ()
    conflict_cell_count: int = 0
    # Every product's candidate cells (shared cell objects per coordinates),
    # kept on the result so callers and tests can audit the geometry.
    candidates: tuple[tuple[Cell, ...], ...] = field(default_factory=tuple)


def grid_dimensions(width: int, height: int, stride: int) -> tuple[int, int]:
    """Number of grid cells along x and y, counting a partial edge cell."""

    return (width + stride - 1) // stride, (height + stride - 1) // stride


def cell_center(gx: int, gy: int, stride: int) -> tuple[float, float]:
    """Full-cell center for ``(gx, gy)``, identical for partial edge cells."""

    return (gx + 0.5) * stride, (gy + 0.5) * stride


def mask_target(mask: np.ndarray) -> tuple[float, float]:
    """Mean of the foreground pixel centers; ``mask`` must be non-empty."""

    rows, cols = np.nonzero(mask)
    # Pixel (x, y) has center (x + 0.5, y + 0.5). Sum rather than mean to
    # avoid keeping both index arrays alive longer than necessary.
    count = rows.size
    target_x = float(cols.sum(dtype=np.float64)) / count + 0.5
    target_y = float(rows.sum(dtype=np.float64)) / count + 0.5
    return target_x, target_y


def _occupied_cells(mask: np.ndarray, stride: int) -> set[tuple[int, int]]:
    """Grid coordinates of the cells containing at least one foreground pixel.

    A reshape-based OR reduction keeps this to a handful of numpy calls even
    for dense 512x512 masks. Partial edge cells get zero padding so they have
    the same stride x stride block shape as interior cells.
    """

    height, width = mask.shape
    grid_width, grid_height = grid_dimensions(width, height, stride)
    padded = np.zeros((grid_height * stride, grid_width * stride), dtype=bool)
    padded[:height, :width] = mask
    blocks = padded.reshape(grid_height, stride, grid_width, stride)
    occupied = blocks.any(axis=(1, 3))
    return {(int(gx), int(gy)) for gy, gx in np.argwhere(occupied)}


def candidate_cells(
    mask: np.ndarray, stride: int, offset_bound: float
) -> tuple[tuple[Cell, ...], tuple[float, float]]:
    """Cells that contain mask foreground and respect the offset bound.

    Returns the ordered candidate cells (row-major) and the mask target.
    """

    height, width = mask.shape
    grid_width, grid_height = grid_dimensions(width, height, stride)
    target_x, target_y = mask_target(mask)

    cells: list[Cell] = []
    occupied = _occupied_cells(mask, stride)
    for gy in range(grid_height):
        for gx in range(grid_width):
            if (gx, gy) not in occupied:
                continue
            center_x, center_y = cell_center(gx, gy, stride)
            offset_x = (target_x - center_x) / stride
            offset_y = (target_y - center_y) / stride
            if (
                abs(offset_x) <= offset_bound + _BOUND_EPS
                and abs(offset_y) <= offset_bound + _BOUND_EPS
            ):
                cells.append(Cell(gx=gx, gy=gy, center_x=center_x, center_y=center_y))
    return tuple(cells), (target_x, target_y)


# --------------------------------------------------------------------------- #
# Maximum matching and the Hall witness
# --------------------------------------------------------------------------- #


def _maximum_matching(
    adjacency: Sequence[Sequence[int]],
) -> tuple[list[int], list[int]]:
    """Kuhn's algorithm. Returns ``(match_left, match_right)`` with -1 gaps."""

    n_left = len(adjacency)
    n_right = max(
        (max(neighbours, default=-1) + 1 for neighbours in adjacency), default=0
    )
    match_left = [-1] * n_left
    match_right = [-1] * n_right

    def augment(left: int, seen: list[bool]) -> bool:
        for right in adjacency[left]:
            if seen[right]:
                continue
            seen[right] = True
            if match_right[right] == -1 or augment(match_right[right], seen):
                match_right[right] = left
                match_left[left] = right
                return True
        return False

    for left in range(n_left):
        augment(left, [False] * n_right)
    return match_left, match_right


def _hall_witness(
    adjacency: Sequence[Sequence[int]], match_right: Sequence[int]
) -> tuple[tuple[int, ...], int]:
    """Subset of left vertices whose neighbours are fewer than the subset.

    Alternating BFS from every unmatched left vertex over edges that are
    non-matching left-to-right and matching right-to-left. The reached left
    vertices are the Hall witness; their reachable neighbours are exactly the
    reached right vertices and count fewer than the left vertices.
    """

    n_right = len(match_right)
    reached_left: set[int] = set()
    reached_right: set[int] = set()
    # Unmatched left vertices are the BFS seeds for the alternating search.
    unmatched_left = [
        left
        for left in range(len(adjacency))
        if all(match_right[right] != left for right in range(n_right))
    ]
    queue = list(unmatched_left)
    for left in queue:
        reached_left.add(left)

    head = 0
    while head < len(queue):
        left = queue[head]
        head += 1
        for right in adjacency[left]:
            if right in reached_right:
                continue
            reached_right.add(right)
            owner = match_right[right]
            if owner != -1 and owner not in reached_left:
                reached_left.add(owner)
                queue.append(owner)

    witness = tuple(sorted(reached_left))
    # The union of candidate cells of the witness is exactly reached_right.
    return witness, len(reached_right)


# --------------------------------------------------------------------------- #
# Hungarian algorithm for the minimum-cost perfect matching
# --------------------------------------------------------------------------- #


def _hungarian(cost: np.ndarray) -> tuple[float, list[int]]:
    """Rectangular Hungarian method for an ``n x m`` matrix with ``n <= m``.

    Returns the minimum cost and, for each row ``i``, the assigned column.
    This is the canonical potential-based successive-shortest-path
    formulation, adapted to floating-point costs.
    """

    n, m = cost.shape
    u = np.zeros(n + 1, dtype=np.float64)  # row potentials (rows 1..n)
    v = np.zeros(m + 1, dtype=np.float64)  # column potentials (cols 1..m)
    p = np.zeros(m + 1, dtype=np.int64)  # p[j]: row currently matched to col j
    way = np.zeros(m + 1, dtype=np.int64)  # way[j]: predecessor column on path

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf, dtype=np.float64)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = int(p[j0])
            row = cost[i0 - 1]
            delta = np.inf
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = float(row[j - 1] - u[i0] - v[j])
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[int(p[j])] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        # Flip matches along the augmenting path.
        while j0 != 0:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1

    assigned = [-1] * n
    for j in range(1, m + 1):
        if p[j] != 0:
            assigned[int(p[j]) - 1] = j - 1
    total = float(sum(cost[i, assigned[i]] for i in range(n)))
    return total, assigned


def solve_labels(
    masks: Sequence[np.ndarray], stride: int, offset_bound: float
) -> LabelResult:
    """Build cells/targets and optimally assign one cell per mask.

    Raises ``ValueError`` on empty masks or inconsistent shapes; the API
    layer translates those into 422 responses.
    """

    if not masks:
        raise ValueError("at least one mask is required")
    height, width = masks[0].shape
    for index, mask in enumerate(masks):
        if mask.shape != (height, width):
            raise ValueError(f"mask {index}: all masks must share one size")
        if not mask.any():
            raise ValueError(f"mask {index}: mask contains no pixels")

    grid_width, grid_height = grid_dimensions(width, height, stride)

    per_product_cells: list[tuple[Cell, ...]] = []
    targets: list[tuple[float, float]] = []
    for mask in masks:
        cells, target = candidate_cells(mask, stride, offset_bound)
        per_product_cells.append(cells)
        targets.append(target)

    # Shared cell registry over the union of candidate cells.
    registry: dict[tuple[int, int], Cell] = {}
    for cells in per_product_cells:
        for cell in cells:
            registry.setdefault((cell.gx, cell.gy), cell)
    ordered_cells = sorted(registry.values(), key=lambda c: (c.gy, c.gx))
    column_of = {(cell.gx, cell.gy): idx for idx, cell in enumerate(ordered_cells)}

    adjacency: list[list[int]] = [
        [column_of[(cell.gx, cell.gy)] for cell in cells]
        for cells in per_product_cells
    ]

    match_left, match_right = _maximum_matching(adjacency)
    if any(right == -1 for right in match_left):
        witness, cell_count = _hall_witness(adjacency, match_right)
        return LabelResult(
            feasible=False,
            stride=stride,
            grid_width=grid_width,
            grid_height=grid_height,
            conflict_mask_indices=witness,
            conflict_cell_count=cell_count,
            candidates=tuple(per_product_cells),
        )

    n = len(masks)
    m = len(ordered_cells)
    cost = np.full((n, m), _INFEASIBLE_COST, dtype=np.float64)
    for i, cells in enumerate(per_product_cells):
        tx, ty = targets[i]
        for cell in cells:
            j = column_of[(cell.gx, cell.gy)]
            ox = (tx - cell.center_x) / stride
            oy = (ty - cell.center_y) / stride
            cost[i, j] = ox * ox + oy * oy

    _, assigned_columns = _hungarian(cost)
    assignments: list[Assignment] = []
    for i, column in enumerate(assigned_columns):
        cell = ordered_cells[column]
        tx, ty = targets[i]
        assignments.append(
            Assignment(
                mask_index=i,
                cell=cell,
                offset_x=(tx - cell.center_x) / stride,
                offset_y=(ty - cell.center_y) / stride,
            )
        )

    return LabelResult(
        feasible=True,
        stride=stride,
        grid_width=grid_width,
        grid_height=grid_height,
        assignments=tuple(assignments),
        candidates=tuple(per_product_cells),
    )
