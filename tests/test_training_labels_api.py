"""API tests for POST /v1/training-labels via FastAPI's TestClient."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.helpers import mask_b64


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def pixel_mask(height, width, points):
    mask = np.zeros((height, width), dtype=bool)
    for y, x in points:
        mask[y, x] = True
    return mask


def body(masks, stride=4, bound=0.5):
    return {
        "masks_b64": [mask_b64(mask) for mask in masks],
        "stride": stride,
        "offset_bound": bound,
    }


def test_feasible_assignment_response_shape(client):
    mask_a = pixel_mask(8, 8, [(0, 3), (0, 4)])
    mask_b = pixel_mask(8, 8, [(0, 0)])
    response = client.post("/v1/training-labels", json=body([mask_a, mask_b]))
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "ok"
    assert data["stride"] == 4
    assert data["grid_size"] == [2, 2]
    assignments = data["assignments"]
    assert [item["mask_index"] for item in assignments] == [0, 1]
    chosen = {item["mask_index"]: item["cell"] for item in assignments}
    assert chosen == {0: [1, 0], 1: [0, 0]}
    # Offsets are signed, in strides, and within the bound.
    for item in assignments:
        ox, oy = item["offset"]
        assert abs(ox) <= 0.5 + 1e-9
        assert abs(oy) <= 0.5 + 1e-9
    # Cell coordinates are zero-based and pairwise distinct.
    cells = [tuple(item["cell"]) for item in assignments]
    assert len(set(cells)) == 2


def test_conflict_response_lists_witness_and_cell_count(client):
    mask = pixel_mask(8, 8, [(0, 0)])
    response = client.post(
        "/v1/training-labels", json=body([mask, mask.copy()], bound=0.5)
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "conflict"
    assert data["available_cell_count"] == 1
    assert len(data["conflict_mask_indices"]) == 2
    assert data["available_cell_count"] < len(data["conflict_mask_indices"])
    # The conflict contract carries only the witness and the cell count.
    assert set(data) == {
        "status",
        "conflict_mask_indices",
        "available_cell_count",
    }
    assert "assignments" not in data


def test_offset_round_trip_through_full_pipeline(client):
    # A 2x2 block centered exactly on cell (1,1)'s center at pixel (6,6).
    mask = pixel_mask(8, 8, [(5, 5), (5, 6), (6, 5), (6, 6)])
    response = client.post(
        "/v1/training-labels", json=body([mask], stride=4, bound=1.0)
    )
    data = response.json()
    assert data["status"] == "ok"
    item = data["assignments"][0]
    assert item["cell"] == [1, 1]
    assert item["offset"] == pytest.approx([0.0, 0.0], abs=1e-12)


def test_stride_8_and_16_are_accepted(client):
    mask = pixel_mask(16, 16, [(8, 8)])
    for stride, expected_cells in ((8, 2), (16, 1)):
        response = client.post(
            "/v1/training-labels",
            json=body([mask], stride=stride, bound=2.0),
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["grid_size"] == [expected_cells, expected_cells]


def test_invalid_stride_is_rejected(client):
    mask = pixel_mask(8, 8, [(0, 0)])
    response = client.post("/v1/training-labels", json=body([mask], stride=3))
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"


def test_offset_bound_range_is_enforced(client):
    mask = pixel_mask(8, 8, [(0, 0)])
    for bad in (0.4, 2.1):
        response = client.post(
            "/v1/training-labels", json=body([mask], bound=bad)
        )
        assert response.status_code == 422


def test_too_many_masks_is_rejected(client):
    masks = [pixel_mask(8, 8, [(0, 0)]) for _ in range(13)]
    response = client.post("/v1/training-labels", json=body(masks))
    assert response.status_code == 422


def test_empty_mask_list_is_rejected(client):
    response = client.post(
        "/v1/training-labels",
        json={"masks_b64": [], "stride": 4, "offset_bound": 1.0},
    )
    assert response.status_code == 422


def test_empty_mask_is_rejected(client):
    response = client.post(
        "/v1/training-labels",
        json=body([np.zeros((8, 8), dtype=bool)]),
    )
    assert response.status_code == 422
    assert "no pixels" in response.json()["detail"]


def test_size_mismatch_between_masks_is_rejected(client):
    response = client.post(
        "/v1/training-labels",
        json=body([pixel_mask(8, 8, [(0, 0)]), pixel_mask(8, 7, [(0, 0)])]),
    )
    assert response.status_code == 422
    assert "share one size" in response.json()["detail"]


def test_side_above_512_is_rejected(client):
    mask = np.ones((513, 8), dtype=bool)
    response = client.post("/v1/training-labels", json=body([mask]))
    assert response.status_code == 422
    assert "512" in response.json()["detail"]


def test_exactly_512_side_is_accepted(client):
    mask = np.zeros((512, 512), dtype=bool)
    mask[256, 256] = True
    response = client.post(
        "/v1/training-labels", json=body([mask], stride=16, bound=2.0)
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "ok"
    assert data["grid_size"] == [32, 32]


def test_corrupt_mask_payload_is_rejected(client):
    response = client.post(
        "/v1/training-labels",
        json={
            "masks_b64": ["bm90IGFuIGltYWdl"],
            "stride": 4,
            "offset_bound": 1.0,
        },
    )
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_input"
