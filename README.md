# Shelf Instance Compositor

FastAPI service that pastes up to four masked commodity patches onto a shelf
base image to generate occlusion-rich instance-segmentation training samples.

Given a shelf base image (with optional non-overlapping instance masks for
already-annotated products) and up to four product cut-outs, each with a mask
and up to five candidate top-left positions, the service:

1. tries every candidate combination (at most `5^4 = 625`),
2. discards combinations where any instance would keep less than **half of its
   original mask area** after clipping to the canvas and being covered by
   later patches,
3. among the survivors, picks the one covering the **fewest annotated base
   pixels** (union, not sum; ties prefer the largest visibility margin, then
   candidate order),
4. returns the composite, each instance's canvas-sized visible mask, and the
   chosen position.

If no combination is feasible, it returns `{"status": "infeasible", "reason": ...}`
and no image.

## Training-label endpoint

`POST /v1/training-labels` turns up to twelve same-sized, non-empty foreground
masks into per-product feature-grid labels. It fixes the "two products land in
the same feature cell and one label gets dropped" problem by assigning at most
one product per cell while minimising every product's offset from its cell
center.

Request:

```jsonc
{
  // Base64 8-bit mask PNGs, the same non-zero-equals-foreground format as
  // /v1/compose masks. All masks share one pixel grid; each side <= 512 px
  // and every mask must contain at least one foreground pixel. 1..12 masks.
  "masks_b64": ["<base64 mask PNG>", "..."],
  "stride": 4,                      // cell side in pixels: 4, 8 or 16
  "offset_bound": 0.5               // per-axis limit in strides: 0.5..2
}
```

Rules:

- The grid starts at the top-left corner. A strip shorter than one stride at
  an edge still forms a cell, whose center is the *full-cell* center:
  cell `(gx, gy)` is centered at `((gx + 0.5) * stride, (gy + 0.5) * stride)`.
- Pixel centers start at `(0.5, 0.5)`; each product's target is the mean of
  its foreground pixel centers.
- A product can only take a cell that contains its own foreground and whose
  center is within `offset_bound` strides of the target on **both** axes.
- Each cell goes to at most one product. The service minimises the sum of
  squared signed x/y offsets (in strides) over all products and returns each
  product's zero-based `[gx, gy]` cell plus its signed `[offset_x, offset_y]`.

Feasible response:

```jsonc
{
  "status": "ok",
  "stride": 4,
  "grid_size": [2, 2],               // [cells_x, cells_y], edge cells counted
  "assignments": [
    {"mask_index": 0, "cell": [1, 0], "offset": [-0.5, -0.375]},
    {"mask_index": 1, "cell": [0, 0], "offset": [-0.375, -0.375]}
  ]
}
```

When not every product can be placed, the response reports only a Hall-witness
subset of products and the number of distinct cells they can jointly choose
(the count is fewer than the number of products, so no assignment can cover
them all):

```jsonc
{
  "status": "conflict",
  "conflict_mask_indices": [0, 1],
  "available_cell_count": 1          // always < len(conflict_mask_indices)
}
```

### API

| Method | Path | Purpose |
| ------ | ---- | ------- |
| `GET`  | `/health` | Liveness probe (`{"status": "ok"}`). |
| `POST` | `/v1/compose` | Composite the patches. |
| `POST` | `/v1/training-labels` | Assign one feature-grid cell per masked product. |
| `GET`  | `/docs` | Interactive OpenAPI docs. |

### Request body

```jsonc
{
  "base": {
    "image_b64": "<base64 PNG/JPEG>",
    // Optional. One mask per pre-annotated shelf instance, same pixel grid as
    // the base, pairwise non-overlapping. Their union is the annotated pixel
    // set the search avoids covering. May be empty.
    "instance_masks": ["<base64 8-bit mask PNG>", "..."]
  },
  "patches": [
    {
      "image_b64": "<base64 PNG/JPEG>",
      "mask_b64": "<base64 8-bit mask PNG, same size as the patch image>",
      // At most five top-left (x, y) positions in canvas coordinates.
      // Negative or overhanging positions are allowed: the patch may stick
      // out of any canvas edge.
      "candidates": [[0, 0], [100, 40], [-12, 300]]
    }
    // ...up to four patches, in back-to-front stacking order
  ]
}
```

Masks are non-zero-equals-instance, must align with their image (no
resampling), and are sent as base64. Data URLs, URL-safe base64 and missing
padding are accepted.

### Response: feasible

```jsonc
{
  "status": "ok",
  "composed_image_b64": "<base64 PNG>",
  "instances": [                       // one per patch, in stacking order
    {
      "patch_index": 0,
      "position": [120, 118],          // chosen top-left of the patch
      "visible_pixels": 1152,
      "original_mask_pixels": 1152,
      "visible_ratio": 1.0,            // always >= 0.5
      "visible_mask_b64": "<base64 canvas-sized 8-bit PNG, 255 = visible>"
    }
  ],
  "base_instances": [                  // visibility of each base annotation
    {"base_index": 0, "visible_pixels": 2304, "original_mask_pixels": 2304,
     "visible_mask_b64": "<base64 PNG>"}
  ],
  "overlapped_base_pixels": 0,         // annotated pixels under pasted instances
  "evaluated_combinations": 300
}
```

The `visible_mask_b64` masks of different instances are pairwise disjoint and
already include both cropping and occlusion, so they can be used directly as
segmentation targets.

### Response: no feasible placement

```jsonc
{"status": "infeasible", "reason": "no candidate combination keeps at least half of every product instance's original mask area"}
```

### Errors

| Status | Meaning |
| ------ | ------- |
| `400`  | Malformed `Content-Length`. |
| `413`  | Body above `MAX_REQUEST_BYTES`. |
| `422`  | Invalid input: bad base64/image, size mismatch, empty mask, overlapping base masks, all candidates outside the canvas, a search space above the node budget, or a training-label request outside the stride/bound/count/512-px limits. Body: `{"error": "invalid_input" \| "invalid_request", "detail": ...}`. |

## Run with Docker Compose

```bash
docker compose up --build          # http://localhost:8000
curl http://localhost:8000/health
```

`HOST_PORT=9000 docker compose up --build` publishes on another port.
Environment overrides (see `app/config.py`):

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `MAX_IMAGE_BYTES` | `4194304` | Max decoded bytes per image/mask. |
| `MAX_IMAGE_PIXELS` | `4194304` | Max pixels per image/mask (2048×2048). |
| `MAX_REQUEST_BYTES` | `33554432` | Max request body. |
| `MAX_BASE_INSTANCES` | `64` | Max base annotation masks. |
| `MIN_VISIBLE_RATIO` | `1/2` | Area floor for pasted instances. |

## Run locally

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --port 8000
pytest -q
```

## Callable examples

`examples/demo_request.py` builds a synthetic 320×240 shelf (two annotated
base instances) plus four non-rectangular product patches with several
candidate positions each (including edge-overhanging ones), POSTs it to the
service, and writes:

- `examples/out/request.json` — the exact request (replayable with `curl`)
- `examples/out/composed.png` — the composite
- `examples/out/instance_<i>_visible.png` — visible mask per instance
- `examples/out/response.json` — the full response

```bash
docker compose up --build -d
pip install numpy pillow
python examples/demo_request.py                       # default localhost:8000
python examples/demo_request.py --base-url http://127.0.0.1:8000 --out examples/out
python examples/demo_request.py --offline             # only write request.json
```

Equivalent `curl` using the generated request:

```bash
curl -sS -X POST http://localhost:8000/v1/compose \
     -H 'Content-Type: application/json' \
     --data @examples/out/request.json > examples/out/response.json
```

`examples/training_labels_demo.py` builds the cell-contention requests for
`/v1/training-labels`. The default `contention` case places two products in
the same cell; the solver moves one to its second eligible cell. `--case
deadlock` produces the `conflict` response:

```bash
python examples/training_labels_demo.py                       # contention, live
python examples/training_labels_demo.py --case deadlock       # Hall-witness report
python examples/training_labels_demo.py --offline             # request JSON only

curl -sS -X POST http://localhost:8000/v1/training-labels \
     -H 'Content-Type: application/json' \
     --data @examples/out/training_labels_contention_request.json
```

## Algorithm

The search is exhaustive because the objective is not separable: a later patch
can hide an earlier one, which changes both the feasibility of the earlier
patch and the set of covered annotated pixels. Greedy per-patch choices can be
suboptimal. The DFS visits patches top-most first while maintaining the union
of already-placed masks, so:

- the visible count of the patch being placed is final at that node (only
  later patches can cover it), enabling immediate pruning below the 50 % floor;
- the incremental covered-annotation count is exact and additive.

At most 625 leaves are visited; per-node work is proportional to patch area.
Ties are broken by the largest minimum visible-area margin, then by candidate
order, making the output deterministic.

## Layout

```
app/
  main.py        FastAPI app, size limiting, error mapping
  schemas.py     Pydantic request/response models and limits
  geometry.py    decoding, exact placement search, compositing, encoding
  grid_labels.py grid cells, mean-pixel-center targets, min-cost cell
                 assignment (Hungarian) and Hall-witness conflict reports
  config.py      limits and the area-floor ratio (env overridable)
  errors.py      shared exception type
tests/
  test_geometry.py          unit tests for masking, clipping, occlusion, objective
  test_api.py               /v1/compose endpoint tests via TestClient
  test_reference.py         differential tests against a naive brute-force reference
  test_grid_labels.py       grid geometry, matching, Hungarian and brute-force fuzz
  test_training_labels_api.py  /v1/training-labels endpoint tests
examples/demo_request.py
examples/training_labels_demo.py
Dockerfile, docker-compose.yml, requirements*.txt
```

## Training-label algorithm

Each mask yields a target (mean foreground pixel center) and a list of
eligible cells: cells that contain mask foreground and whose center is within
the shared offset bound on both axes. The products and cells form a bipartite
graph whose edge cost is the squared per-axis offset in strides.

1. Kuhn's augmenting-path algorithm checks for a matching that covers every
   product. If none exists, an alternating BFS from the unmatched products
   produces a Hall witness: products whose reachable neighbour cells are
   fewer than the products. Those indices and the distinct-cell count are the
   `conflict` response (the count is strictly smaller by construction).
2. Otherwise the rectangular Hungarian algorithm (successive-shortest-path
   potentials) finds the minimum sum-of-squared-offsets one-to-one
   assignment. With at most 12 products and a handful of cells per mask, the
   graphs are small; the optimiser is differential-tested against exhaustive
   permutation enumeration on 400 random cases.

## Limitations

- Patch images are pasted upright; no rotation or scaling is performed.
- Only the advertised limits are enforced; `/v1/compose` images above
  2048×2048 and `/v1/training-labels` masks above 512 px per side are rejected.
- The service is stateless and does not persist requests or outputs.
- Hungarian tie-breaking among equal-cost matchings is deterministic for a
  fixed request (cells are ordered row-major) but is not part of the API
  contract; only the minimum total squared offset is guaranteed.

## Verification status

- `pytest -q` → 372 passed (342 pre-existing compositing tests plus 30 new
  grid/endpoint tests).
- Differential fuzzing: 400 random grid cases compare the Hungarian
  assignment against an independent exhaustive-enumeration reference on
  feasibility, the Hall witness and the optimum cost; zero mismatches. The
  pre-existing 300-case compositing differential suite still passes.
- Live smoke test with `uvicorn`: `GET /health`, both
  `/v1/training-labels` branches (contention resolved, deadlock reported),
  `curl --data @examples/out/training_labels_contention_request.json`,
  OpenAPI including the new schemas, and `/v1/compose` unchanged.
- Worst-case sizing: 12 masks on a 512×512 grid completes well under a second
  (per-mask work is one `np.nonzero` plus a few candidate cells).

Not verified here: `docker compose up --build` itself (Docker is not installed
in this environment); the Dockerfile and compose file were reviewed by hand
and their Python commands were run directly via uvicorn.
