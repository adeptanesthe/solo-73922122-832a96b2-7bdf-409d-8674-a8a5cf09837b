"""Example request/response captured from a live run, plus how to replay it.

Run ``python examples/demo_request.py --offline`` to regenerate
``request.json`` without a server, then POST it with curl:

    curl -sS -X POST http://localhost:8000/v1/compose \
         -H 'Content-Type: application/json' \
         --data @examples/request.json

Response shape (trimmed; masks and the image are elided):

    {
      "status": "ok",
      "instances": [
        {"patch_index": 0, "position": [120, 118], "visible_pixels": 1152,
         "original_mask_pixels": 1152, "visible_ratio": 1.0,
         "visible_mask_b64": "..."}
      ],
      "base_instances": [
        {"base_index": 0, "visible_pixels": 2304,
         "original_mask_pixels": 2304, "visible_mask_b64": "..."}
      ],
      "overlapped_base_pixels": 0,
      "evaluated_combinations": 300
    }

The infeasible branch returns ``{"status": "infeasible", "reason": "..."}``
with no image, for example when every candidate clips a patch below half of
its mask area.

## Training labels

Generate the cell-contention request and POST it to ``/v1/training-labels``:

    python examples/training_labels_demo.py --offline
    curl -sS -X POST http://localhost:8000/v1/training-labels \
         -H 'Content-Type: application/json' \
         --data @examples/out/training_labels_contention_request.json

The contention case (two products eligible for cell ``[0, 0]``; product 0 is
also eligible for ``[1, 0]``) answers:

    {
      "status": "ok",
      "stride": 4,
      "grid_size": [2, 2],
      "assignments": [
        {"mask_index": 0, "cell": [1, 0], "offset": [-0.5, -0.375]},
        {"mask_index": 1, "cell": [0, 0], "offset": [-0.375, -0.375]}
      ]
    }

Offsets are signed, in strides, from the cell center to each mask's mean
foreground-pixel center. When no one-to-one assignment exists (``--case deadlock``) the service
instead reports only the Hall witness and its distinct-cell count:

    {
      "status": "conflict",
      "conflict_mask_indices": [0, 1],
      "available_cell_count": 1
    }

``available_cell_count`` is always smaller than the number of witness indices.
"""
