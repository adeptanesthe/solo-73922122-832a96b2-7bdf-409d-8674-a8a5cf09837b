"""Generate training-label requests that demonstrate feature-cell contention.

Usage
-----
1. Start the service:
     docker compose up --build                       # port 8000

2. Run this script:
     python examples/training_labels_demo.py                     # localhost:8000
     python examples/training_labels_demo.py --case conflict    # no-solution case
     python examples/training_labels_demo.py --offline          # write JSON only

The default case (``--case contention``) puts two products into the same
feature cell. Both are eligible for cell (0, 0), but only one product can
occupy it; the solver moves the other product to its second eligible cell and
returns both cells plus signed offsets in strides.

``--case deadlock`` makes both products eligible for exactly one, identical
cell, so no one-to-one assignment exists. The API then answers with
``status: "conflict"`` and a Hall-witness subset: the two product indices and
the single distinct cell they can jointly choose.

Writes ``training_labels_<case>_request.json`` (replayable with curl) and,
when the API is reachable, ``training_labels_<case>_response.json``.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
STRIDE = 4
BOUND = 0.5
HEIGHT = WIDTH = 8


def b64_mask(mask: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def pixel_mask(points: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    for y, x in points:
        mask[y, x] = True
    return mask


def contention_request() -> dict:
    """Two products competing for cell (0, 0) on an 8x8 grid with stride 4.

    Product 0 straddles the x=4 cell boundary symmetrically, so its mean
    foreground center sits exactly on the boundary and both cells (0,0) and
    (1,0) satisfy the 0.5-stride offset bound. Product 1 has a single
    foreground pixel and can only use cell (0,0). The optimum keeps product 1
    in cell (0,0) and moves product 0 to cell (1,0).
    """

    product_0 = pixel_mask([(0, 3), (0, 4)])  # mean center (3.5, 0.5)
    product_1 = pixel_mask([(0, 0)])  # mean center (0.5, 0.5)
    return {
        "masks_b64": [b64_mask(product_0), b64_mask(product_1)],
        "stride": STRIDE,
        "offset_bound": BOUND,
    }


def deadlock_request() -> dict:
    """Two products whose only eligible cell is the same cell (0, 0)."""

    product_0 = pixel_mask([(0, 0)])
    product_1 = pixel_mask([(0, 1)])  # same containing cell, distinct mask
    return {
        "masks_b64": [b64_mask(product_0), b64_mask(product_1)],
        "stride": STRIDE,
        "offset_bound": BOUND,
    }


CASES = {
    "contention": contention_request,
    "deadlock": deadlock_request,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--out", type=Path, default=HERE / "out")
    parser.add_argument("--case", choices=sorted(CASES), default="contention")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Only write the request JSON; do not call the API.",
    )
    args = parser.parse_args(argv)

    request = CASES[args.case]()
    args.out.mkdir(parents=True, exist_ok=True)
    request_path = args.out / f"training_labels_{args.case}_request.json"
    request_path.write_text(json.dumps(request, indent=2))
    print(f"wrote {request_path} ({request_path.stat().st_size} bytes)")
    if args.offline:
        return 0

    http_request = urllib.request.Request(
        f"{args.base_url}/v1/training-labels",
        data=json.dumps(request).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(http_request, timeout=60) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        print("HTTP", exc.code, exc.read()[:2000], file=sys.stderr)
        return 1

    response_path = args.out / f"training_labels_{args.case}_response.json"
    response_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {response_path}")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
