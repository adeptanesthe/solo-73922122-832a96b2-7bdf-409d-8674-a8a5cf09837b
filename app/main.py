"""FastAPI application for shelf-instance compositing."""

from __future__ import annotations

import logging

import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import geometry, grid_labels
from .config import MAX_LABEL_SIDE, MAX_REQUEST_BYTES
from .errors import InvalidInput
from .schemas import (
    BaseInstanceSummary,
    ComposeRequest,
    ComposeResponse,
    InfeasibleResponse,
    InstanceResult,
    LabelAssignment,
    LabelConflictResponse,
    TrainingLabelRequest,
    TrainingLabelResponse,
)

logger = logging.getLogger("shelf_compositor")

app = FastAPI(
    title="Shelf Instance Compositor",
    version="1.0.0",
    description=(
        "Composites masked product patches onto a shelf base image in back-to-front "
        "order, choosing candidate placements that cover the fewest annotated base "
        "pixels while every instance keeps at least half of its original mask area."
    ),
)


@app.middleware("http")
async def _limit_body(request: Request, call_next):
    """Reject oversized bodies before they are buffered into memory.

    The declared ``Content-Length`` is checked first; chunked uploads without
    that header are streamed and counted as well, so a client cannot bypass
    the limit by omitting the header.
    """

    length = request.headers.get("content-length")
    if length is not None:
        try:
            declared = int(length)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "invalid_content_length",
                    "detail": "Content-Length is not an integer",
                },
            )
        if declared < 0:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "invalid_content_length",
                    "detail": "Content-Length is negative",
                },
            )
        if declared > MAX_REQUEST_BYTES:
            return _too_large()

    if length is None:
        total = 0
        chunks: list[bytes] = []
        async for chunk in request.stream():
            total += len(chunk)
            if total > MAX_REQUEST_BYTES:
                return _too_large()
            chunks.append(chunk)
        body = b"".join(chunks)

        async def replay():
            # Starlette's Request.stream() polls receive() until more_body is
            # False; replay the buffered body exactly once.
            if getattr(replay, "sent", False):  # pragma: no cover - defensive
                return {"type": "http.disconnect"}
            replay.sent = True  # type: ignore[attr-defined]
            return {"type": "http.request", "body": body, "more_body": False}

        request = Request(request.scope, receive=replay)
    return await call_next(request)


def _too_large() -> JSONResponse:
    return JSONResponse(
        status_code=413,
        content={
            "error": "payload_too_large",
            "detail": f"request body exceeds {MAX_REQUEST_BYTES} bytes",
        },
    )


def _jsonable_validation_errors(exc: RequestValidationError) -> list[dict]:
    """Make pydantic errors JSON-safe (``ctx`` may hold Exception instances)."""

    cleaned = []
    for error in exc.errors():
        item = {key: value for key, value in error.items() if key != "ctx"}
        if "ctx" in error:
            item["ctx"] = {key: str(value) for key, value in error["ctx"].items()}
        cleaned.append(item)
    return cleaned


@app.exception_handler(InvalidInput)
async def _invalid_input_handler(request: Request, exc: InvalidInput):
    logger.info("rejected request: %s", exc)
    return JSONResponse(
        status_code=422,
        content={"error": "invalid_input", "detail": str(exc)},
    )


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "invalid_request",
            "detail": _jsonable_validation_errors(exc),
        },
    )


@app.get("/health", summary="Liveness probe")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/v1/compose",
    response_model=ComposeResponse,
    summary="Composite product patches onto a shelf base",
    responses={
        200: {
            "description": (
                "A feasible composite, or an {'status': 'infeasible'} report with no "
                "image when no placement keeps half of every instance's mask."
            )
        },
        413: {"description": "Request body above the configured limit."},
        422: {"description": "Malformed or inconsistent input."},
    },
)
def compose(payload: ComposeRequest):
    """Composite patches onto the base and return per-instance visible masks.

    Do not use ``response_model`` overrides here: the infeasible branch must be
    able to answer with a different shape (no image, no instances).
    """

    base = geometry.build_base(payload.base.image_b64, payload.base.instance_masks)

    decoded: list[tuple[object, ...]] = []
    for index, patch in enumerate(payload.patches):
        label = f"patches[{index}]"
        rgb, size = geometry.decode_rgb(label, patch.image_b64)
        mask = geometry.decode_mask(label, patch.mask_b64, size)
        decoded.append((rgb, mask, list(patch.candidates)))

    solution, reason = geometry.solve(base, decoded)
    if solution is None:
        return JSONResponse(
            status_code=200,
            content=InfeasibleResponse(reason=reason or "infeasible").model_dump(),
        )

    instances: list[InstanceResult] = []
    for index, (_, mask, _) in enumerate(decoded):
        original = int(mask.sum())
        visible = solution.visible_pixels[index]
        instances.append(
            InstanceResult(
                patch_index=index,
                position=solution.positions[index],
                visible_pixels=visible,
                original_mask_pixels=original,
                visible_ratio=visible / original,
                visible_mask_b64=geometry.encode_mask(solution.visible_masks[index]),
            )
        )

    base_instances = [
        BaseInstanceSummary(
            base_index=index,
            visible_pixels=solution.base_visible_pixels[index],
            original_mask_pixels=base.original_pixels[index],
            visible_mask_b64=geometry.encode_mask(
                solution.base_visible_masks[index]
            ),
        )
        for index in range(len(base.masks))
    ]

    return ComposeResponse(
        composed_image_b64=geometry.encode_png(solution.composed),
        instances=instances,
        base_instances=base_instances,
        overlapped_base_pixels=solution.overlapped_base_pixels,
        evaluated_combinations=solution.evaluated_combinations,
    )


@app.post(
    "/v1/training-labels",
    response_model=TrainingLabelResponse,
    summary="Assign one feature-grid cell to each masked product",
    responses={
        200: {
            "description": (
                "Either a feasible one-product-per-cell assignment, or a "
                "{'status': 'conflict'} Hall-witness report listing products whose "
                "combined candidate cells are fewer than the products."
            )
        },
        413: {"description": "Request body above the configured limit."},
        422: {"description": "Malformed or inconsistent input."},
    },
)
def training_labels(payload: TrainingLabelRequest):
    """Generate per-product grid-cell labels from foreground masks.

    Like ``/v1/compose``, the conflict branch returns a different shape (the
    witness indices and cell count, no assignments) via a plain JSONResponse
    rather than the declared response model.
    """

    decoded_masks: list = []
    size: tuple[int, int] | None = None
    for index, mask_b64 in enumerate(payload.masks_b64):
        label = f"masks_b64[{index}]"
        image = geometry.decode_b64_image(label, mask_b64)
        width, height = image.size
        if width > MAX_LABEL_SIDE or height > MAX_LABEL_SIDE:
            raise InvalidInput(
                f"{label}: mask is {width}x{height}; each side must be at most "
                f"{MAX_LABEL_SIDE} pixels"
            )
        if size is None:
            size = (width, height)
        elif image.size != size:
            raise InvalidInput(
                f"{label}: mask is {width}x{height} but earlier masks are "
                f"{size[0]}x{size[1]}; all masks must share one size"
            )
        mask = geometry._mask_array(image)
        if not mask.any():
            raise InvalidInput(f"{label}: mask contains no pixels")
        decoded_masks.append(np.ascontiguousarray(mask))

    try:
        result = grid_labels.solve_labels(
            decoded_masks, payload.stride, payload.offset_bound
        )
    except ValueError as exc:
        # Defensive: solve_labels validates shape/emptiness again for callers
        # that bypass the API.
        raise InvalidInput(str(exc)) from exc

    grid_size = [result.grid_width, result.grid_height]
    if not result.feasible:
        # The wire conflict response carries only the witness product indices
        # and their distinct-cell count, per the API contract.
        return JSONResponse(
            status_code=200,
            content=LabelConflictResponse(
                conflict_mask_indices=list(result.conflict_mask_indices),
                available_cell_count=result.conflict_cell_count,
            ).model_dump(),
        )

    assignments = [
        LabelAssignment(
            mask_index=item.mask_index,
            cell=[item.cell.gx, item.cell.gy],
            offset=[item.offset_x, item.offset_y],
        )
        for item in result.assignments
    ]
    return TrainingLabelResponse(
        stride=result.stride,
        grid_size=grid_size,
        assignments=assignments,
    )
