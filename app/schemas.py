"""Request/response schemas.

The wire format is JSON with base64-encoded image payloads. Every image
carries an explicit pixel mask so placements are evaluated on exact masks
rather than on alpha thresholds. Masks must be stored-mode, i.e. exactly the
same width and height as their image; there is no resampling.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .config import (
    LABEL_OFFSET_BOUND_MAX,
    LABEL_OFFSET_BOUND_MIN,
    LABEL_STRIDES,
    MAX_CANDIDATES,
    MAX_LABEL_MASKS,
    MAX_PATCHES,
)


_B64_HELP = "Base64 image payload. Data URLs and missing padding are accepted."


class BaseImage(BaseModel):
    """Shelf base image with its pre-annotated, non-overlapping instance masks."""

    image_b64: str = Field(..., description=_B64_HELP)
    instance_masks: list[str] = Field(
        default_factory=list,
        description=(
            "One base64 mask per annotated shelf instance, each aligned "
            "pixel-for-pixel with the base image and pairwise non-overlapping. "
            "Their pixels are the annotated pixels the placement search tries "
            "hardest not to cover. May be empty when the base has no annotations."
        ),
    )


class Patch(BaseModel):
    """One commodity cut-out that may be pasted onto the shelf."""

    image_b64: str = Field(..., description=_B64_HELP)
    mask_b64: str = Field(
        ...,
        description=(
            "Base64 8-bit mask aligned pixel-for-pixel with the patch image. "
            "Non-zero samples are the product's pixels."
        ),
    )
    candidates: list[tuple[int, int]] = Field(
        ...,
        min_length=1,
        max_length=MAX_CANDIDATES,
        description=(
            "Top-left (x, y) pixel positions for this patch in canvas "
            "coordinates, at most five. A patch may hang over any canvas edge."
        ),
    )

    @field_validator("candidates")
    @classmethod
    def _unique_candidates(cls, value: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if len(set(value)) != len(value):
            raise ValueError("candidate coordinates must be unique per patch")
        return value


class ComposeRequest(BaseModel):
    """Shelf base image plus up to four product patches, in stacking order."""

    base: BaseImage = Field(...)
    patches: list[Patch] = Field(
        ...,
        min_length=1,
        max_length=MAX_PATCHES,
        description=(
            "Patches in back-to-front stacking order: index 0 is the bottom-most "
            "layer and the last patch is drawn last, covering the others."
        ),
    )


class InstanceResult(BaseModel):
    """One pasted instance in the composed image."""

    patch_index: int = Field(..., ge=0, description="Index into the request's patches.")
    position: tuple[int, int] = Field(
        ..., description="Chosen top-left (x, y) of the patch in canvas coordinates."
    )
    visible_pixels: int = Field(
        ..., ge=0, description="Unoccluded, in-canvas pixels of this instance."
    )
    original_mask_pixels: int = Field(
        ..., ge=1, description="Area of the patch mask as submitted."
    )
    visible_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="visible_pixels / original_mask_pixels, always at least 0.5.",
    )
    visible_mask_b64: str = Field(
        ...,
        description=(
            "Base64 canvas-sized 8-bit PNG: 255 where this instance is visible in "
            "the final composite, 0 elsewhere. Disjoint across instances and "
            "suitable as a training target."
        ),
    )


class BaseInstanceSummary(BaseModel):
    """How much of a pre-annotated base instance remains visible."""

    base_index: int = Field(..., ge=0, description="Index into base.instance_masks.")
    visible_pixels: int = Field(..., ge=0)
    original_mask_pixels: int = Field(..., ge=1)
    visible_mask_b64: str = Field(
        ...,
        description=(
            "Base64 canvas-sized 8-bit PNG: 255 where this base instance stays "
            "visible after pasting, 0 elsewhere. Disjoint from the patch masks."
        ),
    )


class ComposeResponse(BaseModel):
    status: Literal["ok"] = "ok"
    composed_image_b64: str = Field(..., description="Base64 PNG of the composite.")
    instances: list[InstanceResult] = Field(
        ...,
        description=(
            "One entry per patch, in stacking order. Visible masks are pairwise "
            "disjoint, so they can be used directly as instance-segmentation targets."
        ),
    )
    base_instances: list[BaseInstanceSummary] = Field(
        ...,
        description="Visibility of the pre-annotated base instances, for auditing.",
    )
    overlapped_base_pixels: int = Field(
        ...,
        ge=0,
        description="Annotated base pixels covered by at least one pasted instance.",
    )
    evaluated_combinations: int = Field(
        ...,
        ge=0,
        description="Candidate combinations explored before choosing this placement.",
    )


class InfeasibleResponse(BaseModel):
    status: Literal["infeasible"] = "infeasible"
    reason: str = Field(..., description="Why no arrangement satisfies the constraints.")


class ErrorResponse(BaseModel):
    error: str
    detail: object


# --------------------------------------------------------------------------- #
# Training-label generation (POST /v1/training-labels)
# --------------------------------------------------------------------------- #


class TrainingLabelRequest(BaseModel):
    """Up to twelve same-sized, non-empty foreground masks plus the grid spec."""

    masks_b64: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_LABEL_MASKS,
        description=(
            "Base64 8-bit mask PNGs (the same non-zero-equals-foreground format "
            "as /v1/compose masks). All masks share one pixel grid; each side is "
            f"at most {512} pixels and every mask contains at least one pixel."
        ),
    )
    stride: int = Field(
        ...,
        description=(
            "Feature-grid cell side in pixels. The grid starts at the top-left "
            "corner; an edge strip shorter than one stride still gets a full "
            "cell whose center is the full-cell center."
        ),
    )
    offset_bound: float = Field(
        ...,
        ge=LABEL_OFFSET_BOUND_MIN,
        le=LABEL_OFFSET_BOUND_MAX,
        description=(
            "Shared per-axis offset limit measured in strides. A cell is eligible "
            f"for a product only when |offset_x| and |offset_y| are both within "
            f"[{LABEL_OFFSET_BOUND_MIN}, {LABEL_OFFSET_BOUND_MAX}]."
        ),
    )

    @field_validator("stride")
    @classmethod
    def _stride_allowed(cls, value: int) -> int:
        if value not in LABEL_STRIDES:
            raise ValueError(f"stride must be one of {list(LABEL_STRIDES)}")
        return value


class LabelAssignment(BaseModel):
    """One product's assigned cell and its signed offset in strides."""

    mask_index: int = Field(..., ge=0, description="Index into the request's masks_b64.")
    cell: tuple[int, int] = Field(
        ...,
        description=(
            "Zero-based grid coordinates [gx, gy] (column, row) of the assigned "
            "cell; its pixel-space center is ((gx + 0.5) * stride, "
            "(gy + 0.5) * stride), including for partial edge cells."
        ),
    )
    offset: tuple[float, float] = Field(
        ...,
        description=(
            "Signed offset [offset_x, offset_y] in strides, from the cell center "
            "to the mean foreground-pixel center of this mask."
        ),
    )


class TrainingLabelResponse(BaseModel):
    status: Literal["ok"] = "ok"
    stride: int = Field(..., description="Cell side in pixels, echoed from the request.")
    grid_size: tuple[int, int] = Field(
        ...,
        description="Grid dimensions [cells_x, cells_y], partial edge cells counted.",
    )
    assignments: list[LabelAssignment] = Field(
        ...,
        description=(
            "One entry per mask, in request order. Cells are pairwise distinct; "
            "the assignment minimises the sum of squared x/y offsets over all "
            "products."
        ),
    )


class LabelConflictResponse(BaseModel):
    status: Literal["conflict"] = "conflict"
    conflict_mask_indices: list[int] = Field(
        ...,
        description=(
            "Indices of a Hall-witness subset of products that cannot all be "
            "placed simultaneously."
        ),
    )
    available_cell_count: int = Field(
        ...,
        ge=0,
        description=(
            "Number of distinct cells the witness products can choose between; "
            "strictly smaller than the number of witness products."
        ),
    )
