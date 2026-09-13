"""Model configuration for RID2 roof-superstructure segmentation."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import RID2_IMAGE_SIZE_PX, RID2_SUPERSTRUCTURE_CLASS_IDS


@dataclass(frozen=True)
class ModelConfig:
    """Mask2Former configuration for RID2 roof-superstructure segmentation."""

    input_channels: int = 3
    input_size_px: int = RID2_IMAGE_SIZE_PX
    superstructure_classes: int = len(RID2_SUPERSTRUCTURE_CLASS_IDS)
    mask2former_query_classes: int | None = None
    # Match semantic background during training; dense background remains a residual.
    match_background_targets: bool = True
    mask2former_model_name: str = "facebook/mask2former-swin-large-coco-panoptic"
    mask2former_class_weight: float = 1.8425044558804362
    mask2former_mask_weight: float = 6.5842353139496925
    mask2former_dice_weight: float = 4.693810851459411
    mask2former_no_object_weight: float = 0.12
