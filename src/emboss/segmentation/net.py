"""Mask2Former model wrapper for RID2 roof-superstructure segmentation."""

from __future__ import annotations

from importlib.resources import files

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class HfMask2FormerSegmentation(nn.Module):
    """Hugging Face Mask2Former adapted to dense RID2 semantic logits."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        if config.input_channels != 3:
            raise ValueError("Mask2Former checkpoints expect normalized RGB input.")
        query_classes = (
            config.mask2former_query_classes or config.superstructure_classes
        )
        valid_query_classes = {
            config.superstructure_classes - 1,
            config.superstructure_classes,
        }
        if query_classes not in valid_query_classes:
            raise ValueError(
                "Mask2Former query classes must contain every foreground class and may "
                "optionally contain semantic background."
            )
        self.query_classes = query_classes

        try:
            from transformers import (
                Mask2FormerConfig,
                Mask2FormerForUniversalSegmentation,
            )
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "transformers and scipy are required for HF Mask2Former. "
                "Install project dependencies before inference."
            ) from exc

        id2label = {index: f"class_{index}" for index in range(self.query_classes)}
        label2id = {label: index for index, label in id2label.items()}
        if (
            config.mask2former_model_name
            != "facebook/mask2former-swin-large-coco-panoptic"
        ):
            raise ValueError(
                "Only the pinned Swin-L COCO Mask2Former architecture is supported."
            )
        # Architecture from upstream commit 85b535928a783691eaf27467a573b26d543336ea.
        # The complete fine-tuned state is loaded strictly by the inference client.
        architecture = Mask2FormerConfig.from_json_file(
            str(files(__package__).joinpath("mask2former-config.json"))
        )
        architecture.update(
            {
                "num_labels": self.query_classes,
                "id2label": id2label,
                "label2id": label2id,
                "class_weight": config.mask2former_class_weight,
                "mask_weight": config.mask2former_mask_weight,
                "dice_weight": config.mask2former_dice_weight,
                "no_object_weight": config.mask2former_no_object_weight,
            }
        )
        self.model = Mask2FormerForUniversalSegmentation(architecture)
        for parameter in self.model.parameters():
            parameter.requires_grad_(True)

    def _mask_labels(
        self,
        targets: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        mask_labels: list[torch.Tensor] = []
        class_labels: list[torch.Tensor] = []
        target_classes = self.query_classes
        if not self.config.match_background_targets:
            target_classes = min(target_classes, self.config.superstructure_classes - 1)
        for target in targets:
            present_classes = torch.unique(target)
            present_classes = present_classes[
                (present_classes >= 0) & (present_classes < target_classes)
            ]
            if present_classes.numel() == 0:
                masks = target.new_zeros((0, *target.shape), dtype=torch.float32)
                class_labels.append(target.new_empty((0,), dtype=torch.long))
                mask_labels.append(masks)
                continue
            masks = torch.stack(
                [(target == class_id).float() for class_id in present_classes], dim=0
            )
            mask_labels.append(masks)
            class_labels.append(present_classes.long())
        return mask_labels, class_labels

    def _semantic_logits(
        self,
        class_logits: torch.Tensor,
        mask_logits: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        if mask_logits.shape[-2:] != output_size:
            mask_logits = F.interpolate(
                mask_logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

        class_prob = class_logits.softmax(dim=-1)[..., : self.query_classes]
        mask_prob = mask_logits.sigmoid()
        background_index = self.config.superstructure_classes - 1
        foreground_prob = torch.einsum(
            "bqc,bqhw->bchw",
            class_prob[..., :background_index],
            mask_prob,
        )
        background_prob = (1.0 - foreground_prob.amax(dim=1, keepdim=True)).clamp(
            1e-6, 1.0
        )
        semantic_prob = torch.cat(
            (foreground_prob.clamp_min(1e-6), background_prob), dim=1
        )
        return semantic_prob.clamp_min(1e-6).log()

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor | None = None,
        output_size: tuple[int, int] | torch.Size | None = None,
    ) -> dict[str, torch.Tensor]:
        kwargs: dict[str, object] = {"pixel_values": x, "return_dict": True}
        if targets is not None:
            mask_labels, class_labels = self._mask_labels(targets)
            kwargs["mask_labels"] = mask_labels
            kwargs["class_labels"] = class_labels
        outputs = self.model(**kwargs)
        semantic_logits = self._semantic_logits(
            outputs.class_queries_logits,
            outputs.masks_queries_logits,
            tuple(output_size or x.shape[-2:]),
        )
        result = {"superstructure_class": semantic_logits}
        if outputs.loss is not None:
            result["loss"] = outputs.loss
        return result


def build_model(config: ModelConfig | None = None) -> HfMask2FormerSegmentation:
    """Build the RID2 Mask2Former segmentation model."""

    return HfMask2FormerSegmentation(config or ModelConfig())


def compile_training_model(model: HfMask2FormerSegmentation) -> None:
    """Compile the tensor-only model core without compiling target matching."""

    model.model.model.compile(
        options={"fallback_random": True},
        fullgraph=False,
        dynamic=False,
    )
