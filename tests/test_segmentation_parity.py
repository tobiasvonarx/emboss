"""Opt-in bitwise parity against a read-only snapshot of the original implementation.

EMBOSS_REFERENCE_ROOT=/path/to/reference EMBOSS_RUN_MODEL_PARITY=1 pytest -q \
    tests/test_segmentation_parity.py

No original repository imports, data loaders, or training dependencies are required.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import types
from pathlib import Path

import numpy as np
import pytest


def _reference_root():
    configured = os.environ.get("EMBOSS_REFERENCE_ROOT")
    if not configured:
        pytest.skip(
            "Set EMBOSS_REFERENCE_ROOT to the isolated original source snapshot."
        )
    return Path(configured)


def _reference_module(path, package, replacements=()):
    source = path.read_text()
    for old, new in replacements:
        source = source.replace(old, new)
    namespace = {
        "__name__": "isolated_reference",
        "__package__": package,
        "__file__": str(path),
    }
    # Register for dataclass introspection; exec never creates source-tree bytecode.
    import sys

    module = types.ModuleType("isolated_reference")
    module.__dict__.update(namespace)
    sys.modules[module.__name__] = module
    exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102 - explicit isolated reference execution
    return module


def test_original_semantic_projection_is_bitwise_identical():
    import torch

    from emboss.segmentation.config import ModelConfig
    from emboss.segmentation.net import HfMask2FormerSegmentation

    reference = _reference_module(
        _reference_root() / "src/roof_superstructures/model/net.py",
        "emboss.segmentation",
    )
    config = ModelConfig()
    proxy = types.SimpleNamespace(config=config, query_classes=6)
    generator = torch.Generator().manual_seed(44)
    classes = torch.randn((2, 11, 7), generator=generator)
    masks = torch.randn((2, 11, 13, 17), generator=generator)
    expected = reference.HfMask2FormerSegmentation._semantic_logits(
        proxy, classes, masks, (29, 31)
    )
    actual = HfMask2FormerSegmentation._semantic_logits(proxy, classes, masks, (29, 31))
    assert torch.equal(expected, actual)


def test_full_original_checkpoint_predictions_are_bitwise_identical(tmp_path):
    if os.environ.get("EMBOSS_RUN_MODEL_PARITY") != "1":
        pytest.skip(
            "Set EMBOSS_RUN_MODEL_PARITY=1 for the real checkpoint parity test."
        )
    import torch

    from emboss import roof_superstructures as current
    from emboss.segmentation.checkpoint import MODEL_SHA256, default_checkpoint_path
    from emboss.segmentation.config import ModelConfig
    from emboss.segmentation.net import build_model

    root = _reference_root()
    reference_net = _reference_module(
        root / "src/roof_superstructures/model/net.py", "emboss.segmentation"
    )
    # Extract only the original literal constant assignments, leaving all data-loading
    # imports unexecuted. Keep their dtype, reshape and arithmetic exactly intact.
    loader_tree = ast.parse(
        (root / "src/roof_superstructures/dataset/loader.py").read_text()
    )
    constants = [
        node
        for node in loader_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id in {"RID2_IMAGENET_MEAN", "RID2_IMAGENET_STD"}
            for target in node.targets
        )
    ]
    values = {"torch": torch}
    exec(  # noqa: S102 - original literal constants only
        compile(
            ast.Module(body=constants, type_ignores=[]), "reference_constants", "exec"
        ),
        values,
    )
    current_normalization = __import__(
        "emboss.segmentation.normalization", fromlist=["RGB_MEAN"]
    )
    assert torch.equal(values["RID2_IMAGENET_MEAN"], current_normalization.RGB_MEAN)
    assert torch.equal(values["RID2_IMAGENET_STD"], current_normalization.RGB_STD)
    reference_adapter = _reference_module(
        root / "src/emboss/roof_superstructures.py",
        "emboss",
        (
            (
                "from src.roof_superstructures.dataset.schema",
                "from emboss.segmentation.schema",
            ),
            (
                "from src.roof_superstructures.dataset.loader import CanonicalRoofDataset",
                "# Original constants supplied in isolated module globals.",
            ),
        ),
    )
    reference_adapter.CanonicalRoofDataset = types.SimpleNamespace(
        RGB_MEAN=values["RID2_IMAGENET_MEAN"], RGB_STD=values["RID2_IMAGENET_STD"]
    )
    path = default_checkpoint_path()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = ModelConfig(**current._model_config_kwargs_from_checkpoint(checkpoint))
    device = torch.device(
        os.environ.get(
            "EMBOSS_PARITY_DEVICE", "cuda:1" if torch.cuda.device_count() > 1 else "cpu"
        )
    )
    torch.set_num_threads(4)
    images = [np.random.default_rng(44).integers(0, 256, (512, 512, 3), dtype=np.uint8)]
    image_paths = list(
        dict.fromkeys(
            filter(
                None,
                [
                    os.environ.get("EMBOSS_PARITY_RGB", ""),
                    *os.environ.get("EMBOSS_PARITY_RGBS", "").split(os.pathsep),
                ],
            )
        )
    )
    for image_path in image_paths:
        from PIL import Image

        images.append(np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8))
    results = []
    logits = []
    inputs = []
    for implementation in ("reference", "standalone"):
        if implementation == "reference":
            model = reference_net.build_model(config)
            client = reference_adapter.RoofSuperstructureMaskClient(
                reference_adapter.RoofSuperstructureOptions(checkpoint_path=path)
            )
        else:
            model = build_model(config)
            client = current.RoofSuperstructureMaskClient(
                current.RoofSuperstructureOptions(checkpoint_path=path)
            )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device).eval()
        client._model, client._checkpoint, client._device = model, checkpoint, device
        captured_logits, captured_inputs = [], []

        def capture(
            _module,
            arguments,
            output,
            captured_inputs=captured_inputs,
            captured_logits=captured_logits,
        ):
            captured_inputs.append(arguments[0].detach().cpu())
            captured_logits.append(output["superstructure_class"].detach().cpu())

        hook = model.register_forward_hook(capture)
        prediction = tuple(
            client.predict_hard_segmentation(
                image, vector_roof_mask=np.indices(image.shape[:2])[0] > 5
            )
            for image in images
        )
        hook.remove()
        results.append(prediction)
        logits.append(captured_logits)
        inputs.append(captured_inputs)
        del client, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for expected, actual in zip(inputs[0], inputs[1], strict=True):
        assert torch.equal(expected, actual)
    for expected, actual in zip(logits[0], logits[1], strict=True):
        assert torch.equal(expected, actual), (
            f"maximum logit error: {(expected - actual).abs().max()}"
        )
    for expected, actual in zip(results[0], results[1], strict=True):
        for field in (
            "class_ids",
            "foreground",
            "hard_mask",
            "foreground_class_map",
            "hard_class_map",
        ):
            np.testing.assert_array_equal(
                getattr(expected, field), getattr(actual, field)
            )
    cache_replays = []
    for image_path, image, prediction in zip(
        image_paths, images[1:], results[1][1:], strict=True
    ):
        fixture = Path(image_path).parent
        manifest_path = fixture / "fixture.json"
        metadata_path = fixture / "emboss/image_superstructures_class_map.json"
        if not manifest_path.exists() or not metadata_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        metadata = json.loads(metadata_path.read_text())
        replay = {"building_fid": manifest["building_fid"]}
        cache_replays.append(replay)
        if not metadata["checkpoint_path"].endswith("main_no_boundary/seed_44/best.pt"):
            replay["status"] = "different historical checkpoint; cache replay excluded"
            continue
        import tifffile
        from building_data.geometry import load_vector_house
        from building_data.raster import rasterize_polygon

        house = load_vector_house(fixture / "surfaces.gpkg", manifest["building_fid"])
        correction = json.loads((fixture / "orthophoto_correction.json").read_text())
        x0, y0, x1, y1 = correction["extent_lv95"]
        roof_mask = rasterize_polygon(
            house.roof_envelope,
            extent=(x0, x1, y0, y1),
            width=image.shape[1],
            height=image.shape[0],
        )
        assert (
            current._segmentation_input_sha256(image, roof_mask)
            == metadata["input_sha256"]
        )
        replay_client = current.RoofSuperstructureMaskClient(
            current.RoofSuperstructureOptions(checkpoint_path=path)
        )
        clipped = replay_client._finalize_segmentation(
            prediction.class_ids,
            prediction.foreground,
            dict(prediction.diagnostics),
            vector_roof_mask=roof_mask,
        ).foreground_class_map
        cached = tifffile.imread(fixture / "emboss/image_superstructures_class_map.tif")
        np.testing.assert_array_equal(clipped, cached)
        replay.update(
            {
                "status": "bitwise equal",
                "input_sha256": metadata["input_sha256"],
                "changed_pixels": 0,
                "class_pixel_counts": {
                    str(int(value)): int(count)
                    for value, count in zip(
                        *np.unique(clipped, return_counts=True), strict=True
                    )
                },
            }
        )
    report = {
        "checkpoint_sha256": MODEL_SHA256,
        "torch": torch.__version__,
        "device": str(device),
        "strict_state_tensors": len(checkpoint["model_state_dict"]),
        "model_input_size": checkpoint["training_config"]["model_input_size"],
        "images": [
            {
                "shape": list(image.shape),
                "rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                "class_ids_sha256": hashlib.sha256(
                    result.class_ids.tobytes()
                ).hexdigest(),
                "class_pixel_counts": result.diagnostics["class_pixel_counts"],
            }
            for image, result in zip(images, results[1], strict=True)
        ],
        "normalized_resized_inputs_bitwise_equal": True,
        "dense_logits_bitwise_equal": True,
        "rid_classes_and_all_masks_bitwise_equal": True,
        "historical_cache_replays": cache_replays,
    }
    report_path = Path(
        os.environ.get("EMBOSS_PARITY_REPORT", str(tmp_path / "parity.json"))
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n")
