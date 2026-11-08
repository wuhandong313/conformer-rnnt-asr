"""Explicit offline compression stages, including Mel-input compression.

Layer 0 means the model's preprocessed Mel input, before projection/stacking.
Positive layer numbers refer to the global Conformer layer numbering.
"""

from copy import deepcopy
from typing import Any, Dict, List

import torch

from conformer_rnnt.compression.natural import NaturalPeakCompression


def normalize_compression_stages(stages: List[dict], num_layers: int) -> List[dict]:
    """Reject ambiguous or unsupported stage recipes instead of ignoring them."""
    if type(num_layers) is not int or num_layers < 1:
        raise ValueError("Staged compression requires a positive integer model.num_layers")
    if not isinstance(stages, list) or not stages:
        raise ValueError("compression.stages must be a non-empty list")
    allowed = {
        "after_layer", "detector", "mixer", "window_size",
        "weight_predictor_mode", "neighborhood_radius",
    }
    normalized = []
    previous = -1
    for stage in stages:
        if not isinstance(stage, dict) or set(stage) - allowed:
            raise ValueError("Unknown compression stage options; rate/context controls are unsupported")
        layer = stage.get("after_layer")
        if type(layer) is not int or not 0 <= layer < num_layers:
            raise ValueError("after_layer must be 0 (Mel input) or an interior global layer")
        if layer <= previous:
            raise ValueError("Compression stages must have unique, increasing after_layer values")
        previous = layer
        detector = deepcopy(stage.get("detector", {"type": "cosine_diff"}))
        if (not isinstance(detector, dict)
                or set(detector) - {"type", "preprocess"}
                or detector.get("type") != "cosine_diff"
                or detector.get("preprocess", "none") != "none"):
            raise ValueError("Placement experiments require unsmoothed cosine_diff peaks and valleys")
        mixer = deepcopy(stage.get("mixer", {"type": "learned"}))
        if not isinstance(mixer, dict) or mixer not in ({"type": "learned"}, {"type": "hard"}):
            raise ValueError("Placement stages support learned five_frames or hard frame retention")
        radius = stage.get("neighborhood_radius", 0)
        if type(radius) is not int or radius < 0:
            raise ValueError("neighborhood_radius must be a non-negative integer")
        if radius and mixer["type"] != "hard":
            raise ValueError("Neighborhood retention keeps original frames and requires the hard mixer")
        if mixer["type"] == "hard":
            if "window_size" in stage or "weight_predictor_mode" in stage:
                raise ValueError("Hard retention has no fusion window or weight predictor")
            learned_options = {}
        else:
            if stage.get("window_size", 5) != 5 or stage.get("weight_predictor_mode", "five_frames") != "five_frames":
                raise ValueError("Placement experiments use a five-frame, frame-input weight predictor")
            learned_options = {"window_size": 5, "weight_predictor_mode": "five_frames"}
        normalized.append({
            "after_layer": layer,
            "detector": {"type": "cosine_diff", "preprocess": "none"},
            "mixer": mixer,
            "neighborhood_radius": radius,
            **learned_options,
        })
    return normalized


def placement_options_from_config(model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Share placement parsing between the standalone and Lightning factories."""
    compression = model_config.get("compression", {})
    if "stages" not in compression:
        if "num_layers" in model_config:
            raise ValueError("model.num_layers is only used with explicit compression.stages")
        return {}
    if model_config.get("sequence_mode") != "offline":
        raise ValueError("compression.stages requires sequence_mode: offline")
    if set(compression) != {"stages"}:
        raise ValueError("compression.stages cannot be mixed with single-stage/rate/context options")
    if any(key in model_config for key in (
        "num_layers_block1", "num_layers_block2", "left_context_length", "right_context_length",
    )):
        raise ValueError("Staged recipes use model.num_layers and global positions, without context/block splits")
    num_layers = model_config.get("num_layers")
    return {
        "encoder_num_layers": num_layers,
        "compression_stages": normalize_compression_stages(compression["stages"], num_layers),
    }


class CompressionSchedule(torch.nn.Module):
    """Own trainable modules and expose per-stage observed lengths to the runner."""

    def __init__(self, stages: List[dict], num_layers: int, input_dim: int, hidden_dim: int):
        super().__init__()
        self.definitions = []
        modules = []
        for stage in normalize_compression_stages(stages, num_layers):
            dim = input_dim if stage["after_layer"] == 0 else hidden_dim
            self.definitions.append({
                **stage, "feature_dim": dim,
                "input_domain": "model_mel_input" if stage["after_layer"] == 0 else "encoder",
            })
            modules.append(NaturalPeakCompression(
                detector=stage["detector"], mixer=stage["mixer"], feature_dim=dim,
                window_size=stage.get("window_size", 5),
                weight_predictor_mode=stage.get("weight_predictor_mode", "five_frames"),
                neighborhood_radius=stage["neighborhood_radius"],
            ))
        self.stages = torch.nn.ModuleList(modules)
        self._index = {stage["after_layer"]: i for i, stage in enumerate(self.definitions)}
        self.last_lengths = None
        # Opt-in index traces are observations, never an input to the selector.
        self.record_indices = False
        self.last_indices = None

    def start_batch(self, lengths):
        self.last_lengths = {"input": lengths.tolist(), "stages": []}
        self.last_indices = [] if self.record_indices else None

    def at_layer(self, layer, frames, lengths):
        if layer not in self._index:
            return frames, lengths
        index = self._index[layer]
        module = self.stages[index]
        module.record_indices = self.record_indices
        output, result = module(frames, lengths)
        self.last_lengths["stages"].append({
            "after_layer": layer,
            "input_domain": self.definitions[index]["input_domain"],
            "input": list(module.last_lengths["input"]),
            "candidates": list(module.last_candidate_lengths),
            "output": list(module.last_lengths["output"]),
        })
        if self.record_indices:
            self.last_indices.append({"after_layer": layer, **module.last_indices})
        return output, result
