"""Inter-block dynamic compression package (detector × mixer axes)."""

from conformer_rnnt.compression.core import DynamicCompression
from conformer_rnnt.compression.natural import NaturalPeakCompression
from conformer_rnnt.compression.detectors import DETECTOR_REGISTRY, Detector, build_detector
from conformer_rnnt.compression.mixers import MIXER_REGISTRY, build_mixer

__all__ = [
    "DynamicCompression",
    "NaturalPeakCompression",
    "Detector",
    "DETECTOR_REGISTRY",
    "build_detector",
    "MIXER_REGISTRY",
    "build_mixer",
]
