"""
ComfyUI Adaptive Noise Scale

Wraps any SAMPLER with two-pass ANS calibration.
Connect: [KSamplerSelect] → [Adaptive Noise Scale Sampler] → [KSampler (Advanced)]
"""
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print("✅ ComfyUI-Adaptive-Noise-Scale loaded successfully!")
