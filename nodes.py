"""
ComfyUI node for Adaptive Noise Scale sampler wrapper.
"""
try:
    from .ans_sampler import ANSWrappedSampler
except ImportError:
    from ans_sampler import ANSWrappedSampler


class AdaptiveNoiseScaleSampler:
    """
    Wraps any SAMPLER with two-pass Adaptive Noise Scale calibration.

    Connect: [KSamplerSelect] → [this node] → [KSampler (Advanced)]

    Pass 1 measures how quickly the denoised output changes relative to the
    sigma schedule, aborts early, and computes a correction factor.
    Pass 2 restarts from the same initial latent with intra-step noise
    scaled by the correction.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": ("SAMPLER",),
                "warmup_steps": (
                    "INT",
                    {
                        "default": 5,
                        "min": 1,
                        "max": 20,
                        "tooltip": "Texture-phase steps (0.5 < sigma < 5.0) to collect before calibrating",
                    },
                ),
                "correction_power": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.1,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Exponent: 0.5=gentle (sqrt), 1.0=linear, >1=aggressive",
                    },
                ),
                "dampen_floor": (
                    "FLOAT",
                    {
                        "default": 0.80,
                        "min": 0.5,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Minimum correction multiplier",
                    },
                ),
                "boost_ceiling": (
                    "FLOAT",
                    {
                        "default": 1.15,
                        "min": 1.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Maximum correction multiplier",
                    },
                ),
                "phase_binned": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Separate correction factors for structural / texture / cleanup phases",
                    },
                ),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "get_sampler"
    CATEGORY = "sampling/custom_sampling/samplers"

    def get_sampler(self, sampler, warmup_steps, correction_power,
                    dampen_floor, boost_ceiling, phase_binned):
        return (ANSWrappedSampler(
            inner_sampler=sampler,
            warmup=warmup_steps,
            power=correction_power,
            floor_val=dampen_floor,
            ceiling_val=boost_ceiling,
            use_binned=phase_binned,
        ),)


NODE_CLASS_MAPPINGS = {
    "AdaptiveNoiseScaleSampler": AdaptiveNoiseScaleSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdaptiveNoiseScaleSampler": "Adaptive Noise Scale Sampler",
}
