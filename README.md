# ComfyUI Adaptive Noise Scale

A ComfyUI custom node that wraps any sampler with two-pass **Adaptive Noise Scale** calibration, ported from [sd-webui-adaptive-noise-scale](https://github.com/nawka12/sd-webui-adaptive-noise-scale).

## How it works

During a short calibration pass, the node measures how quickly the denoised output changes relative to the sigma schedule (the "excess" ratio). After collecting enough samples in the texture phase (σ between 0.5 and 5.0), it computes a correction factor and **restarts sampling from scratch** with the corrected noise scale applied from step 0.

```
excess  = (‖d_i − d_{i−1}‖ / ‖d_{i−1} − d_{i−2}‖) / (σ_i / σ_{i−1})

correction = clamp(1 / median_excess^power, floor, ceiling)
```

Two-pass execution:
- **Pass 1 (calibration):** runs the sampler at scale 1.0 until the warmup window completes, then aborts.
- **Pass 2 (production):** restarts from the original latent with the computed correction applied to all intra-step noise injections.

## Usage

Connect the node between `KSamplerSelect` and `KSampler (Advanced)`:

```
[KSamplerSelect] → [Adaptive Noise Scale Sampler] → [KSampler (Advanced)]
```

The node is found under `sampling/custom_sampling/samplers`.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| Warmup steps | 5 | Texture-phase steps to collect before computing correction |
| Correction power | 0.5 | Exponent: 0.5 = gentle (√), 1.0 = linear, >1 = aggressive |
| Dampen floor | 0.80 | Minimum correction multiplier |
| Boost ceiling | 1.15 | Maximum correction multiplier |
| Phase-binned correction | On | Separate correction factors for structural / texture / cleanup phases |

## Sampler compatibility

**Non-SDE samplers** (Euler a, DPM2 a, DPM++ 2S a, DPM fast, etc.) — fully supported. Noise is intercepted via `default_noise_sampler`.

**SDE samplers** (DPM++ SDE, DPM++ 2M SDE, DPM++ 3M SDE) — fully supported. A scaled `BrownianTreeNoiseSampler` is injected for pass 2.

**Deterministic samplers** (DPM++ 2M, LMS) — the calibration pass completes without finding a texture phase; the node prints a warning and returns the result unchanged.

## Installation

### Via ComfyUI Manager

Search for **ComfyUI-Adaptive-Noise-Scale** in the Manager.

### Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/nawka12/comfyui-adaptive-noise-scale
```

Restart ComfyUI.

## Credits

Algorithm ported from [sd-webui-adaptive-noise-scale](https://github.com/nawka12/sd-webui-adaptive-noise-scale). Node structure based on [ComfyUI-Adept-Sampler](https://github.com/nawka12/ComfyUI-Adept-Sampler).

## License

MIT
