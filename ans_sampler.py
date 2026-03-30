"""
Adaptive Noise Scale sampler wrapper for ComfyUI.

Ports the two-pass ANS calibration algorithm from:
  /home/kayfa/sd-webui-adaptive-noise-scale/scripts/adaptive_noise_scale.py

Pattern reference:
  /home/kayfa/adept-comfy/ComfyUI-Adept-Sampler/
"""
import torch

_SDE_SAMPLER_NAMES = frozenset({
    "sample_dpmpp_sde",
    "sample_dpmpp_sde_gpu",
    "sample_dpmpp_2m_sde",
    "sample_dpmpp_2m_sde_gpu",
    "sample_dpmpp_3m_sde",
    "sample_dpmpp_3m_sde_gpu",
})


class _CalibrationAbort(Exception):
    """Raised inside the calibration callback to abort pass 1 cleanly."""


def _compute_correction(excess_samples, power, floor_val, ceiling_val):
    """
    correction = clamp(1 / median_excess^power, floor, ceiling)
    Returns (correction, median_excess).
    """
    if not excess_samples:
        return 1.0, 0.0
    sorted_s = sorted(excess_samples)
    median = sorted_s[len(sorted_s) // 2]
    correction = 1.0 / (median ** power) if median > 0 else 1.0
    return max(floor_val, min(ceiling_val, correction)), median


def _compute_binned_corrections(bins, global_corr, power, floor_val, ceiling_val,
                                 min_samples=3):
    """Per-phase corrections; bins with < min_samples fall back to global."""
    result = {}
    for name, samples in bins.items():
        if len(samples) >= min_samples:
            c, _ = _compute_correction(samples, power, floor_val, ceiling_val)
            result[name] = c
        else:
            result[name] = global_corr
    return result


def _get_phase_correction(sigma_val, bin_corrections, global_correction):
    """Return the correction for the phase that sigma_val falls in."""
    if bin_corrections is None:
        return global_correction
    if sigma_val > 5.0:
        return bin_corrections.get("structural", global_correction)
    elif sigma_val > 0.5:
        return bin_corrections.get("texture", global_correction)
    else:
        return bin_corrections.get("cleanup", global_correction)


class ANSWrappedSampler:
    """
    Wraps any ComfyUI SAMPLER to apply two-pass Adaptive Noise Scale calibration.

    Pass 1 (calibration): runs the inner sampler with a callback that collects
    excess-ratio samples in the texture phase (0.5 < sigma < 5.0) and raises
    _CalibrationAbort once `warmup` samples are collected.

    Pass 2 (production): restarts from the same noise + latent_image inputs
    (inner sampler reconstructs identical x_initial) with intra-step noise
    scaled by the computed correction factor.
    """

    def __init__(self, inner_sampler, warmup, power, floor_val, ceiling_val,
                 use_binned):
        self.inner = inner_sampler
        self.warmup = warmup
        self.power = power
        self.floor_val = floor_val
        self.ceiling_val = ceiling_val
        self.use_binned = use_binned

    def _is_sde(self):
        func = getattr(self.inner, "sampler_function", None)
        return getattr(func, "__name__", "") in _SDE_SAMPLER_NAMES

    def sample(self, model_wrap, sigmas, extra_args, callback, noise,
               latent_image=None, denoise_mask=None, disable_pbar=False):
        state = {
            "prev_denoised": None,
            "prev_change_norm": None,
            "prev_sigma_val": None,
            "excess_samples": [],
            "excess_bins": {"structural": [], "texture": [], "cleanup": []},
        }
        warmup = self.warmup
        use_binned = self.use_binned

        def cal_callback(step, x0, x, total_steps):
            sigma_val = sigmas[step].item()
            prev_denoised = state["prev_denoised"]
            prev_sigma_val = state["prev_sigma_val"]
            prev_change_norm = state["prev_change_norm"]

            if prev_denoised is not None:
                change_norm = torch.norm(
                    (x0 - prev_denoised).flatten(1), dim=1
                ).mean().item()

                if (prev_change_norm is not None
                        and sigma_val > 0
                        and prev_sigma_val is not None
                        and prev_sigma_val > 0):
                    change_ratio = change_norm / (prev_change_norm + 1e-8)
                    sigma_ratio = sigma_val / (prev_sigma_val + 1e-8)
                    excess = change_ratio / (sigma_ratio + 1e-8)

                    if use_binned:
                        if prev_sigma_val > 5.0:
                            state["excess_bins"]["structural"].append(excess)
                        elif prev_sigma_val > 0.5:
                            state["excess_bins"]["texture"].append(excess)
                        else:
                            state["excess_bins"]["cleanup"].append(excess)

                    if 0.5 < prev_sigma_val < 5.0:
                        state["excess_samples"].append(excess)
                        if len(state["excess_samples"]) >= warmup:
                            raise _CalibrationAbort()

                state["prev_change_norm"] = change_norm
            else:
                state["prev_change_norm"] = None

            state["prev_denoised"] = x0
            state["prev_sigma_val"] = sigma_val

        # ── Pass 1: calibration ────────────────────────────────────────────
        calibrated = False
        try:
            self.inner.sample(
                model_wrap, sigmas, extra_args, cal_callback,
                noise, latent_image, denoise_mask, disable_pbar,
            )
        except _CalibrationAbort:
            calibrated = True

        if not calibrated:
            print("[ANS] Warning: texture phase not reached — no correction applied.")
            return self.inner.sample(
                model_wrap, sigmas, extra_args, callback,
                noise, latent_image, denoise_mask, disable_pbar,
            )

        # ── Compute correction ─────────────────────────────────────────────
        correction, median = _compute_correction(
            state["excess_samples"], self.power, self.floor_val, self.ceiling_val,
        )
        bin_corrections = None
        if use_binned:
            bin_corrections = _compute_binned_corrections(
                state["excess_bins"], correction,
                self.power, self.floor_val, self.ceiling_val,
            )
            bin_info = {k: f"{v:.3f}" for k, v in bin_corrections.items()}
            print(
                f"[ANS] Calibrated — global={correction:.3f} "
                f"(median_excess={median:.3f}), phases={bin_info}"
            )
        else:
            print(
                f"[ANS] Calibrated — correction={correction:.3f} "
                f"(median_excess={median:.3f})"
            )

        # ── Pass 2: production with noise interception ─────────────────────
        if self._is_sde():
            return self._run_pass2_sde(
                model_wrap, sigmas, extra_args, callback,
                noise, latent_image, denoise_mask, disable_pbar,
                correction, bin_corrections,
            )
        return self._run_pass2_non_sde(
            model_wrap, sigmas, extra_args, callback,
            noise, latent_image, denoise_mask, disable_pbar,
            correction, bin_corrections,
        )

    def _run_pass2_non_sde(self, model_wrap, sigmas, extra_args, callback,
                            noise, latent_image, denoise_mask, disable_pbar,
                            correction, bin_corrections):
        import comfy.k_diffusion.sampling as k_sampling
        orig_dns = k_sampling.default_noise_sampler

        def scaled_dns(x, **kwargs):
            base_ns = orig_dns(x, **kwargs)
            def ns(sigma, sigma_next):
                sigma_val = sigma.item() if torch.is_tensor(sigma) else float(sigma)
                scale = _get_phase_correction(sigma_val, bin_corrections, correction)
                return base_ns(sigma, sigma_next) * scale
            return ns

        k_sampling.default_noise_sampler = scaled_dns
        try:
            return self.inner.sample(
                model_wrap, sigmas, extra_args, callback,
                noise, latent_image, denoise_mask, disable_pbar,
            )
        finally:
            k_sampling.default_noise_sampler = orig_dns

    def _run_pass2_sde(self, model_wrap, sigmas, extra_args, callback,
                       noise, latent_image, denoise_mask, disable_pbar,
                       correction, bin_corrections):
        if latent_image is not None:
            x_approx = latent_image + noise * sigmas[0]
        else:
            x_approx = noise * sigmas[0]

        sigma_min = sigmas[sigmas > 0].min().item()
        sigma_max = sigmas[0].item()

        try:
            from comfy.k_diffusion.sampling import BrownianTreeNoiseSampler
            raw_ns = BrownianTreeNoiseSampler(x_approx, sigma_min, sigma_max)

            def scaled_ns(sigma, sigma_next):
                sigma_val = sigma.item() if torch.is_tensor(sigma) else float(sigma)
                scale = _get_phase_correction(sigma_val, bin_corrections, correction)
                return raw_ns(sigma, sigma_next) * scale
        except Exception:
            def scaled_ns(sigma, sigma_next):
                sigma_val = sigma.item() if torch.is_tensor(sigma) else float(sigma)
                scale = _get_phase_correction(sigma_val, bin_corrections, correction)
                return torch.randn_like(x_approx) * scale

        try:
            orig_extra = dict(self.inner.extra_options)
        except AttributeError:
            return self._run_pass2_non_sde(
                model_wrap, sigmas, extra_args, callback,
                noise, latent_image, denoise_mask, disable_pbar,
                correction, bin_corrections,
            )

        self.inner.extra_options = {**orig_extra, "noise_sampler": scaled_ns}
        try:
            return self.inner.sample(
                model_wrap, sigmas, extra_args, callback,
                noise, latent_image, denoise_mask, disable_pbar,
            )
        finally:
            self.inner.extra_options = orig_extra
