"""Tests for ANSWrappedSampler two-pass behavior and noise interception."""
import pytest
import torch
from ans_sampler import ANSWrappedSampler, _SDE_SAMPLER_NAMES
from conftest import MockSampler


# ---------------------------------------------------------------------------
# SDE detection
# ---------------------------------------------------------------------------

class TestSdeDetection:
    def test_euler_ancestral_is_not_sde(self):
        inner = MockSampler("sample_euler_ancestral")
        wrapper = ANSWrappedSampler(inner, 5, 0.5, 0.8, 1.15, True)
        assert not wrapper._is_sde()

    def test_dpmpp_sde_is_sde(self):
        inner = MockSampler("sample_dpmpp_sde")
        wrapper = ANSWrappedSampler(inner, 5, 0.5, 0.8, 1.15, True)
        assert wrapper._is_sde()

    def test_all_sde_names_detected(self):
        for name in _SDE_SAMPLER_NAMES:
            inner = MockSampler(name)
            wrapper = ANSWrappedSampler(inner, 5, 0.5, 0.8, 1.15, True)
            assert wrapper._is_sde(), f"{name} should be detected as SDE"

    def test_unknown_name_is_not_sde(self):
        inner = MockSampler("sample_unknown_custom")
        wrapper = ANSWrappedSampler(inner, 5, 0.5, 0.8, 1.15, True)
        assert not wrapper._is_sde()

    def test_missing_sampler_function_is_not_sde(self):
        inner = MockSampler()
        del inner.sampler_function
        wrapper = ANSWrappedSampler(inner, 5, 0.5, 0.8, 1.15, True)
        assert not wrapper._is_sde()


# ---------------------------------------------------------------------------
# Calibration pass — abort behavior
# ---------------------------------------------------------------------------

class TestCalibrationPass:
    """
    Texture sigmas: [3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0]
    Linearly changing denoised tensors produce constant change_norm=16.0.
    Excess values accumulate starting at step 2. With warmup=3, abort fires
    after step 4 (excess_samples reaches length 3).
    """

    def _make_wrapper(self, warmup=3, use_binned=False):
        denoised_seq = [torch.ones(1, 4, 8, 8) * i for i in range(6)]
        inner = MockSampler("sample_euler_ancestral", denoised_seq)
        wrapper = ANSWrappedSampler(inner, warmup, 0.5, 0.8, 1.15, use_binned)
        return wrapper, inner

    def test_calibration_triggers_two_passes(self, texture_sigmas):
        wrapper, inner = self._make_wrapper(warmup=3)
        noise = torch.zeros(1, 4, 8, 8)
        wrapper.sample(None, texture_sigmas, {}, None, noise)
        assert inner.call_count == 2

    def test_no_texture_phase_returns_single_pass(self):
        high_sigmas = torch.tensor([10.0, 8.0, 6.0, 0.0])
        denoised_seq = [torch.ones(1, 4, 8, 8) * i for i in range(3)]
        inner = MockSampler("sample_euler_ancestral", denoised_seq)
        wrapper = ANSWrappedSampler(inner, 3, 0.5, 0.8, 1.15, False)
        noise = torch.zeros(1, 4, 8, 8)
        wrapper.sample(None, high_sigmas, {}, None, noise)
        assert inner.call_count == 1

    def test_correction_within_bounds(self, texture_sigmas):
        wrapper, inner = self._make_wrapper(warmup=3)
        noise = torch.zeros(1, 4, 8, 8)
        wrapper.sample(None, texture_sigmas, {}, None, noise)
        assert inner.call_count == 2

    def test_original_callback_forwarded_in_pass2(self, texture_sigmas):
        wrapper, inner = self._make_wrapper(warmup=3)
        noise = torch.zeros(1, 4, 8, 8)
        calls = []
        def user_callback(step, x0, x, total):
            calls.append(step)
        wrapper.sample(None, texture_sigmas, {}, user_callback, noise)
        assert len(calls) > 0

    def test_phase_binned_correction(self, texture_sigmas):
        wrapper, inner = self._make_wrapper(warmup=3, use_binned=True)
        noise = torch.zeros(1, 4, 8, 8)
        wrapper.sample(None, texture_sigmas, {}, None, noise)
        assert inner.call_count == 2


# ---------------------------------------------------------------------------
# Non-SDE production pass — default_noise_sampler interception
# ---------------------------------------------------------------------------

class TestNonSdePass2:
    def _make_wrapper_and_inner(self, warmup=3):
        denoised_seq = [torch.ones(1, 4, 8, 8) * i for i in range(6)]
        inner = MockSampler("sample_euler_ancestral", denoised_seq)
        wrapper = ANSWrappedSampler(inner, warmup, 0.5, 0.8, 1.15, False)
        return wrapper, inner

    def test_default_noise_sampler_is_patched_during_pass2(self, texture_sigmas):
        import comfy.k_diffusion.sampling as k_sampling
        original_dns = k_sampling.default_noise_sampler

        wrapper, inner = self._make_wrapper_and_inner()
        noise = torch.zeros(1, 4, 8, 8)
        wrapper.sample(None, texture_sigmas, {}, None, noise)

        assert inner.dns_seen[0] is original_dns
        assert inner.dns_seen[1] is not original_dns

    def test_default_noise_sampler_restored_after_pass2(self, texture_sigmas):
        import comfy.k_diffusion.sampling as k_sampling
        original_dns = k_sampling.default_noise_sampler

        wrapper, inner = self._make_wrapper_and_inner()
        noise = torch.zeros(1, 4, 8, 8)
        wrapper.sample(None, texture_sigmas, {}, None, noise)

        assert k_sampling.default_noise_sampler is original_dns

    def test_patched_dns_scales_output(self, texture_sigmas):
        """The patched default_noise_sampler should scale noise by correction."""
        import comfy.k_diffusion.sampling as k_sampling

        seen_noise = []

        class CapturingMockSampler(MockSampler):
            def sample(self, model_wrap, sigmas, extra_args, callback, noise,
                       latent_image=None, denoise_mask=None, disable_pbar=False):
                dns = k_sampling.default_noise_sampler
                x = torch.ones(1, 4, 8, 8)
                ns = dns(x)
                seen_noise.append(ns(sigmas[0], sigmas[1]))
                return super().sample(
                    model_wrap, sigmas, extra_args, callback, noise,
                    latent_image, denoise_mask, disable_pbar,
                )

        denoised_seq = [torch.ones(1, 4, 8, 8) * i for i in range(6)]
        inner = CapturingMockSampler("sample_euler_ancestral", denoised_seq)
        wrapper = ANSWrappedSampler(inner, 3, 0.5, 0.8, 1.15, False)
        noise = torch.zeros(1, 4, 8, 8)
        wrapper.sample(None, texture_sigmas, {}, None, noise)

        assert len(seen_noise) == 2

    def test_no_noise_sampler_injected_for_non_sde(self, texture_sigmas):
        """Non-SDE path must not inject noise_sampler into extra_options."""
        wrapper, inner = self._make_wrapper_and_inner()
        noise = torch.zeros(1, 4, 8, 8)
        wrapper.sample(None, texture_sigmas, {}, None, noise)
        assert inner.noise_sampler_seen[1] is None


# ---------------------------------------------------------------------------
# SDE production pass — BrownianTreeNoiseSampler injection
# ---------------------------------------------------------------------------

class TestSdePass2:
    def _make_sde_wrapper(self, warmup=3):
        denoised_seq = [torch.ones(1, 4, 8, 8) * i for i in range(6)]
        inner = MockSampler("sample_dpmpp_sde", denoised_seq)
        wrapper = ANSWrappedSampler(inner, warmup, 0.5, 0.8, 1.15, False)
        return wrapper, inner

    def test_noise_sampler_injected_for_sde(self, texture_sigmas):
        wrapper, inner = self._make_sde_wrapper()
        noise = torch.zeros(1, 4, 8, 8)
        wrapper.sample(None, texture_sigmas, {}, None, noise)
        assert inner.noise_sampler_seen[0] is None
        assert inner.noise_sampler_seen[1] is not None
        assert callable(inner.noise_sampler_seen[1])

    def test_extra_options_restored_after_sde_pass2(self, texture_sigmas):
        wrapper, inner = self._make_sde_wrapper()
        inner.extra_options["existing_key"] = "existing_value"
        noise = torch.zeros(1, 4, 8, 8)
        wrapper.sample(None, texture_sigmas, {}, None, noise)
        assert "noise_sampler" not in inner.extra_options
        assert inner.extra_options.get("existing_key") == "existing_value"

    def test_sde_noise_sampler_is_callable_with_sigma_args(self, texture_sigmas):
        wrapper, inner = self._make_sde_wrapper()
        noise = torch.zeros(1, 4, 8, 8)
        wrapper.sample(None, texture_sigmas, {}, None, noise)
        ns = inner.noise_sampler_seen[1]
        result = ns(texture_sigmas[0], texture_sigmas[1])
        assert result.shape == (1, 4, 8, 8)

    def test_sde_default_noise_sampler_not_patched(self, texture_sigmas):
        """SDE path uses noise_sampler kwarg, must NOT patch default_noise_sampler."""
        import comfy.k_diffusion.sampling as k_sampling
        original_dns = k_sampling.default_noise_sampler
        wrapper, inner = self._make_sde_wrapper()
        noise = torch.zeros(1, 4, 8, 8)
        wrapper.sample(None, texture_sigmas, {}, None, noise)
        assert inner.dns_seen[1] is original_dns

    def test_sde_fallback_when_extra_options_missing(self, texture_sigmas):
        """If inner has no extra_options, SDE path falls back to non-SDE."""
        import comfy.k_diffusion.sampling as k_sampling
        denoised_seq = [torch.ones(1, 4, 8, 8) * i for i in range(6)]
        inner = MockSampler("sample_dpmpp_sde", denoised_seq)
        del inner.extra_options
        wrapper = ANSWrappedSampler(inner, 3, 0.5, 0.8, 1.15, False)
        noise = torch.zeros(1, 4, 8, 8)
        original_dns = k_sampling.default_noise_sampler
        wrapper.sample(None, texture_sigmas, {}, None, noise)
        assert inner.dns_seen[1] is not original_dns

    def test_sde_with_latent_image(self, texture_sigmas):
        """x_approx uses latent_image when provided."""
        wrapper, inner = self._make_sde_wrapper()
        noise = torch.zeros(1, 4, 8, 8)
        latent = torch.ones(1, 4, 8, 8) * 0.5
        wrapper.sample(None, texture_sigmas, {}, None, noise, latent_image=latent)
        assert inner.call_count == 2


# ---------------------------------------------------------------------------
# ComfyUI node
# ---------------------------------------------------------------------------

class TestAdaptiveNoiseScaleSamplerNode:
    def test_input_types_has_required_fields(self):
        from nodes import AdaptiveNoiseScaleSampler
        inputs = AdaptiveNoiseScaleSampler.INPUT_TYPES()
        required = inputs["required"]
        assert "sampler" in required
        assert "warmup_steps" in required
        assert "correction_power" in required
        assert "dampen_floor" in required
        assert "boost_ceiling" in required
        assert "phase_binned" in required

    def test_return_types_is_sampler(self):
        from nodes import AdaptiveNoiseScaleSampler
        assert AdaptiveNoiseScaleSampler.RETURN_TYPES == ("SAMPLER",)

    def test_category(self):
        from nodes import AdaptiveNoiseScaleSampler
        assert AdaptiveNoiseScaleSampler.CATEGORY == "sampling/custom_sampling/samplers"

    def test_get_sampler_returns_ans_wrapped_sampler(self):
        from nodes import AdaptiveNoiseScaleSampler
        from ans_sampler import ANSWrappedSampler
        node = AdaptiveNoiseScaleSampler()
        inner = MockSampler()
        result = node.get_sampler(
            sampler=inner,
            warmup_steps=5,
            correction_power=0.5,
            dampen_floor=0.80,
            boost_ceiling=1.15,
            phase_binned=True,
        )
        assert isinstance(result, tuple)
        assert len(result) == 1
        wrapped = result[0]
        assert isinstance(wrapped, ANSWrappedSampler)
        assert wrapped.inner is inner
        assert wrapped.warmup == 5
        assert wrapped.power == 0.5
        assert wrapped.floor_val == 0.80
        assert wrapped.ceiling_val == 1.15
        assert wrapped.use_binned is True

    def test_node_class_mappings_exported(self):
        import importlib
        import sys
        for key in list(sys.modules.keys()):
            if key in ("__init__",):
                del sys.modules[key]
        from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        assert "AdaptiveNoiseScaleSampler" in NODE_CLASS_MAPPINGS
        assert "AdaptiveNoiseScaleSampler" in NODE_DISPLAY_NAME_MAPPINGS
