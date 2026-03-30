"""
Test fixtures: mock comfy module and MockSampler.

comfy.k_diffusion.sampling is mocked before any module import so that
ans_sampler.py can be imported without a ComfyUI installation.
"""
import sys
import types
import torch
import pytest


def pytest_configure(config):
    """Install comfy mock into sys.modules before any test imports."""
    comfy = types.ModuleType("comfy")
    comfy_kd = types.ModuleType("comfy.k_diffusion")
    comfy_kd_sampling = types.ModuleType("comfy.k_diffusion.sampling")

    def default_noise_sampler(x):
        return lambda sigma, sigma_next: torch.randn_like(x)

    class MockBrownianTreeNoiseSampler:
        def __init__(self, x, sigma_min, sigma_max, seed=None,
                     transform=None, cpu=False):
            self.x = x

        def __call__(self, sigma, sigma_next):
            return torch.randn_like(self.x)

    comfy_kd_sampling.default_noise_sampler = default_noise_sampler
    comfy_kd_sampling.BrownianTreeNoiseSampler = MockBrownianTreeNoiseSampler
    comfy.k_diffusion = comfy_kd
    comfy_kd.sampling = comfy_kd_sampling

    sys.modules.setdefault("comfy", comfy)
    sys.modules.setdefault("comfy.k_diffusion", comfy_kd)
    sys.modules.setdefault("comfy.k_diffusion.sampling", comfy_kd_sampling)


class MockSampler:
    """Minimal sampler mock. Calls callback(i, denoised_seq[i], x, n_steps)
    for each step, then returns a zero tensor."""

    def __init__(self, func_name="sample_euler_ancestral", denoised_sequence=None):
        import types as _types
        self.sampler_function = _types.SimpleNamespace(__name__=func_name)
        self.extra_options = {}
        self.call_count = 0
        self.denoised_sequence = denoised_sequence or []
        self.noise_sampler_seen = []
        self.dns_seen = []

    def sample(self, model_wrap, sigmas, extra_args, callback, noise,
               latent_image=None, denoise_mask=None, disable_pbar=False):
        import comfy.k_diffusion.sampling as k_sampling
        self.call_count += 1
        self.noise_sampler_seen.append(self.extra_options.get("noise_sampler"))
        self.dns_seen.append(k_sampling.default_noise_sampler)

        n_steps = len(sigmas) - 1
        x = torch.zeros(1, 4, 8, 8)
        for i in range(n_steps):
            if i < len(self.denoised_sequence):
                denoised = self.denoised_sequence[i]
            else:
                denoised = torch.zeros(1, 4, 8, 8)
            if callback is not None:
                callback(i, denoised, x, n_steps)
        return x


@pytest.fixture
def mock_sampler():
    return MockSampler


@pytest.fixture
def texture_sigmas():
    """7 sigma values spanning the texture phase (0.5 < sigma < 5.0)."""
    return torch.tensor([3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0])


@pytest.fixture
def constant_denoised():
    """6 linearly increasing denoised tensors (constant change_norm = 16.0)."""
    return [torch.ones(1, 4, 8, 8) * i for i in range(6)]
