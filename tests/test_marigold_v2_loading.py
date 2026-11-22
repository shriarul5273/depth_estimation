"""Offline tests for Marigold V2's trainables.safetensors loading.

The checkpoint keys are "<component>.<module key>" (e.g.
"Diffuser.img_in.lora_A.default.weight", "VAE.decoder.conv_in.weight").
An earlier loader expected unprefixed keys and silently loaded nothing.
"""

import pytest
import torch
import torch.nn as nn

from depth_estimation.models.marigold_v2.configuration_marigold_v2 import MarigoldV2Config
from depth_estimation.models.marigold_v2.modeling_marigold_v2 import MarigoldV2Model


class _Dit(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_in = nn.Linear(4, 4)


class _Vae(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Linear(4, 4)


def _load(monkeypatch, state_dict):
    import huggingface_hub
    import safetensors.torch

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: "unused")
    monkeypatch.setattr(safetensors.torch, "load_file", lambda *a, **k: state_dict)
    dit, vae = _Dit(), _Vae()
    MarigoldV2Model(MarigoldV2Config())._load_trainable_weights(vae, dit)
    return dit, vae


def test_loads_prefixed_components(monkeypatch):
    sd = {
        "Diffuser.img_in.weight": torch.full((4, 4), 2.0),
        "VAE.decoder.weight": torch.full((4, 4), 3.0),
        "iREPAStudentProjector.out.weight": torch.zeros(2, 2),  # training-only
    }
    dit, vae = _load(monkeypatch, sd)
    assert torch.equal(dit.img_in.weight, sd["Diffuser.img_in.weight"])
    assert torch.equal(vae.decoder.weight, sd["VAE.decoder.weight"])


def test_unmatched_keys_raise(monkeypatch):
    with pytest.raises(KeyError, match="match no parameter"):
        _load(monkeypatch, {"Diffuser.not_a_module.weight": torch.zeros(4, 4)})


def test_missing_diffuser_raises(monkeypatch):
    with pytest.raises(KeyError, match="no 'Diffuser.' weights"):
        _load(monkeypatch, {"VAE.decoder.weight": torch.zeros(4, 4)})
