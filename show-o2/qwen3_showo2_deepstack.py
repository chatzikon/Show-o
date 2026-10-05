"""V2 feature alignment and alpha fusion (Transformers 4.57.6 tuple API)."""
import math
from contextlib import contextmanager
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F


class DeepStackAdapter(nn.Module):
    def __init__(self, showo_dim=1536, qwen_dim=2560, levels=3):
        super().__init__()
        if levels < 1:
            raise ValueError("Qwen must expose at least one DeepStack level")
        # Preserve V1's proj keys for explicit warm starts.
        self.proj = nn.Linear(showo_dim, qwen_dim)
        self.deepstack = nn.ModuleList(nn.Linear(showo_dim, qwen_dim) for _ in range(levels))

    def forward(self, features):
        return self.proj(features), [head(features) for head in self.deepstack]


def unpack_visual(output):
    """Accept the pinned tuple API and explicit ModelOutput fields."""
    if hasattr(output, "pooler_output"):
        final, deep = output.pooler_output, output.deepstack_features
    elif isinstance(output, (tuple, list)) and len(output) == 2:
        final, deep = output
    else:
        raise TypeError("Unsupported Qwen visual output; expected final and DeepStack features")
    if not isinstance(deep, (tuple, list)) or not deep:
        raise ValueError("Missing native Qwen DeepStack features")
    return final, list(deep)


def blend(native, adapted, alpha):
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0, 1]")
    if native.shape != adapted.shape:
        raise ValueError(f"Feature shapes differ: {native.shape} versus {adapted.shape}")
    # Exact endpoints, including when the unused representation has NaNs.
    if alpha == 1:
        return native
    adapted = adapted.to(native)
    if alpha == 0:
        return adapted
    return alpha * native + (1 - alpha) * adapted


def fuse_features(native, adapted, alpha):
    qfinal, qdeep = native
    sfinal, sdeep = adapted
    if len(qdeep) != len(sdeep) or not qdeep:
        raise ValueError("DeepStack level count mismatch")
    return blend(qfinal, sfinal, alpha), [blend(q, s, alpha) for q, s in zip(qdeep, sdeep)]


def feature_loss(predicted, target, deepstack_weight=1.0):
    def loss(p, t):
        if p.shape != t.shape:
            raise ValueError("Alignment target shape mismatch")
        p, t = p.float(), t.detach().float()
        return F.mse_loss(p, t) + 0.1 * (1 - F.cosine_similarity(p, t, dim=-1).mean())
    p, pd = predicted
    t, td = target
    if len(pd) != len(td) or not pd:
        raise ValueError("Alignment level count mismatch")
    final = loss(p, t)
    deep = torch.stack([loss(a, b) for a, b in zip(pd, td)]).mean()
    return final + deepstack_weight * deep, final, deep


def load_adapter(checkpoint, device="cpu", dtype=torch.float32):
    if checkpoint.get("fusion_version") != 2:
        raise ValueError("V2 inference requires trained DeepStack heads; V1 is only a training warm start")
    adapter = DeepStackAdapter(**checkpoint["adapter_config"])
    adapter.load_state_dict(checkpoint["adapter"], strict=True)
    return adapter.to(device=device, dtype=dtype)


@contextmanager
def patched_image_features(qwen, final, deep):
    """Single-image generation, preserving Qwen's native DeepStack injection path."""
    model = qwen.model
    original = model.get_image_features
    had_override = "get_image_features" in model.__dict__
    old_rope = getattr(model, "rope_deltas", None)
    def replacement(self, pixel_values, image_grid_thw=None, **kwargs):
        if image_grid_thw is None or image_grid_thw.shape[0] != 1:
            raise ValueError("V2 generation currently supports one image per call")
        return (final,), deep
    model.get_image_features = MethodType(replacement, model)
    model.rope_deltas = None
    try:
        yield
    finally:
        if had_override:
            model.get_image_features = original
        else:
            del model.get_image_features
        model.rope_deltas = old_rope