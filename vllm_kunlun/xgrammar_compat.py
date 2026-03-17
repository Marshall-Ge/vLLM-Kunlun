"""Compatibility helpers for structured output on Kunlun."""

from __future__ import annotations

import importlib
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


def patch_xgrammar() -> bool:
    """Patch xgrammar token-bitmask application for Kunlun.

    In Kunlun deployments the runtime stack is not compatible with
    xgrammar's default Triton path. Force a safe fallback implementation and
    accept an optional ``backend=`` keyword so patched vLLM call sites remain
    compatible across xgrammar versions.
    """

    try:
        xgrammar_mod = importlib.import_module("xgrammar")
        matcher_mod = importlib.import_module("xgrammar.matcher")
        cpu_mod = importlib.import_module(
            "xgrammar.kernels.apply_token_bitmask_inplace_cpu"
        )
        torch_compile_mod = importlib.import_module(
            "xgrammar.kernels.apply_token_bitmask_inplace_torch_compile"
        )
    except Exception:
        return False

    if getattr(matcher_mod, "_vllm_kunlun_patched", False):
        return True

    apply_cpu = cpu_mod.apply_token_bitmask_inplace_cpu
    fallback = torch_compile_mod.apply_token_bitmask_inplace_torch_compile

    def _apply_token_bitmask_inplace(
        logits: Any,
        bitmask: Any,
        *,
        vocab_size=None,
        indices=None,
        backend=None,
    ):
        del backend

        if bitmask.device != logits.device:
            raise ValueError(
                "logits and bitmask should be on the same device. "
                + f"But got logits.device: {logits.device}, "
                + f"bitmask.device: {bitmask.device}"
            )

        if logits.device.type == "cpu":
            return apply_cpu(logits, bitmask, vocab_size, indices)

        return fallback(logits, bitmask, vocab_size, indices)

    matcher_mod.apply_token_bitmask_inplace = _apply_token_bitmask_inplace
    matcher_mod._vllm_kunlun_patched = True
    xgrammar_mod.apply_token_bitmask_inplace = _apply_token_bitmask_inplace
    logger.info(
        "Patched xgrammar token bitmask application for Kunlun "
        "compatibility."
    )
    return True
