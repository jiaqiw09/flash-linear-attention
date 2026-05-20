# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend backend for common FLA chunk operations.

This backend is intentionally registered as a separate backend type instead of
sharing the CUDA Triton kernels. Triton-Ascend accepts Triton-like code, but its
launch model, physical-core limits, UB pressure, and supported tuning knobs are
different enough that kernels should be introduced behind their own verifier.
"""

from __future__ import annotations

from importlib.util import find_spec

import torch

from fla.ops.backends import BaseBackend


class TritonAscendBackend(BaseBackend):
    """Opt-in Triton-Ascend backend for Ascend NPU."""

    backend_type = "triton_ascend"
    package_name = None
    env_var = "FLA_TRITON_ASCEND"
    default_enable = False
    priority = 1

    @classmethod
    def is_available(cls) -> bool:
        if find_spec("torch_npu") is None or find_spec("triton") is None:
            return False
        import torch_npu  # noqa: F401

        return hasattr(torch, "npu") and torch.npu.is_available()

    def chunk_gated_delta_rule_fwd_h_verifier(
        self,
        k: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        g: torch.Tensor | None = None,
        gk: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        chunk_size: int = 64,
        save_new_value: bool = True,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
    ) -> tuple[bool, str | None]:
        if k.device.type != "npu":
            return False, "Triton-Ascend backend requires NPU tensors"
        if cu_seqlens is not None or cu_seqlens_cpu is not None or chunk_indices is not None:
            return False, "first Triton-Ascend draft supports fixed-length batches only"
        if state_v_first:
            return False, "first Triton-Ascend draft supports [K, V] state layout only"
        if gk is not None:
            return False, "first Triton-Ascend draft supports scalar gate g, not per-K gate gk"
        if chunk_size != 64:
            return False, "first Triton-Ascend draft keeps the existing GDR chunk_size=64 contract"
        if k.shape[-1] > 64:
            return False, f"first Triton-Ascend draft supports K <= 64, got K={k.shape[-1]}"

        B, T, H, K = k.shape
        HV, V = u.shape[2], u.shape[-1]
        nv = (V + 31) // 32
        logical_programs = B * HV * nv
        if logical_programs > 65535:
            return False, f"coreDim would exceed 65535 ({logical_programs})"
        if H > HV or HV % H != 0:
            return False, "expected HV to be a multiple of H for grouped value attention"
        return True, None

    def chunk_gated_delta_rule_fwd_h(
        self,
        k: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        g: torch.Tensor | None = None,
        gk: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        chunk_size: int = 64,
        save_new_value: bool = True,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        from fla.ops.common.backends.triton_ascend.chunk_delta_h import (
            chunk_gated_delta_rule_fwd_h_triton_ascend,
        )

        return chunk_gated_delta_rule_fwd_h_triton_ascend(
            k=k,
            w=w,
            u=u,
            g=g,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
            save_new_value=save_new_value,
        )
