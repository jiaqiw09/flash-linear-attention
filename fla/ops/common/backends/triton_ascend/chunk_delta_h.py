# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp2


@triton.heuristics({
    "USE_G": lambda args: args["g"] is not None,
    "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
    "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
    "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
})
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_h_triton_ascend_kernel(
    k,
    u,
    w,
    v_new,
    g,
    h,
    h0,
    ht,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
):
    # 1D grid: logical tile = (batch, value-head, V-block).
    # Keep coreDim under 65535 in the Python verifier.
    pid = tl.program_id(0)
    i_v = pid % NV
    i_nh = pid // NV
    i_b = i_nh // HV
    i_hv = i_nh % HV
    i_h = i_hv // (HV // H)
    nt = tl.cdiv(T, BT)

    o_t = tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V

    h_state = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        h_state += tl.load(p_h0, mask=m_k[:, None] & m_v[None, :], other=0.).to(tl.float32)

    base_t = i_b * T
    base_k = (base_t * H + i_h) * K
    base_hv_k = (base_t * HV + i_hv) * K
    base_hv_v = (base_t * HV + i_hv) * V
    base_h = (i_b * nt * HV + i_hv) * K * V

    for i_t in range(0, nt):
        if i_t * BT >= T:
            return

        p_h = h + base_h + i_t * HV * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_h, h_state.to(p_h.dtype.element_ty), mask=m_k[:, None] & m_v[None, :])

        t = i_t * BT + o_t
        m_t = t < T

        p_w = w + base_hv_k + t[:, None] * HV * K + o_k[None, :]
        w_tile = tl.load(p_w, mask=m_t[:, None] & m_k[None, :], other=0.)
        projected = tl.dot(w_tile, h_state.to(w_tile.dtype))

        p_u = u + base_hv_v + t[:, None] * HV * V + o_v[None, :]
        u_tile = tl.load(p_u, mask=m_t[:, None] & m_v[None, :], other=0.)
        new_v = u_tile - projected

        if SAVE_NEW_VALUE:
            p_v_new = v_new + base_hv_v + t[:, None] * HV * V + o_v[None, :]
            tl.store(p_v_new, new_v.to(p_v_new.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])

        if USE_G:
            last_idx = min((i_t + 1) * BT, T) - 1
            p_g = g + base_t * HV + t * HV + i_hv
            g_tile = tl.load(p_g, mask=m_t, other=0.).to(tl.float32)
            g_last = tl.load(g + (base_t + last_idx) * HV + i_hv).to(tl.float32)
            new_v = new_v * tl.where(m_t, exp2(g_last - g_tile), 0.)[:, None]
            h_state *= exp2(g_last)

        p_k = k + base_k + t[:, None] * H * K + o_k[None, :]
        k_tile = tl.load(p_k, mask=m_t[:, None] & m_k[None, :], other=0.)
        h_state += tl.dot(tl.trans(k_tile), new_v.to(k_tile.dtype))

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, h_state.to(p_ht.dtype.element_ty), mask=m_k[:, None] & m_v[None, :])


def chunk_gated_delta_rule_fwd_h_triton_ascend(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    B, T, H, K = k.shape
    HV, V = u.shape[2], u.shape[-1]
    BT = chunk_size
    BK = max(16, triton.next_power_of_2(K))
    BV = min(32, max(16, triton.next_power_of_2(V)))
    NV = triton.cdiv(V, BV)
    logical_programs = B * HV * NV
    if logical_programs > 65535:
        raise RuntimeError(f"Triton-Ascend coreDim would exceed 65535: {logical_programs}")

    h = k.new_empty(B, triton.cdiv(T, BT), HV, K, V)
    v_new = torch.empty_like(u) if save_new_value else None
    final_state = k.new_empty(B, HV, K, V, dtype=torch.float32) if output_final_state else None

    chunk_gated_delta_rule_fwd_h_triton_ascend_kernel[(logical_programs,)](
        k=k,
        u=u,
        w=w,
        v_new=v_new,
        g=g,
        h=h,
        h0=initial_state,
        ht=final_state,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        NV=NV,
    )
    return h, v_new, final_state
