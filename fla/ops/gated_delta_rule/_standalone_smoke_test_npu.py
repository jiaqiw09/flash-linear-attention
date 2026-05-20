"""Standalone smoke test for ALL 4 NPU kernels in this directory.

Covers:
    [1] fused_recurrent_npu.py      -> fused_recurrent_gated_delta_rule_npu
    [2] gate_npu.py                 -> gdn_gate_chunk_cumsum_npu
    [3] chunk_fwd_npu.py            -> chunk_gated_delta_rule_fwd_intra_npu
    [4] wy_fast_npu.py              -> covered indirectly via [3]
                                       (chunk_fwd_intra internally calls
                                        recompute_w_u_fwd_npu from wy_fast_npu)

Why this script is "standalone"
-------------------------------
The fla top-level package cannot be imported on this Triton version: an
unrelated kernel (`fla/ops/simple_gla/parallel.py`) trips the new Triton 3.x
JIT source parser because of a 3-layer decorator stack
(`@triton.heuristics + @triton.autotune + @triton.jit`). That failure short-
circuits `fla/__init__.py` so even `from fla.utils import ...` fails.

This script pre-registers stub modules in sys.modules for every fla.* path
the four NPU files import, then loads the NPU files via importlib by file
path. The stubs provide the minimal helpers actually used in the NPU code
paths exercised here (cu_seqlens=None, no autograd, etc.).

How to run
----------
    cd flash-linear-attention
    python fla/ops/gated_delta_rule/_standalone_smoke_test_npu.py
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

import torch
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# 1. NPU device
# ----------------------------------------------------------------------------
try:
    import torch_npu  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise SystemExit("torch_npu not installed; this test must run on an Ascend host.") from e

assert hasattr(torch, "npu") and torch.npu.is_available(), "NPU device not available."

# ----------------------------------------------------------------------------
# 2. Pre-stub fla.* so the NPU files can `from fla... import ...` cleanly
# ----------------------------------------------------------------------------
_HERE = pathlib.Path(__file__).resolve().parent


def _stub(name: str, attrs: dict | None = None) -> types.ModuleType:
    if name in sys.modules:
        m = sys.modules[name]
    else:
        m = types.ModuleType(name)
        sys.modules[name] = m
    for k, v in (attrs or {}).items():
        setattr(m, k, v)
    return m


def _noop_decorator(fn=None, **kwargs):
    """Stand-in for `input_guard`, `autocast_custom_fwd`, `autocast_custom_bwd`.

    These decorators are only meaningful at training/AMP time. For a smoke
    test (no autograd, all inputs already contiguous on the right device)
    a no-op is sufficient.
    """
    if fn is None:
        return lambda f: f
    return fn


def _prepare_chunk_indices_stub(*args, **kwargs):
    raise RuntimeError(
        "prepare_chunk_indices() was called -- this stub is only safe when "
        "cu_seqlens is None throughout the smoke test.",
    )


_stub("fla")
_stub("fla.ops")
_stub("fla.ops.gated_delta_rule")
_stub("fla.utils", {
    "input_guard": _noop_decorator,
    "autocast_custom_fwd": _noop_decorator,
    "autocast_custom_bwd": _noop_decorator,
})
_stub("fla.ops.utils", {"prepare_chunk_indices": _prepare_chunk_indices_stub})
_stub("fla.ops.utils.index", {"prepare_chunk_indices": _prepare_chunk_indices_stub})


# ----------------------------------------------------------------------------
# 3. Load the 4 NPU modules (order matters: chunk_fwd_npu depends on wy_fast_npu)
# ----------------------------------------------------------------------------
def _load(fq_name: str, filename: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(fq_name, _HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fq_name] = mod              # register BEFORE exec for cross-imports
    spec.loader.exec_module(mod)
    return mod


fr_mod = _load("fla.ops.gated_delta_rule.fused_recurrent_npu", "fused_recurrent_npu.py")
gate_mod = _load("fla.ops.gated_delta_rule.gate_npu", "gate_npu.py")
wy_mod = _load("fla.ops.gated_delta_rule.wy_fast_npu", "wy_fast_npu.py")
chunk_mod = _load("fla.ops.gated_delta_rule.chunk_fwd_npu", "chunk_fwd_npu.py")

fused_recurrent_gated_delta_rule_npu = fr_mod.fused_recurrent_gated_delta_rule_npu
gdn_gate_chunk_cumsum_npu = gate_mod.gdn_gate_chunk_cumsum_npu
fused_gdn_gate_npu = gate_mod.fused_gdn_gate_npu
chunk_gated_delta_rule_fwd_intra_npu = chunk_mod.chunk_gated_delta_rule_fwd_intra_npu
recompute_w_u_fwd_npu = wy_mod.recompute_w_u_fwd_npu

print("[bootstrap] all 4 NPU modules loaded ✓\n")


# ============================================================================
# PyTorch references (self-contained, no fla.* deps)
# ============================================================================
@torch.no_grad()
def ref_fused_recurrent(q, k, v, g, beta, initial_state=None, scale=None, use_qk_l2norm=True):
    """Naive recurrent gated delta rule -- one token at a time."""
    B, T, H, K = q.shape
    _, _, HV, V = v.shape
    if scale is None:
        scale = K ** -0.5

    qf, kf, vf, gf, bf = q.float(), k.float(), v.float(), g.float(), beta.float()
    if use_qk_l2norm:
        qf = qf / (qf.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
        kf = kf / (kf.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
    qf = qf * scale

    if HV != H:
        repeats = HV // H
        qf = qf.repeat_interleave(repeats, dim=2)
        kf = kf.repeat_interleave(repeats, dim=2)

    h = torch.zeros(B, HV, K, V, dtype=torch.float32, device=q.device)
    if initial_state is not None:
        h = h + initial_state.float()
    o = torch.empty(B, T, HV, V, dtype=v.dtype, device=v.device)

    for t in range(T):
        qt, kt, vt, gt, bt = qf[:, t], kf[:, t], vf[:, t], gf[:, t], bf[:, t]
        h = h * gt[:, :, None, None].exp()
        h_t_k = (h * kt[:, :, :, None]).sum(dim=2)
        v_new = bt[:, :, None] * (vt - h_t_k)
        h = h + kt[:, :, :, None] * v_new[:, :, None, :]
        ot = (h * qt[:, :, :, None]).sum(dim=2)
        o[:, t] = ot.to(v.dtype)
    return o, h


@torch.no_grad()
def ref_gdn_gate_chunk_cumsum(g, A_log, dt_bias, chunk_size, scale=None):
    """Reference for `gdn_gate_chunk_cumsum_scalar_kernel` (forward, REVERSE=False).

    Per-chunk: cumsum(-exp(A_log) * softplus(g + dt_bias)) along the T axis,
    then optionally multiplied by `scale`.
    """
    B, T, H = g.shape
    BT = chunk_size
    gf = g.float()
    if dt_bias is not None:
        gf = gf + dt_bias.float()[None, None, :]
    gate = -A_log.float().exp()[None, None, :] * F.softplus(gf)  # [B, T, H]

    out = torch.empty_like(gate)
    NT = (T + BT - 1) // BT
    for nt in range(NT):
        s, e = nt * BT, min((nt + 1) * BT, T)
        out[:, s:e] = gate[:, s:e].cumsum(dim=1)
    if scale is not None:
        out = out * scale
    return out


@torch.no_grad()
def ref_chunk_fwd_intra(k, v, g_log2, beta, chunk_size=64):
    """Reference for `chunk_gated_delta_rule_fwd_intra_npu`.

    Produces:
        w [B, T, HV, K], u [B, T, HV, V], A [B, T, HV, BT]
    Where, per chunk:
        A_raw[i, j] = beta[i] * exp2(g[i] - g[j]) * <k[i], k[j]>     for i > j
        A_raw[i, j] = 0                                                otherwise
        A_inv      = (I + A_raw)^-1
        w[i] = sum_j A_inv[i, j] * (k[j] * beta[j] * exp2(g[j]))
        u[i] = sum_j A_inv[i, j] * (v[j] * beta[j])
    """
    B, T, H, K = k.shape
    _, _, HV, V = v.shape
    BT = chunk_size

    if HV != H:
        repeats = HV // H
        k_hv = k.repeat_interleave(repeats, dim=2)
    else:
        k_hv = k

    w = torch.zeros(B, T, HV, K, dtype=k.dtype, device=k.device)
    u = torch.zeros_like(v)
    A_out = torch.zeros(B, T, HV, BT, dtype=k.dtype, device=k.device)

    NT = (T + BT - 1) // BT
    for nt in range(NT):
        s = nt * BT
        e = min(s + BT, T)
        L = e - s

        kc = k_hv[:, s:e].float().permute(0, 2, 1, 3)  # [B, HV, L, K]
        vc = v[:, s:e].float().permute(0, 2, 1, 3)      # [B, HV, L, V]
        gc = g_log2[:, s:e].float().permute(0, 2, 1)    # [B, HV, L]
        bc = beta[:, s:e].float().permute(0, 2, 1)      # [B, HV, L]

        KKt = torch.matmul(kc, kc.transpose(-2, -1))    # [B, HV, L, L]
        gate = torch.exp2(gc[..., :, None] - gc[..., None, :])  # [B, HV, L, L]

        A_raw = bc[..., :, None] * gate * KKt           # [B, HV, L, L]
        idx = torch.arange(L, device=k.device)
        mask_lt = (idx[:, None] > idx[None, :]).to(A_raw.dtype)
        A_raw = A_raw * mask_lt

        I_mat = torch.eye(L, device=k.device, dtype=A_raw.dtype).expand(*A_raw.shape)
        A_inv = torch.linalg.solve(I_mat + A_raw, I_mat)  # [B, HV, L, L]

        A_out[:, s:e, :, :L] = A_inv.permute(0, 2, 1, 3).to(A_out.dtype)

        kbg = kc * bc[..., None] * torch.exp2(gc)[..., None]    # [B, HV, L, K]
        w_c = torch.matmul(A_inv, kbg).permute(0, 2, 1, 3)      # [B, L, HV, K]
        w[:, s:e] = w_c.to(w.dtype)

        vb = vc * bc[..., None]                                  # [B, HV, L, V]
        u_c = torch.matmul(A_inv, vb).permute(0, 2, 1, 3)
        u[:, s:e] = u_c.to(u.dtype)

    return w, u, A_out


# ============================================================================
# Compare helper
# ============================================================================
def _cmp(tag, a, b, atol=2e-3, rtol=2e-3):
    a_cpu = a.detach().float().cpu()
    b_cpu = b.detach().float().cpu()
    abs_err = (a_cpu - b_cpu).abs()
    rel_err = abs_err / b_cpu.abs().clamp_min(1e-6)
    ok = torch.allclose(a_cpu, b_cpu, atol=atol, rtol=rtol)
    print(
        f"  [{tag}] max_abs={abs_err.max().item():.3e}  "
        f"mean_abs={abs_err.mean().item():.3e}  "
        f"max_rel={rel_err.max().item():.3e}  "
        f"allclose={ok}",
    )
    return ok


# ============================================================================
# Test [1] fused_recurrent_npu
# ============================================================================
def test_fused_recurrent():
    print("=" * 70)
    print("[TEST 1] fused_recurrent_gated_delta_rule_npu")
    print("=" * 70)
    cases = [
        (1, 16, 1, 1, 64, 64, torch.float32, "smallest fp32"),
        (2, 32, 2, 2, 64, 64, torch.float32, "equal H=HV fp32"),
        (1, 64, 2, 4, 64, 64, torch.float32, "GVA HV=2*H fp32"),
        (2, 64, 2, 2, 64, 64, torch.float16, "fp16"),
    ]
    all_ok = True
    for i, (B, T, H, HV, K, V, dtype, label) in enumerate(cases):
        print(f"\n  case {i}: {label}  B={B} T={T} H={H} HV={HV} K={K} V={V} dtype={dtype}")
        torch.manual_seed(100 + i)
        q = torch.randn(B, T, H, K, dtype=dtype, device="npu")
        k = torch.randn(B, T, H, K, dtype=dtype, device="npu")
        v = torch.randn(B, T, HV, V, dtype=dtype, device="npu")
        g = F.logsigmoid(torch.rand(B, T, HV, device="npu", dtype=torch.float32)).to(dtype)
        beta = torch.rand(B, T, HV, dtype=dtype, device="npu").sigmoid()
        h0 = torch.randn(B, HV, K, V, device="npu", dtype=torch.float32)

        try:
            o_npu, ht_npu = fused_recurrent_gated_delta_rule_npu(
                q=q.clone(), k=k.clone(), v=v.clone(),
                g=g.clone(), beta=beta.clone(),
                initial_state=h0.clone(),
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    NPU kernel raised: {type(exc).__name__}: {exc}")
            all_ok = False
            continue

        o_ref, ht_ref = ref_fused_recurrent(q, k, v, g, beta, initial_state=h0, use_qk_l2norm=True)
        atol = 5e-3 if dtype != torch.float32 else 2e-3
        ok_o = _cmp("o ", o_npu, o_ref, atol=atol, rtol=atol)
        ok_h = _cmp("ht", ht_npu, ht_ref, atol=atol, rtol=atol)
        all_ok &= ok_o and ok_h
    return all_ok


# ============================================================================
# Test [2] gate_npu (gdn_gate_chunk_cumsum)
# ============================================================================
def test_gate_chunk_cumsum():
    print("\n" + "=" * 70)
    print("[TEST 2] gdn_gate_chunk_cumsum_npu  (gate.py path)")
    print("=" * 70)
    cases = [
        # (B, T, H, dtype, with_bias, with_scale, label)
        (1, 64, 4, torch.float32, False, None, "T=BT no-bias"),
        (2, 128, 4, torch.float32, True, None, "T=2*BT with-bias"),
        (1, 96, 4, torch.float32, True, 0.5, "T not div BT, with-scale"),
        (2, 64, 8, torch.float16, True, None, "fp16 with-bias"),
    ]
    all_ok = True
    for i, (B, T, H, dtype, with_bias, scale, label) in enumerate(cases):
        print(f"\n  case {i}: {label}  B={B} T={T} H={H} dtype={dtype}")
        torch.manual_seed(200 + i)
        g = torch.randn(B, T, H, dtype=dtype, device="npu")
        A_log = torch.randn(H, dtype=torch.float32, device="npu")
        dt_bias = torch.randn(H, dtype=torch.float32, device="npu") if with_bias else None

        try:
            out_npu = gdn_gate_chunk_cumsum_npu(
                g=g.clone(),
                A_log=A_log.clone(),
                chunk_size=64,
                scale=scale,
                dt_bias=dt_bias.clone() if dt_bias is not None else None,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    NPU kernel raised: {type(exc).__name__}: {exc}")
            all_ok = False
            continue

        out_ref = ref_gdn_gate_chunk_cumsum(g, A_log, dt_bias, chunk_size=64, scale=scale)
        atol = 5e-3 if dtype != torch.float32 else 2e-3
        all_ok &= _cmp("out", out_npu, out_ref, atol=atol, rtol=atol)
    return all_ok


# ============================================================================
# Test [3] chunk_fwd_intra_npu (covers chunk_fwd_npu + wy_fast_npu)
# ============================================================================
def test_chunk_fwd_intra():
    print("\n" + "=" * 70)
    print("[TEST 3] chunk_gated_delta_rule_fwd_intra_npu  (chunk_fwd + wy_fast)")
    print("=" * 70)
    cases = [
        # (B, T, H, HV, K, V, dtype, label)
        (1, 64, 1, 1, 64, 64, torch.float32, "1 chunk fp32"),
        (2, 128, 2, 2, 64, 64, torch.float32, "2 chunks fp32"),
        (1, 64, 2, 4, 64, 64, torch.float32, "GVA HV=2*H fp32"),
        (1, 128, 2, 2, 64, 64, torch.float16, "fp16"),
    ]
    all_ok = True
    for i, (B, T, H, HV, K, V, dtype, label) in enumerate(cases):
        print(f"\n  case {i}: {label}  B={B} T={T} H={H} HV={HV} K={K} V={V} dtype={dtype}")
        torch.manual_seed(300 + i)
        k = torch.randn(B, T, H, K, dtype=dtype, device="npu")
        v = torch.randn(B, T, HV, V, dtype=dtype, device="npu")
        # g must be chunk-cumsum'd already (kernel does exp2(g[i]-g[j])).
        # Build a plausible one: per-chunk cumsum of small negative numbers.
        BT = 64
        g_raw = -torch.rand(B, T, HV, dtype=torch.float32, device="npu") * 0.1
        g = torch.empty_like(g_raw)
        for s in range(0, T, BT):
            e = min(s + BT, T)
            g[:, s:e] = g_raw[:, s:e].cumsum(dim=1)
        g = g.to(dtype)
        beta = torch.rand(B, T, HV, dtype=dtype, device="npu").sigmoid()

        try:
            w_npu, u_npu, A_npu = chunk_gated_delta_rule_fwd_intra_npu(
                k=k.clone(), v=v.clone(), g=g.clone(), beta=beta.clone(),
                chunk_size=BT,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    NPU kernel raised: {type(exc).__name__}: {exc}")
            all_ok = False
            continue

        w_ref, u_ref, A_ref = ref_chunk_fwd_intra(k, v, g, beta, chunk_size=BT)
        atol = 5e-3 if dtype != torch.float32 else 2e-3
        all_ok &= _cmp("w ", w_npu, w_ref, atol=atol, rtol=atol)
        all_ok &= _cmp("u ", u_npu, u_ref, atol=atol, rtol=atol)
        all_ok &= _cmp("A ", A_npu, A_ref, atol=atol, rtol=atol)
    return all_ok


# ============================================================================
# Main
# ============================================================================
def main() -> int:
    results = {
        "fused_recurrent": test_fused_recurrent(),
        "gate_chunk_cumsum": test_gate_chunk_cumsum(),
        "chunk_fwd_intra (+ wy_fast)": test_chunk_fwd_intra(),
    }
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for name, ok in results.items():
        print(f"  {name:35s} {'PASS ✅' if ok else 'FAIL ❌'}")
    all_pass = all(results.values())
    print(f"\nOVERALL: {'PASS ✅' if all_pass else 'FAIL ❌'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
