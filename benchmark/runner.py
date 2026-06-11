"""ActivationScope memory & throughput benchmark.

Compares ActivationScope (C++ hooks) against naive Python forward hooks
using subprocess isolation for clean memory measurements.  Each mode is
measured across *N* independent subprocess passes and results are averaged.

Memory is measured via ``resource.getrusage().ru_maxrss`` (peak resident
set size) — a reliable cross-platform metric that reflects actual physical
memory use without virtual-address-space noise.

Usage:
  PYTHONPATH=. python -m benchmark.runner                     # toy model (default)
  PYTHONPATH=. python -m benchmark.runner --model resnet18    # pretrained ResNet-18
  PYTHONPATH=. python -m benchmark.runner --passes 5          # fewer passes
"""

import math
import os
import subprocess
import sys

import torch


# ══════════════════════════════════════════════════════════════════════
#  subprocess helpers
# ══════════════════════════════════════════════════════════════════════


def _run_worker(
    mode: str, model: str, n: int, batch: int, dim: int, layers: int
) -> dict | None:
    """Spawn ``benchmark._worker`` in a clean subprocess, parse KEY=VALUE stdout.

    Returns ``None`` if the subprocess fails (non-zero exit or unparseable).
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()

    r = subprocess.run(
        [
            sys.executable, "-u", "-m", "benchmark._worker",
            "--mode", mode,
            "--model", model,
            "--n", str(n),
            "--batch", str(batch),
            "--dim", str(dim),
            "--layers", str(layers),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )

    if r.returncode != 0:
        msg = r.stderr.strip()[:400] or r.stdout.strip()[:400]
        print(f"\n      [FAIL] exit={r.returncode}  {msg}")
        return None

    out: dict = {}
    for line in r.stdout.strip().split("\n"):
        if "=" in line:
            k, v = line.split("=", 1)
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v

    required = {"MEM_PEAK", "MS_PER_FWD", "DATA_MIB", "READBACK_MS"}
    if not required.issubset(out.keys()):
        print(f"\n      [FAIL] missing keys: {required - out.keys()}")
        return None
    return out


# ══════════════════════════════════════════════════════════════════════
#  stats helpers
# ══════════════════════════════════════════════════════════════════════


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Sample mean and sample standard deviation."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mu = sum(values) / n
    sd = math.sqrt(sum((v - mu) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    return mu, sd


def _sem(sd: float, n: int) -> float:
    """Standard error of the mean."""
    return sd / math.sqrt(n) if n > 0 else 0.0


def _fmt_val(mu: float, sd: float, unit: str = "", decimals: int = 1) -> str:
    """Format as ``123.4 ± 5.6 MiB`` when sd is non-trivial, else just ``123.4 MiB``."""
    val = f"{mu:.{decimals}f}{unit}"
    if sd < mu * 0.0005:
        return val
    sd_d = decimals + 1 if sd < 1 else decimals
    return f"{val} ± {sd:.{sd_d}f}{unit}"


def _fmt_pct(v1: float, v2: float) -> str:
    """Relative change as a signed percentage, e.g. ``+13.2%``."""
    if v2 == 0:
        return "—"
    pct = (v1 / v2 - 1) * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


# ══════════════════════════════════════════════════════════════════════
#  formatting
# ══════════════════════════════════════════════════════════════════════


def _model_label(model: str) -> str:
    if model == "toy":
        return "toy Sequential(Linear)"
    return "pretrained ResNet-18"


# ══════════════════════════════════════════════════════════════════════
#  main
# ══════════════════════════════════════════════════════════════════════


def main():
    import argparse

    ap = argparse.ArgumentParser(description="ActivationScope benchmark")
    ap.add_argument("--model", choices=("toy", "resnet18"), default="toy")
    ap.add_argument("--n", type=int, default=None, help="number of forwards")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--dim", type=int, default=None, help="hidden dim (toy model)")
    ap.add_argument("--layers", type=int, default=None, help="layers (toy model)")
    ap.add_argument("--passes", type=int, default=10, help="subprocess passes per mode")
    args = ap.parse_args()

    # ── per-model defaults ───────────────────────────────────────────
    model = args.model
    if model == "toy":
        N = args.n or 200
        B = args.batch or 32
        D = args.dim or 256
        L = args.layers or 48
    else:
        N = args.n or 20
        B = args.batch or 8
        D = 0
        L = 0

    passes = args.passes
    label = _model_label(model)

    print("=" * 72)
    print("  ActivationScope — Memory & Throughput Benchmark")
    print(f"  Model: {label}  |  {N} forwards  |  batch={B}  |  {passes} passes")
    if model == "toy":
        print(f"  {L} × Linear({D},{D})  |  input [{B},{D}]")
    else:
        print(f"  input [{B}, 3, 224, 224]")
    print("=" * 72)

    # ── collect passes for one mode ──────────────────────────────────
    def _collect(mode: str, tag: str):
        rss_vals, ms_vals, data_vals, read_vals = [], [], [], []
        n_tracked = 0
        failed = 0
        for p in range(passes):
            if p == 0:
                print(f"\n  [{tag}] {mode} ... ", end="", flush=True)
            else:
                print(".", end="", flush=True)

            r = _run_worker(mode, model, N, B, D, L)
            if r is None:
                failed += 1
                continue
            rss_vals.append(r["MEM_PEAK"])
            ms_vals.append(r["MS_PER_FWD"])
            data_vals.append(r["DATA_MIB"])
            read_vals.append(r["READBACK_MS"])
            n_tracked = int(r.get("TRACKED_LAYERS", n_tracked))

        ok = passes - failed
        color = "❌" if failed > 0 else "✓"
        print(f"\n      {ok}/{passes} ok {color}  ({failed} failures, {n_tracked} layers)")
        return rss_vals, ms_vals, data_vals, read_vals, n_tracked, ok

    # ── run all three modes ──────────────────────────────────────────
    b_rss, b_ms, b_data, b_read, b_layers, b_ok = _collect("none", "1/3")
    n_rss, n_ms, n_data, n_read, n_layers, n_ok = _collect("naive", "2/3")
    s_rss, s_ms, s_data, s_read, s_layers, s_ok = _collect("scope", "3/3")

    # ── compute stats ─────────────────────────────────────────────────
    stats = {}
    for name, rss, ms, rd in [
        ("none", b_rss, b_ms, b_read),
        ("naive", n_rss, n_ms, n_read),
        ("scope", s_rss, s_ms, s_read),
    ]:
        r_mu, r_sd = _mean_std(rss)
        m_mu, m_sd = _mean_std(ms)
        rd_mu, rd_sd = _mean_std(rd) if rd else (0.0, 0.0)
        stats[name] = dict(rss_mu=r_mu, rss_sd=r_sd, ms_mu=m_mu, ms_sd=m_sd,
                           rd_mu=rd_mu, rd_sd=rd_sd, n=len(rss))

    bas = stats["none"]
    nav = stats["naive"]
    scp = stats["scope"]

    # ── summary table ─────────────────────────────────────────────────
    d_avg = sum(b_data) / len(b_data) if b_data else 0

    print(f"\n{'=' * 72}")
    print("  SUMMARY")
    print(f"{'=' * 72}")
    print(f"  {'':<24} {'ms/fwd':>16}  {'peak VMS':>16}  {'data':>10}  {'readback':>10}")

    def _line(name, s):
        ms_str = _fmt_val(s["ms_mu"], s["ms_sd"], "ms", 3) if s["n"] else "—"
        rss_str = _fmt_val(s["rss_mu"], s["rss_sd"], " MiB") if s["n"] else "—"
        dt_str = f"{d_avg:.0f} MiB" if d_avg else "—"
        if name == "No tracking":
            rd_str = "—"
        else:
            rd_str = _fmt_val(s["rd_mu"], s["rd_sd"], "ms", 1) if s["rd_mu"] > 0 else "—"
        print(f"  {name:<24} {ms_str:>16}  {rss_str:>16}  {dt_str:>10}  {rd_str:>10}")

    _line("No tracking", bas)
    _line("Naive Python hooks", nav)
    _line("ActivationScope C++", scp)

    # ── memory comparison ─────────────────────────────────────────────
    print(f"\n  ── Memory ──")
    if nav["n"] and scp["n"]:
        diff = nav["rss_mu"] - scp["rss_mu"]
        pct = abs(diff) / nav["rss_mu"] * 100
        prefix = "Scope uses" if diff < 0 else "Scope saves"
        print(f"  Peak VMS:  Naive {_fmt_val(nav['rss_mu'], nav['rss_sd'], ' MiB')}")
        print(f"             Scope {_fmt_val(scp['rss_mu'], scp['rss_sd'], ' MiB')}")
        # At ~400K MiB VMS, <0.1% difference (~400 MiB) is ASLR noise.
        if pct < 0.1:
            print(f"  {prefix} {abs(diff):.0f} MiB ({pct:.2f}% of VMS) — identical, within ASLR noise")
        else:
            print(f"  {prefix} {abs(diff):.0f} MiB ({pct:.1f}%)")

    # ── throughput comparison ──────────────────────────────────────────
    print(f"\n  ── Throughput ──")
    if bas["n"]:
        print(f"  No tracking:      {bas['ms_mu']:.3f} ms/fwd  (±{bas['ms_sd']:.3f})")
        n_ovh = _fmt_pct(nav["ms_mu"], bas["ms_mu"])
        s_ovh = _fmt_pct(scp["ms_mu"], bas["ms_mu"])
        print(f"  Naive overhead:   {n_ovh} vs baseline  ({nav['ms_mu']:.3f} ± {nav['ms_sd']:.3f} ms/fwd)")
        print(f"  Scope overhead:   {s_ovh} vs baseline  ({scp['ms_mu']:.3f} ± {scp['ms_sd']:.3f} ms/fwd)")

    if nav["n"] and scp["n"]:
        speedup = nav["ms_mu"] / scp["ms_mu"]
        if speedup > 1.0:
            print(f"\n  Scope is {speedup:.2f}x faster than naive  ({nav['ms_mu']:.3f} → {scp['ms_mu']:.3f} ms/fwd)")
        else:
            print(f"\n  Scope is {speedup:.2f}x vs naive  ({nav['ms_mu']:.3f} → {scp['ms_mu']:.3f} ms/fwd)")

    # ── readback ───────────────────────────────────────────────────────
    if scp["rd_mu"] > 0 and d_avg > 0:
        gbps = (d_avg / scp["rd_mu"] * 1000) / 1024
        print(f"\n  ── Readback ──")
        print(f"  Scope zero-copy:  {_fmt_val(scp['rd_mu'], scp['rd_sd'], 'ms', 1)}  ({d_avg:.0f} MiB = {gbps:.0f} GiB/s)")

    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
