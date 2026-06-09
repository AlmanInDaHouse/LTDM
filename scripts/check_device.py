#!/usr/bin/env python3
"""Gate G-0b: verify the accelerator stack before S-H3 (PLAN-LTDM-H §0).

Exit 0 if the required backend is live and a smoke matmul runs; exit 1 otherwise.
On ROCm, torch exposes the torch.cuda namespace (HIP masquerade) - this script
reports which one you actually have.
"""
import argparse, sys, time

import torch

sys.path.insert(0, "src")
from ltdm import device as dv  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--require", choices=["rocm", "cuda", "cpu", "any"], default="any")
    args = ap.parse_args()

    be = dv.backend()
    dev = dv.get_device()
    print(f"torch          : {torch.__version__}")
    print(f"backend        : {be} (torch.version.hip={getattr(torch.version, 'hip', None)}, "
          f"torch.version.cuda={torch.version.cuda})")
    print(f"device         : {dev}")
    if be != "cpu":
        print(f"device name    : {torch.cuda.get_device_name(0)}")

    x = torch.randn(512, 512, device=dev)
    dv.accel_sync(); t0 = time.perf_counter()
    for _ in range(10):
        x = x @ x.T / 512
    dv.accel_sync(); dt = (time.perf_counter() - t0) / 10
    print(f"matmul 512x512 : {dt*1e3:.2f} ms/iter | mem_allocated={dv.accel_mem_allocated()/2**20:.1f} MiB")

    ok = args.require in ("any", be)
    print("G-0b:", "PASS" if ok else f"FAIL (required {args.require}, got {be})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
