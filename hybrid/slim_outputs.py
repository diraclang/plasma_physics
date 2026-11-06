"""Shrink large self-consistent output files by thinning the per-particle histories.

Keeps every field snapshot and every non-history array unchanged, keeps the particle
histories (x, v, K, mu) and the time axis only every `--stride` steps plus the final
step (so restarts still work), and records the kept step indices in `hist_step`.
Each file is verified against the original before it is replaced.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

HISTORY_KEYS = ("x", "v", "K", "mu")


def slim_file(path: Path, stride: int) -> tuple[bool, str]:
    with np.load(path, allow_pickle=False) as d:
        missing = [k for k in ("x", "v", "K", "mu", "t") if k not in d.files]
        if missing:
            return False, f"missing arrays {missing} (not a complete run file)"
        ntime = d["t"].shape[0]
        # Real simulation step of every stored column (identity for files without hist_step).
        old_hist = d["hist_step"] if "hist_step" in d.files else np.arange(ntime, dtype=int)
        if old_hist.shape[0] != ntime:
            return False, "hist_step length does not match the time axis"
        # Keep columns whose *real step number* is a multiple of `stride`, plus the first and last.
        keep = (old_hist % stride == 0)
        keep[0] = True
        keep[-1] = True
        idx_arr = np.nonzero(keep)[0]
        idx = idx_arr.tolist()
        if len(idx) >= ntime:
            return False, "nothing to thin (already at or below this stride)"

        out: dict[str, np.ndarray] = {}
        for key in d.files:
            if key == "hist_step":
                continue
            arr = d[key]
            if key in HISTORY_KEYS or key == "t":
                out[key] = np.ascontiguousarray(arr[:, idx_arr] if arr.ndim >= 2 else arr[idx_arr])
            else:
                out[key] = arr
            del arr
        out["hist_step"] = old_hist[idx_arr]

        fd, tmp_name = tempfile.mkstemp(suffix=".npz", dir=str(path.parent))
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            np.savez_compressed(tmp, **out)
            # verify the thinned file against the original, array by array
            with np.load(tmp, allow_pickle=False) as s:
                for key in d.files:
                    if key == "hist_step":
                        continue
                    a = d[key]
                    expected = a[:, idx_arr] if (key in HISTORY_KEYS and a.ndim >= 2) else (a[idx_arr] if key == "t" else a)
                    if not np.array_equal(s[key], expected):
                        raise RuntimeError(f"verification failed for '{key}'")
                    del a, expected
                if not np.array_equal(s["x"][:, -1], d["x"][:, -1]) or not np.array_equal(s["v"][:, -1], d["v"][:, -1]):
                    raise RuntimeError("final particle state differs")
        except Exception as exc:  # noqa: BLE001
            tmp.unlink(missing_ok=True)
            return False, f"kept original ({exc})"
    old = path.stat().st_size
    os.replace(tmp, path)
    return True, f"{old / 1e9:.2f} GB -> {path.stat().st_size / 1e6:.0f} MB, {len(idx)} stored steps"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+")
    p.add_argument("--stride", type=int, default=50)
    args = p.parse_args()
    for f in args.files:
        path = Path(f)
        try:
            ok, msg = slim_file(path, args.stride)
        except Exception as exc:  # noqa: BLE001
            ok, msg = False, f"skipped ({type(exc).__name__}: {exc})"
        print(f"{'OK  ' if ok else 'SKIP'} {path.name}: {msg}", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
