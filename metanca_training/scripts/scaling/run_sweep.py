"""Launch the full T-sweep for one ablation as isolated subprocesses."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

TS = [1, 2, 4, 8, 16, 32]
REPS = [0, 1, 2]
HERE = Path(__file__).resolve().parent


def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ablation", choices=["fixed5", "varying"], required=True)
    p.add_argument("--metaepochs", type=int, default=1200)
    p.add_argument("--results-dir", type=str, default="results/scaling")
    p.add_argument("--t-list", type=_int_list, default=TS,
                   help="comma-separated T values (default 1,2,4,8,16,32)")
    p.add_argument("--reps", type=_int_list, default=REPS,
                   help="comma-separated rep indices (default 0,1,2)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    # rep-major: complete a full T-curve for rep 0, then rep 1, ... so a full (low-n) curve
    # lands early and statistical significance builds up rep by rep.
    for rep in args.reps:
        for t in args.t_list:
            done = Path(args.results_dir) / args.ablation / f"T{t}_rep{rep}.done"
            if done.exists():
                print(f"skip (done): {done}", flush=True)
                continue
            cmd = [sys.executable, str(HERE / "run_scaling.py"),
                   "--ablation", args.ablation, "--T", str(t), "--rep", str(rep),
                   "--metaepochs", str(args.metaepochs), "--results-dir", args.results_dir]
            print(" ".join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
