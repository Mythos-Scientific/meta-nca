#!/usr/bin/env python
"""Distributed rep-major scheduler for the MetaNCA architecture-scaling sweep.

Runs on the LOCAL box. Distributes (T, rep) training+eval jobs across three GPUs — the
local GB10 and two RunPod 5090s (over SSH) — with a **hard barrier between reps**: every T
runs exactly once (rep 0) before any T runs again, so a full scaling curve exists after the
first pass and each later pass adds a point of statistics to every T.

Within a rep, a dynamic longest-job-first queue keeps the GPUs busy (biggest T -> fastest
free worker), and the rep only advances once all its jobs' `.done` markers exist. Jobs log
to wandb (project architecture-scaling-ablation); results are pulled from RunPod after each
rep. Resumable: already-`.done` jobs are skipped and in-flight jobs are re-attached, so the
scheduler can be killed and restarted safely.
"""

import argparse
import subprocess
import time
from pathlib import Path

LOCAL_DIR = "/home/dan/Projects/meta-nca"
REMOTE_DIR = "/workspace/meta-nca"
SSH = ["ssh", "-p", "13104", "-i", "/home/dan/.ssh/id_ed25519-runpod",
       "-o", "ConnectTimeout=25", "root@162.43.172.165"]
SCP_HOST = "root@162.43.172.165"
POLL_SECS = 60

# Measured (fixed3): single-GPU beats multi-GPU decisively (T=8 single: 11.1 s/metaepoch
# @n=1 after one ~14min compile; multi-GPU: still compiling per-arch programs at 10min with
# GPUs idle). So each 5090 is its own single-GPU worker — two jobs run in parallel on the
# node, both within the same rep (rep-major barrier preserved).
WORKERS = [
    {"name": "runpod0", "kind": "ssh", "cuda": "0", "max_t": 10**9},
    {"name": "runpod1", "kind": "ssh", "cuda": "1", "max_t": 10**9},
    {"name": "gb10", "kind": "local", "cuda": "0", "max_t": 2},  # ~2x slower: small T only
]

RSYNC = ["rsync", "-az", "-e", "ssh -p 13104 -i /home/dan/.ssh/id_ed25519-runpod"]


def sync_job_state(worker: dict, T: int, rep: int, ablation: str) -> None:
    """Before launching on `worker`, pull this run's checkpoint + partial eval rows from the
    OTHER machine (if present) so a job can migrate/resume anywhere. Orbax step dirs are
    finalized atomically by rename, so rsync of completed steps is safe; resume picks the
    latest complete step."""
    ckpt = f"local_rule_checkpoints/scaling_{ablation}_T{T}_rep{rep}"
    jsonl = f"results/scaling/{ablation}/T{T}_rep{rep}.jsonl"
    if worker["kind"] == "local":   # migrating remote -> local: pull
        Path(f"{LOCAL_DIR}/local_rule_checkpoints").mkdir(exist_ok=True)
        Path(f"{LOCAL_DIR}/results/scaling/{ablation}").mkdir(parents=True, exist_ok=True)
        subprocess.run(RSYNC + [f"{SCP_HOST}:{REMOTE_DIR}/{ckpt}",
                                f"{LOCAL_DIR}/local_rule_checkpoints/"],
                       capture_output=True, text=True)
        subprocess.run(RSYNC + [f"{SCP_HOST}:{REMOTE_DIR}/{jsonl}",
                                f"{LOCAL_DIR}/results/scaling/{ablation}/"],
                       capture_output=True, text=True)
    else:                            # migrating local -> remote: push (if local state exists)
        _ssh(f"mkdir -p {REMOTE_DIR}/local_rule_checkpoints {REMOTE_DIR}/results/scaling/{ablation}")
        if Path(f"{LOCAL_DIR}/{ckpt}").exists():
            subprocess.run(RSYNC + [f"{LOCAL_DIR}/{ckpt}",
                                    f"{SCP_HOST}:{REMOTE_DIR}/local_rule_checkpoints/"],
                           capture_output=True, text=True)
        if Path(f"{LOCAL_DIR}/{jsonl}").exists():
            subprocess.run(RSYNC + [f"{LOCAL_DIR}/{jsonl}",
                                    f"{SCP_HOST}:{REMOTE_DIR}/results/scaling/{ablation}/"],
                           capture_output=True, text=True)


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def _ssh(remote: str) -> str:
    return _run(SSH + [remote])


def _run_cmd(T: int, rep: int, ablation: str, metaepochs: int) -> str:
    return (f".venv/bin/python metanca_training/scripts/scaling/run_scaling.py "
            f"--ablation {ablation} --T {T} --rep {rep} --metaepochs {metaepochs}")


def _pgrep_pat(T: int, rep: int, ablation: str) -> str:
    # no trailing space: matches both scheduler-launched (…--rep 0 --metaepochs) and manual runs
    return f"run_scaling.py --ablation {ablation} --T {T} --rep {rep}"


def is_running(worker: dict, T: int, rep: int, ablation: str) -> bool:
    pat = _pgrep_pat(T, rep, ablation)
    if worker["kind"] == "local":
        return bool(_run(["pgrep", "-f", pat]))
    return bool(_ssh(f"pgrep -f '{pat}'"))


def is_done(worker: dict, T: int, rep: int, ablation: str) -> bool:
    done = f"results/scaling/{ablation}/T{T}_rep{rep}.done"
    if worker["kind"] == "local":
        return Path(f"{LOCAL_DIR}/{done}").exists()
    return _ssh(f"test -f {REMOTE_DIR}/{done} && echo yes") == "yes"


def launch(worker: dict, T: int, rep: int, ablation: str, metaepochs: int) -> None:
    sync_job_state(worker, T, rep, ablation)   # enables cross-machine resume/migration
    inner = _run_cmd(T, rep, ablation, metaepochs)
    log = f"sched_{ablation}_T{T}_rep{rep}.log"
    # env vars must come BEFORE nohup (they apply to the command nohup runs). cuda=None ->
    # leave all GPUs visible (RunPod multi-GPU); else pin to one device.
    cuda = "" if worker["cuda"] is None else f"CUDA_VISIBLE_DEVICES={worker['cuda']} "
    # Persistent XLA compilation cache: (arch shape, n_update_steps) programs recur across
    # runs (overlapping archs between reps/T's, and every run's eval of the same 25 val
    # archs), so caching to disk eliminates most repeat compile tax after first encounter.
    cache_dir = f"{LOCAL_DIR}/.jax_cache" if worker["kind"] == "local" else f"{REMOTE_DIR}/.jax_cache"
    envs = (f"{cuda}XLA_PYTHON_CLIENT_PREALLOCATE=false "
            f"JAX_COMPILATION_CACHE_DIR={cache_dir} ")
    if worker["kind"] == "local":
        full = (f"cd {LOCAL_DIR} && {envs}nohup {inner} "
                f"> .superpowers/sdd/runlogs/{log} 2>&1 &")
        subprocess.Popen(["bash", "-lc", full])
    else:
        remote = (f"cd {REMOTE_DIR} && {envs}nohup {inner} </dev/null > {log} 2>&1 & disown")
        # fire-and-forget: ssh can hang holding the session open even after `disown`;
        # we never need its exit status — the is_running/is_done poll verifies the job.
        subprocess.Popen(SSH + ["-n"] + [remote],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  [{worker['name']}] launched T={T} rep={rep}", flush=True)


def pull_runpod_results(ablation: str) -> None:
    Path(f"{LOCAL_DIR}/results/scaling/{ablation}").mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["scp", "-P", "13104", "-i", "/home/dan/.ssh/id_ed25519-runpod",
         f"{SCP_HOST}:{REMOTE_DIR}/results/scaling/{ablation}/*",
         f"{LOCAL_DIR}/results/scaling/{ablation}/"],
        capture_output=True, text=True,
    )
    print(f"  pulled RunPod results for {ablation}", flush=True)


def run_rep(ablation: str, rep: int, t_list: list[int], metaepochs: int) -> None:
    pending = sorted(t_list, reverse=True)   # longest-first
    running: dict[str, tuple[dict, int]] = {}   # worker_name -> (worker, T)

    # Re-attach: skip .done, re-attach in-flight (survives a scheduler restart).
    for T in list(pending):
        for w in WORKERS:
            if is_done(w, T, rep, ablation):
                print(f"  rep{rep} T={T} already done on {w['name']} — skip", flush=True)
                pending.remove(T)
                break
            if is_running(w, T, rep, ablation):
                print(f"  rep{rep} T={T} already running on {w['name']} — attach", flush=True)
                running[w["name"]] = (w, T)
                pending.remove(T)
                break

    while pending or running:
        for w in WORKERS:                       # give each free worker its largest eligible T
            if w["name"] in running:
                continue
            eligible = [t for t in pending if t <= w["max_t"]]
            if not eligible:
                continue
            T = max(eligible)
            pending.remove(T)
            launch(w, T, rep, ablation, metaepochs)
            running[w["name"]] = (w, T)
        time.sleep(POLL_SECS)
        for name, (w, T) in list(running.items()):
            if is_done(w, T, rep, ablation):
                print(f"  [{name}] DONE T={T} rep={rep}", flush=True)
                del running[name]
            elif not is_running(w, T, rep, ablation):
                print(f"  [{name}] job T={T} rep={rep} died w/o .done — relaunching", flush=True)
                launch(w, T, rep, ablation, metaepochs)   # run_scaling resumes from checkpoint
    print(f"=== rep {rep} BARRIER reached (all T done) ===", flush=True)
    pull_runpod_results(ablation)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ablation", choices=["fixed3", "varying"], default="fixed3")
    p.add_argument("--reps", type=lambda s: [int(x) for x in s.split(",")], default=[0, 1, 2])
    p.add_argument("--t-list", type=lambda s: [int(x) for x in s.split(",")],
                   default=[1, 2, 4, 8])
    p.add_argument("--metaepochs", type=int, default=1200)
    args = p.parse_args()
    print(f"scheduler: ablation={args.ablation} reps={args.reps} T={args.t_list} "
          f"workers={[w['name'] for w in WORKERS]}", flush=True)
    for rep in args.reps:                        # rep-major with barrier between reps
        print(f"\n===== REP {rep} start =====", flush=True)
        run_rep(args.ablation, rep, args.t_list, args.metaepochs)
    print("\n===== ALL REPS COMPLETE =====", flush=True)


if __name__ == "__main__":
    main()
