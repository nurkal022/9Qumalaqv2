#!/usr/bin/env python3
"""Sequential measurement job runner -- replaces the ad-hoc `pgrep -f "<pattern>"` wait
loops that cost three lost measurements in one night:

  1. a waiter's own command line matched its own wait pattern, so it waited forever;
  2. a zombie waiter from an earlier, already-abandoned attempt ALSO matched the
     pattern, so it blocked a later, legitimate run that was waiting on the same
     pattern to disappear;
  3. a driver's stdout was piped to a process that died with its parent, silently
     losing a finished 50-minute run's output.

The fix is not a better wait loop -- it's to make waiting unnecessary. This runner
executes a list of jobs SEQUENTIALLY IN A SINGLE PROCESS: job 2 only ever starts after
job 1's subprocess has actually exited (subprocess.run() blocks), so there is never a
second process out there to (mis-)identify by matching a text pattern, and never a
"has it finished yet" question to poll for.

Each job's stdout+stderr go to their OWN log file (opened, written, flushed and closed
for that job alone, not a pipe shared with a parent that might die) so a log always
survives the runner regardless of what happens afterwards. A lock file in the run
directory prevents two runners from clobbering the same run directory concurrently; a
manifest.json records enough about every job (command, start/end time, exit code, log
path, whether it completed) that a reader can always tell a finished run from an
abandoned one without re-deriving it from process state.

Usage:
  python3.12 tools/9qum/run_measurements.py --jobs jobs.json --run-dir runs/2026-08-02
  python3.12 tools/9qum/run_measurements.py --jobs jobs.json --run-dir runs/x --dry-run

jobs.json is a JSON list of objects, each with:
  {"name": "gate_a", "cmd": ["python3.12", "tools/9qum/match.py", "--games", "24"]}
`cmd` is always list-form (argv), never a shell string -- so there is no shell
quoting/injection surface and no shell process for a pattern-matching waiter to
mis-identify in the first place.

If a job fails (non-zero exit, or the command itself couldn't even be started, e.g. a
typo'd path), the runner records it and moves on to the next job rather than dying --
later jobs in a measurement run are usually independent of an earlier one's success.
"""
import argparse
import json
import os
import subprocess
import sys
import time


class LockHeldError(RuntimeError):
    """Another runner already holds the run directory's lock and is still alive."""


def _pid_alive(pid: int) -> bool:
    """True if `pid` names a live process we can see (not necessarily one we own)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just owned by someone else
    return True


def acquire_lock(run_dir: str) -> str:
    """Create (or take over) `run_dir`'s lock file; return its path.

    The lock is a plain pid file checked against /proc (via os.kill(pid, 0)), not an
    flock -- deliberately simple and inspectable with `cat`, unlike the pgrep-based
    waiting this replaces. If the file names a still-live pid, refuse to start and say
    which pid holds it. If the named pid is gone (the previous runner crashed without
    cleaning up), the lock is stale: say so, and take it over.
    """
    os.makedirs(run_dir, exist_ok=True)
    lock_path = os.path.join(run_dir, "lock.pid")
    if os.path.exists(lock_path):
        with open(lock_path, encoding="utf-8") as f:
            raw = f.read().strip()
        held_pid = int(raw) if raw.isdigit() else None
        if held_pid is not None and _pid_alive(held_pid):
            raise LockHeldError(
                f"refusing to start: lock at {lock_path} is held by live pid {held_pid}"
            )
        print(f"lock at {lock_path} is stale (pid {raw!r} is not running) -- taking it over",
              flush=True)
    with open(lock_path, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    return lock_path


def release_lock(lock_path: str) -> None:
    try:
        os.remove(lock_path)
    except FileNotFoundError:
        pass


def _write_manifest(manifest_path: str, manifest: dict) -> None:
    tmp_path = manifest_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, manifest_path)


def run_jobs(jobs, run_dir, dry_run=False):
    """Run `jobs` (list of {"name", "cmd", ["cwd"], ["env"]}) sequentially in this
    process. Returns the manifest dict (None for a dry run). Raises LockHeldError if
    another live runner already holds `run_dir`'s lock; never raises for an individual
    job failure -- those are recorded in the manifest and execution continues.
    """
    if dry_run:
        print(f"[dry-run] would run {len(jobs)} job(s) in {run_dir}:")
        for j in jobs:
            print(f"  {j['name']}: {' '.join(j['cmd'])}")
        return None

    os.makedirs(run_dir, exist_ok=True)
    lock_path = acquire_lock(run_dir)
    manifest_path = os.path.join(run_dir, "manifest.json")
    manifest = {
        "run_dir": os.path.abspath(run_dir),
        "runner_pid": os.getpid(),
        "started": time.time(),
        "ended": None,
        "jobs": [],
    }
    _write_manifest(manifest_path, manifest)
    try:
        for j in jobs:
            name = j["name"]
            cmd = j["cmd"]
            log_path = os.path.join(run_dir, f"{name}.log")
            entry = {
                "name": name, "cmd": cmd, "log": log_path,
                "start": time.time(), "end": None,
                "exit_code": None, "completed": False, "error": None,
            }
            manifest["jobs"].append(entry)
            _write_manifest(manifest_path, manifest)

            print(f"[{name}] starting: {' '.join(cmd)}  (log: {log_path})", flush=True)
            with open(log_path, "w", encoding="utf-8") as logf:
                try:
                    proc = subprocess.run(
                        cmd, stdout=logf, stderr=subprocess.STDOUT,
                        cwd=j.get("cwd"), env=j.get("env"),
                    )
                    entry["exit_code"] = proc.returncode
                    entry["completed"] = True
                except OSError as exc:
                    # e.g. the command doesn't exist -- record it and move on, don't
                    # let one bad job kill every job after it.
                    entry["error"] = str(exc)
                    entry["completed"] = False
                finally:
                    logf.flush()
                    os.fsync(logf.fileno())
            entry["end"] = time.time()
            _write_manifest(manifest_path, manifest)

            if entry["completed"] and entry["exit_code"] == 0:
                print(f"[{name}] ok", flush=True)
            elif entry["completed"]:
                print(f"[{name}] FAILED (exit {entry['exit_code']}) -- continuing with remaining jobs",
                      flush=True)
            else:
                print(f"[{name}] FAILED TO START ({entry['error']}) -- continuing with remaining jobs",
                      flush=True)

        manifest["ended"] = time.time()
        _write_manifest(manifest_path, manifest)
    finally:
        release_lock(lock_path)
    return manifest


def load_jobs(jobs_path):
    with open(jobs_path, encoding="utf-8") as f:
        jobs = json.load(f)
    for j in jobs:
        if "name" not in j or "cmd" not in j:
            raise ValueError(f"each job needs 'name' and 'cmd': {j!r}")
        if not isinstance(j["cmd"], list):
            raise ValueError(f"job {j['name']!r}: 'cmd' must be a list (argv form), not a shell string")
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", required=True, help="path to a jobs JSON file (list of {name, cmd})")
    ap.add_argument("--run-dir", required=True, help="directory for this run's lock, logs and manifest")
    ap.add_argument("--dry-run", action="store_true", help="print the plan without executing")
    a = ap.parse_args()

    jobs = load_jobs(a.jobs)
    try:
        manifest = run_jobs(jobs, a.run_dir, dry_run=a.dry_run)
    except LockHeldError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    if manifest is None:  # dry run
        return
    failed = [j for j in manifest["jobs"] if not (j["completed"] and j["exit_code"] == 0)]
    if failed:
        names = ", ".join(j["name"] for j in failed)
        print(f"\n{len(failed)}/{len(manifest['jobs'])} job(s) failed: {names}", file=sys.stderr)
        sys.exit(1)
    print(f"\nall {len(manifest['jobs'])} job(s) completed successfully")


if __name__ == "__main__":
    main()
