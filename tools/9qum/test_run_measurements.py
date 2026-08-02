#!/usr/bin/env python3
"""tools/9qum/run_measurements.py, tested WITHOUT any network access.

Three ad-hoc `until ! pgrep -f "<pattern>"; do sleep; done` wait loops cost three lost
measurements: a waiter matched its own command line and waited forever, a zombie
waiter from an earlier attempt matched the same pattern and blocked a later run, and a
driver's stdout (piped to a process that died with its parent) lost a finished
50-minute run's output. run_measurements.py replaces all of that with one sequential,
single-process runner -- this test exercises its job accounting (both direct, via
run_jobs(), and end to end via the CLI) and its lock.

Every job here is a trivial `python3.12 -c ...` (or `sleep`, to hold a lock briefly) --
no network access, negligible CPU (a live 24-game benchmark is running concurrently and
must not be starved).

Run: python3.12 tools/9qum/test_run_measurements.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(__file__))
import run_measurements as rm

PY = sys.executable
SCRIPT = os.path.join(os.path.dirname(__file__), "run_measurements.py")

TMP_DIR = None


def _run_dir(name):
    return os.path.join(TMP_DIR, name)


# --------------------------------------------------------------------------
# run_jobs(): direct calls -- the core accounting (defect: a lost measurement plus a
# hung waiter, replaced with sequential execution + a manifest)
# --------------------------------------------------------------------------
def test_two_jobs_both_logged_manifest_records_exit_codes_failure_does_not_stop_next():
    run_dir = _run_dir("two_jobs")
    jobs = [
        {"name": "ok_job", "cmd": [PY, "-c", "print('hello from ok_job')"]},
        {"name": "fail_job", "cmd": [PY, "-c", "import sys; print('about to fail'); sys.exit(3)"]},
    ]
    manifest = rm.run_jobs(jobs, run_dir)

    # both logs exist with the expected content
    ok_log = os.path.join(run_dir, "ok_job.log")
    fail_log = os.path.join(run_dir, "fail_job.log")
    assert os.path.exists(ok_log), "ok_job's log must exist"
    assert os.path.exists(fail_log), "fail_job's log must exist"
    with open(ok_log, encoding="utf-8") as f:
        assert "hello from ok_job" in f.read()
    with open(fail_log, encoding="utf-8") as f:
        assert "about to fail" in f.read()

    # the manifest records both exit codes
    by_name = {j["name"]: j for j in manifest["jobs"]}
    assert by_name["ok_job"]["exit_code"] == 0
    assert by_name["ok_job"]["completed"] is True
    assert by_name["fail_job"]["exit_code"] == 3
    assert by_name["fail_job"]["completed"] is True, \
        "a non-zero exit still means the job RAN TO COMPLETION, not that it never finished"

    # the failing job (which ran first) did not stop the second from running
    assert list(by_name) == ["ok_job", "fail_job"], "both jobs must appear, in order"
    for j in manifest["jobs"]:
        assert j["start"] is not None and j["end"] is not None and j["end"] >= j["start"]
        assert j["log"] == os.path.join(run_dir, f"{j['name']}.log")

    # manifest.json on disk must agree with the returned dict
    with open(os.path.join(run_dir, "manifest.json"), encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk == manifest
    assert on_disk["ended"] is not None, "a completed run's manifest must record an end time"


def test_failing_job_runs_first_and_second_job_still_runs():
    """Same idea, but the FAILING job is first -- the defect this guards against is a
    driver dying/stopping partway; make sure a failure never prevents later jobs."""
    run_dir = _run_dir("fail_first")
    jobs = [
        {"name": "fail_first", "cmd": [PY, "-c", "import sys; sys.exit(1)"]},
        {"name": "runs_after", "cmd": [PY, "-c", "print('still ran')"]},
    ]
    manifest = rm.run_jobs(jobs, run_dir)
    by_name = {j["name"]: j for j in manifest["jobs"]}
    assert by_name["fail_first"]["exit_code"] == 1
    assert by_name["runs_after"]["exit_code"] == 0
    with open(os.path.join(run_dir, "runs_after.log"), encoding="utf-8") as f:
        assert "still ran" in f.read()


def test_job_whose_command_does_not_exist_is_recorded_and_does_not_stop_the_run():
    run_dir = _run_dir("bad_command")
    jobs = [
        {"name": "no_such_binary", "cmd": ["/no/such/binary/exists", "--flag"]},
        {"name": "runs_after", "cmd": [PY, "-c", "print('ran anyway')"]},
    ]
    manifest = rm.run_jobs(jobs, run_dir)
    by_name = {j["name"]: j for j in manifest["jobs"]}
    assert by_name["no_such_binary"]["completed"] is False
    assert by_name["no_such_binary"]["exit_code"] is None
    assert by_name["no_such_binary"]["error"], "must record why it didn't run"
    assert by_name["runs_after"]["completed"] is True and by_name["runs_after"]["exit_code"] == 0


def test_dry_run_prints_the_plan_and_executes_nothing():
    run_dir = _run_dir("dry_run")
    jobs = [{"name": "would_run", "cmd": [PY, "-c", "print('should not appear')"]}]
    manifest = rm.run_jobs(jobs, run_dir, dry_run=True)
    assert manifest is None
    assert not os.path.exists(os.path.join(run_dir, "would_run.log")), \
        "a dry run must not execute the job or create its log"
    assert not os.path.exists(os.path.join(run_dir, "manifest.json")), \
        "a dry run must not write a manifest either"


# --------------------------------------------------------------------------
# acquire_lock/release_lock: live vs stale lock detection
# --------------------------------------------------------------------------
def test_acquire_lock_refuses_when_held_by_a_live_pid():
    run_dir = _run_dir("lock_live")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "lock.pid"), "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))  # our own pid: guaranteed alive for the test's duration
    try:
        rm.acquire_lock(run_dir)
    except rm.LockHeldError as exc:
        assert str(os.getpid()) in str(exc)
    else:
        raise AssertionError("expected LockHeldError when the lock names a live pid")


def test_acquire_lock_takes_over_a_stale_lock():
    run_dir = _run_dir("lock_stale")
    os.makedirs(run_dir, exist_ok=True)
    # A real pid that is guaranteed dead: spawn a short-lived process and wait() it, so
    # the OS is free to (and eventually will) recycle the number -- os.kill(pid, 0)
    # reports it as gone immediately after wait() returns.
    proc = subprocess.Popen([PY, "-c", "pass"])
    proc.wait()
    stale_pid = proc.pid
    with open(os.path.join(run_dir, "lock.pid"), "w", encoding="utf-8") as f:
        f.write(str(stale_pid))
    lock_path = rm.acquire_lock(run_dir)  # must NOT raise -- must take over and say so
    with open(lock_path, encoding="utf-8") as f:
        assert f.read().strip() == str(os.getpid()), "taking over the lock must stamp our own pid"


def test_release_lock_removes_the_file_and_is_idempotent():
    run_dir = _run_dir("lock_release")
    os.makedirs(run_dir, exist_ok=True)
    lock_path = rm.acquire_lock(run_dir)
    assert os.path.exists(lock_path)
    rm.release_lock(lock_path)
    assert not os.path.exists(lock_path)
    rm.release_lock(lock_path)  # must not raise on a already-removed lock


# --------------------------------------------------------------------------
# End to end via the CLI: a second, truly concurrent runner must refuse to start.
# --------------------------------------------------------------------------
def test_second_concurrent_runner_refuses_to_start_while_lock_is_held():
    run_dir = _run_dir("concurrent")
    os.makedirs(run_dir, exist_ok=True)
    jobs_path = os.path.join(run_dir, "jobs.json")
    # A job that holds the lock for a little while (negligible CPU: it's a sleep).
    slow_jobs = [{"name": "slow", "cmd": [PY, "-c", "import time; time.sleep(2.5)"]}]
    with open(jobs_path, "w", encoding="utf-8") as f:
        json.dump(slow_jobs, f)

    first = subprocess.Popen([PY, SCRIPT, "--jobs", jobs_path, "--run-dir", run_dir],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        # give the first runner time to acquire the lock and start its (sleeping) job
        deadline = time.time() + 5
        while not os.path.exists(os.path.join(run_dir, "lock.pid")) and time.time() < deadline:
            time.sleep(0.05)
        assert os.path.exists(os.path.join(run_dir, "lock.pid")), "first runner never took the lock"

        second_jobs_path = os.path.join(run_dir, "jobs2.json")
        with open(second_jobs_path, "w", encoding="utf-8") as f:
            json.dump([{"name": "should_not_run", "cmd": [PY, "-c", "print('should not run')"]}], f)
        second = subprocess.run(
            [PY, SCRIPT, "--jobs", second_jobs_path, "--run-dir", run_dir],
            capture_output=True, text=True, timeout=10,
        )
        assert second.returncode != 0, "a second runner must refuse to start while the lock is live"
        with open(os.path.join(run_dir, "lock.pid"), encoding="utf-8") as f:
            first_pid = f.read().strip()
        assert first_pid in second.stderr, \
            f"must say WHICH pid holds the lock; stderr was: {second.stderr!r}"
        assert not os.path.exists(os.path.join(run_dir, "should_not_run.log")), \
            "the second runner's job must never have executed"
    finally:
        first.wait(timeout=15)

    assert first.returncode == 0, "the first runner's own (slow) job should complete successfully"
    with open(os.path.join(run_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["jobs"][0]["name"] == "slow" and manifest["jobs"][0]["exit_code"] == 0
    assert not os.path.exists(os.path.join(run_dir, "lock.pid")), \
        "the lock must be released once the first runner finishes"


def test_cli_reports_nonzero_exit_when_a_job_fails():
    run_dir = _run_dir("cli_failure")
    jobs_path = os.path.join(run_dir, "jobs.json")
    os.makedirs(run_dir, exist_ok=True)
    with open(jobs_path, "w", encoding="utf-8") as f:
        json.dump([{"name": "fails", "cmd": [PY, "-c", "import sys; sys.exit(2)"]}], f)
    result = subprocess.run([PY, SCRIPT, "--jobs", jobs_path, "--run-dir", run_dir],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode != 0, "the CLI must exit non-zero when a job failed"
    assert "fails" in result.stderr


def test_cli_dry_run_writes_nothing():
    run_dir = _run_dir("cli_dry_run")
    jobs_path = os.path.join(run_dir, "jobs.json")
    os.makedirs(run_dir, exist_ok=True)
    with open(jobs_path, "w", encoding="utf-8") as f:
        json.dump([{"name": "would_run", "cmd": [PY, "-c", "print(1)"]}], f)
    result = subprocess.run([PY, SCRIPT, "--jobs", jobs_path, "--run-dir", run_dir, "--dry-run"],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0
    assert "would_run" in result.stdout
    assert not os.path.exists(os.path.join(run_dir, "manifest.json"))
    assert not os.path.exists(os.path.join(run_dir, "lock.pid"))


TESTS = [
    test_two_jobs_both_logged_manifest_records_exit_codes_failure_does_not_stop_next,
    test_failing_job_runs_first_and_second_job_still_runs,
    test_job_whose_command_does_not_exist_is_recorded_and_does_not_stop_the_run,
    test_dry_run_prints_the_plan_and_executes_nothing,
    test_acquire_lock_refuses_when_held_by_a_live_pid,
    test_acquire_lock_takes_over_a_stale_lock,
    test_release_lock_removes_the_file_and_is_idempotent,
    test_second_concurrent_runner_refuses_to_start_while_lock_is_held,
    test_cli_reports_nonzero_exit_when_a_job_fails,
    test_cli_dry_run_writes_nothing,
]


if __name__ == "__main__":
    TMP_DIR = tempfile.mkdtemp(prefix="run_measurements_test_")
    try:
        for t in TESTS:
            t()
        print(f"OK: run_measurements.py sequential runner + lock + manifest ({len(TESTS)}/{len(TESTS)})")
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)
