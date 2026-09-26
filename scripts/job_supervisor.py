"""Detached, finite local pipeline recovery. No Codex/API access is required."""
import argparse
import ctypes
from datetime import datetime, timezone
import json
import msvcrt
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/"artifacts"
CPU = ROOT/".venv/Scripts/python.exe"
GPU = ROOT/".venv_gpu/Scripts/python.exe"


def read(path):
    try:
        return json.loads((ROOT/path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write(path, value):
    path = ROOT/path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def log(message):
    print(datetime.now().astimezone().isoformat(timespec="seconds"), message, flush=True)


def in_job():
    result = ctypes.c_int()
    kernel = ctypes.windll.kernel32
    kernel.IsProcessInJob(kernel.GetCurrentProcess(), None, ctypes.byref(result))
    return bool(result.value)


def active(module, extra=()):
    found = []
    for proc in psutil.process_iter(["pid", "cmdline", "name"]):
        try:
            command = proc.info["cmdline"] or []
            if "python" not in (proc.info["name"] or "").lower() or "-m" not in command:
                continue
            if command[command.index("-m")+1] != module or not all(x in command for x in extra):
                continue
            if Path(proc.cwd()).resolve() == ROOT:
                found.append(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, IndexError):
            continue
    return found


def sqlite_done(path):
    if not (ROOT/path).exists():
        return False
    try:
        db = sqlite3.connect(f"file:{(ROOT/path).as_posix()}?mode=ro", uri=True, timeout=1)
        value = db.execute("SELECT value FROM meta WHERE key='complete'").fetchone()
        db.close()
        return bool(value and value[0] == "yes")
    except sqlite3.OperationalError:
        return False


def selected_model():
    tree = read("artifacts/experiment_v3/experiment_report.json")
    choices = [(tree["selected"]["tune_macro_f0_5"], "artifacts/experiment_v3/model", None)]
    neural = read("artifacts/neural_v1/report.json")
    if read("artifacts/neural_v1/status.json").get("state") == "complete" and neural.get("metrics"):
        choices.append((neural["selected"]["tune_macro_f0_5"], "artifacts/experiment_v3/model", "artifacts/neural_v1"))
    large = read("artifacts/large_v1/report.json")
    if read("artifacts/large_v1/status.json").get("state") == "complete" and large.get("metrics"):
        choices.append((large["selected"]["tune_macro_f0_5"], "artifacts/large_v1/model", None))
    # Stable ties retain the simpler, already confirmed model.
    return max(choices, key=lambda item: item[0])


def launch_child(command, logfile):
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    with (ART/logfile).open("a", encoding="utf-8") as stream:
        return subprocess.Popen([str(x) for x in command], cwd=ROOT, stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=subprocess.STDOUT, close_fds=True,
                                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                                startupinfo=startup).pid


def run():
    ART.mkdir(exist_ok=True)
    lock = (ART/"supervisor.lock").open("a+b")
    lock.seek(0)
    if not lock.read(1):
        lock.write(b"1"); lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        log("Another recovery supervisor is already active; exiting")
        return
    attempts = read("artifacts/supervisor_attempts.json")
    last_launch = {}
    # Temporary system-awake request; does not change the user's power plan.
    awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    log(f"Recovery supervisor PID {psutil.Process().pid}; in Windows job={in_job()}; keep-awake={awake}")
    try:
        while not (ART/"STOP_SUPERVISOR").exists():
            statuses = {}
            def ensure(name, done, module, extra, command, also_running=(), prerequisites=True):
                running = active(module, extra)
                for other_module, other_extra in also_running:
                    running.extend(active(other_module, other_extra))
                if done:
                    statuses[name] = {"state": "complete"}
                elif running:
                    statuses[name] = {"state": "running", "pids": sorted(set(running))}
                elif not prerequisites:
                    statuses[name] = {"state": "waiting_for_dependencies"}
                elif attempts.get(name, 0) >= 3:
                    statuses[name] = {"state": "needs_attention", "reason": "Three launches failed; inspect recovery log"}
                elif time.monotonic()-last_launch.get(name, -1000) < 90:
                    statuses[name] = {"state": "restart_backoff"}
                else:
                    # Preserve the earlier evaluated neural model before a restart.
                    # The current neural run has epoch checkpoints but not an
                    # optimizer resume; its restart repeats training, not data prep.
                    if name == "neural" and (ART/"neural_v1/best.pt").exists():
                        backup = ART/"neural_v1"/f"recovery_backup_{int(time.time())}"
                        backup.mkdir()
                        for filename in ("best.pt", "report.json", "status.json"):
                            source = ART/"neural_v1"/filename
                            if source.exists():
                                shutil.copy2(source, backup/filename)
                    pid = launch_child(command, f"recovery_{name}.log")
                    attempts[name] = attempts.get(name, 0)+1
                    last_launch[name] = time.monotonic()
                    write("artifacts/supervisor_attempts.json", attempts)
                    statuses[name] = {"state": "launched", "pid": pid, "attempt": attempts[name]}
                    log(f"Launched {name}, PID {pid}, attempt {attempts[name]}")

            baseline_done = read("output/run_status.json").get("state") == "complete" and read("output/validation.json").get("status") == "PASS"
            ensure("baseline", baseline_done, "amazon_ml", ("finish", "--workers", "2"),
                   [CPU, "-X", "utf8", "-u", "-m", "amazon_ml", "finish", "--workers", "2", "--threads", "2"],
                   also_running=[("amazon_ml", ("predict", "--output", "output"))])
            ensure("train_index", sqlite_done(Path("artifacts/train.sqlite")), "amazon_ml", ("build-index", "--split", "train"),
                   [CPU, "-X", "utf8", "-u", "-m", "amazon_ml", "build-index", "--split", "train", "--index", "artifacts/train.sqlite"])
            raw_done = sqlite_done(Path("artifacts/test_raw.sqlite"))
            ensure("raw_index", raw_done, "amazon_ml.rescore", ("--build-raw",),
                   [CPU, "-X", "utf8", "-u", "-m", "amazon_ml.rescore", "--build-raw"])
            neural_done = read("artifacts/neural_v1/status.json").get("state") == "complete"
            ensure("neural", neural_done, "amazon_ml.neural", ("--output", "artifacts/neural_v1"),
                   [GPU, "-X", "utf8", "-u", "-m", "amazon_ml.neural", "--output", "artifacts/neural_v1", "--batch-size", "32", "--accumulation", "1"])
            large_done = read("artifacts/large_v1/status.json").get("state") == "complete"
            ensure("large", large_done, "amazon_ml.large", (),
                   [GPU, "-X", "utf8", "-u", "-m", "amazon_ml.large", "--wait-for-neural"])
            dependencies = baseline_done and raw_done and neural_done and large_done
            improved_done = read("output_improved/run_status.json").get("state") == "complete" and read("output_improved/validation.json").get("status") == "PASS"
            selected = None
            if dependencies:
                score, tree_path, neural_path = selected_model()
                selected = {"tune_f0_5": score, "tree_model": tree_path, "neural_model": neural_path}
                write("artifacts/final_model_selection.json", selected)
                command = [GPU, "-X", "utf8", "-u", "-m", "amazon_ml.rescore", "--tree-model", tree_path, "--workers", "4", "--output", "output_improved"]
                if neural_path:
                    command += ["--neural-model", neural_path]
            else:
                command = []
            ensure("improved_prediction", improved_done, "amazon_ml.rescore", ("--output", "output_improved"), command, prerequisites=dependencies)
            finished = all(v["state"] == "complete" for v in statuses.values())
            state = {"state": "complete" if finished else "running", "pid": psutil.Process().pid,
                     "updated_utc": datetime.now(timezone.utc).isoformat(), "inside_windows_job": in_job(),
                     "keep_awake_request": awake, "jobs": statuses, "selected_model": selected,
                     "note": "Local process; no Codex usage, cloud resources or automatic upload. PC must remain powered on."}
            write("artifacts/supervisor_status.json", state)
            if finished:
                log("All training and prediction jobs complete; both submissions validated")
                return
            time.sleep(20)
        write("artifacts/supervisor_status.json", {"state": "stopped_by_marker", "jobs_left_running": True})
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.close()


def detached_launch():
    ART.mkdir(exist_ok=True)
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    command = [sys.executable, str(Path(__file__).resolve()), "--run"]
    with (ART/"supervisor.log").open("a", encoding="utf-8") as stream:
        try:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                       creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB, startupinfo=startup, close_fds=True)
            mode = "detached with job breakaway"
        except OSError:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                       creationflags=flags, startupinfo=startup, close_fds=True)
            mode = "detached; verify Windows job membership in status"
    print(json.dumps({"pid": process.pid, "launch_mode": mode}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    run() if args.run else detached_launch()
