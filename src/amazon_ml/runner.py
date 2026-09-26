"""Run independent inference workers, then merge and strictly validate."""
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from .data import log, write_json
from .predict import merge
from .validate import validate


def finish(data, index, model, output, workers=4, threads=2, batch_size=500):
    if workers < 1 or threads < 1:
        raise ValueError("Workers and threads must be positive")
    root = Path(output)
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes = []
    status = {"state": "predicting", "started_utc": datetime.now(timezone.utc).isoformat(), "workers": workers}
    state_path = root / "run_status.json"
    with ExitStack() as stack:
        try:
            for shard in range(workers):
                logfile = logs / f"shard-{shard:03d}.log"
                stream = stack.enter_context(logfile.open("a", encoding="utf-8"))
                args = [sys.executable, "-u", "-m", "amazon_ml", "predict", "--data", str(data),
                        "--index", str(index), "--model", str(model), "--output", str(output),
                        "--shard", str(shard), "--shards", str(workers), "--threads", str(threads),
                        "--batch-size", str(batch_size)]
                process = subprocess.Popen(args, stdout=stream, stderr=subprocess.STDOUT)
                processes.append(process)
                log(f"Worker {shard}: PID {process.pid}; progress in {logfile}")
            status["pids"] = [p.pid for p in processes]
            write_json(state_path, status)
            failures = [(p.pid, code) for p in processes if (code := p.wait()) != 0]
            if failures:
                raise RuntimeError(f"Prediction workers failed: {failures}; inspect the shard logs and rerun with the same settings to resume")
            status["state"] = "merging"
            write_json(state_path, status)
            merge(root, workers)
            status["state"] = "validating"
            write_json(state_path, status)
            validate(data, root)
            status["state"] = "complete"
            status["finished_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(state_path, status)
            log("Full prediction complete; both output TSVs passed strict validation")
        except BaseException as exc:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
            status["state"] = "failed"
            status["error"] = str(exc)
            write_json(state_path, status)
            raise
