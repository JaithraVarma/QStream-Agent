"""Collect a best-effort NVIDIA GPU snapshot for a benchmark receipt."""
from datetime import datetime, timezone
import subprocess


QUERY = [
    "nvidia-smi",
    "--query-gpu=name,driver_version,temperature.gpu,pstate,clocks.current.sm,clocks.current.memory,"
    "power.draw,memory.used,utilization.gpu",
    "--format=csv",
]


def snapshot(label):
    try:
        result = subprocess.run(QUERY, capture_output=True, text=True, timeout=15)
        return dict(label=label, utc=datetime.now(timezone.utc).isoformat(), command=QUERY,
                    returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(label=label, command=QUERY, error=str(exc))
