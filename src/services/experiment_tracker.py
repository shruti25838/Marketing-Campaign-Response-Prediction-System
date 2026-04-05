import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


def log_experiment(
    log_path: str, config: Dict[str, Any], metrics: Dict[str, float]
) -> None:
    payload = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "config": config,
        "metrics": metrics,
    }
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
