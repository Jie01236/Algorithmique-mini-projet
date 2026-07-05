from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.model import train_model


if __name__ == "__main__":
    metrics = train_model()
    print(json.dumps(metrics, indent=2))
