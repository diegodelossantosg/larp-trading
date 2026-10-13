"""Print live.json: the newest status and chart points, small enough to publish every minute.

    python -m trader.live > /tmp/live.json
"""

from __future__ import annotations

import json

from . import journal


def build(points: int = 15) -> dict:
    return {
        "status": journal.read("status.json", {}),
        "intraday_tail": (journal.read("intraday.json", {}) or {}).get("points", [])[-points:],
    }


if __name__ == "__main__":
    print(json.dumps(build(), separators=(",", ":")))
