from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from packages.marl.evaluation import compare, evaluate
from packages.marl.training import train


def main():
    qmix = train("quick", "qmix", 42)
    idqn = train("quick", "idqn", 42)
    qmix_eval = evaluate(qmix["checkpoint"], [101, 102, 103])
    idqn_eval = evaluate(idqn["checkpoint"], [101, 102, 103])
    result = compare(qmix_eval, idqn_eval)
    path = Path("reports/model-comparison.json")
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

