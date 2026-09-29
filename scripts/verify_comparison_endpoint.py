from __future__ import annotations

import time

from fastapi.testclient import TestClient

from apps.api.main import app


def main() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/models/comparison-runs?preset=quick&seed=42",
            headers={"Authorization": "Bearer local-development-token"},
        )
        response.raise_for_status()
        comparison_id = response.json()["id"]
        result = response.json()
        for _ in range(300):
            result = client.get(f"/models/comparison-runs/{comparison_id}").json()
            if result["status"] != "RUNNING":
                break
            time.sleep(0.1)
        print(
            {
                "status": result["status"],
                "stage": result["stage"],
                "has_result": bool(result.get("result")),
                "report": result.get("report"),
            }
        )
        if result["status"] != "COMPLETED":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
