from fastapi.testclient import TestClient
import time

from apps.api.main import app, create_automatic_incident
from apps.api.state import state


AUTH = {"Authorization":"Bearer local-development-token"}


def test_mission_validation_and_authentication():
    with TestClient(app) as client:
        request = {"description":"medical","pickup":[1,1],"destination":[17,16],"package_weight":1.5,"deadline_steps":100,"priority":"CRITICAL"}
        validation = client.post("/missions/validate",json=request)
        assert validation.status_code == 200 and validation.json()["valid"]
        assert client.post("/missions",json=request).status_code == 401
        created = client.post("/missions",json=request,headers=AUTH)
        assert created.status_code == 200


def test_websocket_receives_real_simulation_events():
    with TestClient(app) as client:
        with client.websocket_connect("/ws/live") as socket:
            run = client.post("/simulations?seed=5&pace=0.001",headers=AUTH).json()
            types = {socket.receive_json()["type"] for _ in range(8)}
        assert "fleet" in types
        assert "decision" in types
        client.post(f"/simulations/{run['id']}/stop",headers=AUTH)


def test_metrics_and_database_readiness():
    with TestClient(app) as client:
        assert client.get("/ready").json()["status"] == "ready"
        assert "api_request_duration_seconds" in client.get("/metrics").text


def test_complete_simulation_generates_replay_and_report():
    with TestClient(app) as client:
        run = client.post("/simulations?seed=42&pace=0.0001", headers=AUTH).json()
        for _ in range(200):
            current = client.get(f"/simulations/{run['id']}").json()
            if current["status"] in {"COMPLETED", "ENDED", "FAILED"}:
                break
            time.sleep(0.01)
        assert current["status"] == "COMPLETED"
        report = client.get(f"/reports/{run['id']}").json()
        assert report["packages_delivered"] == report["packages_total"] == 3
        assert client.get("/events/decision").json()
        assert client.get("/events/communication").json()


def test_entered_destination_controls_actual_drone_target():
    with TestClient(app) as client:
        request = {
            "description": "custom target", "pickup": [1, 1], "destination": [5, 14],
            "package_weight": 1.0, "deadline_steps": 100, "priority": "NORMAL",
        }
        mission = client.post("/missions", json=request, headers=AUTH).json()
        run = client.post(
            f"/simulations?mission_id={mission['id']}&seed=42&pace=0.0001", headers=AUTH
        ).json()
        for _ in range(200):
            current = client.get(f"/simulations/{run['id']}").json()
            if current["status"] in {"COMPLETED", "ENDED", "FAILED"}:
                break
            time.sleep(0.01)
        report = client.get(f"/reports/{run['id']}").json()
        final_position = report["final_drone_states"]["D1"]["position"]
        assert report["packages_delivered"] == report["packages_total"] == 1
        assert abs(final_position[0] - 5) <= 1.1
        assert abs(final_position[1] - 14) <= 1.1


def test_boundary_destination_requires_drone_to_reach_coordinate():
    with TestClient(app) as client:
        request = {
            "description": "boundary target", "pickup": [1, 1], "destination": [20, 10],
            "package_weight": 1.5, "deadline_steps": 300, "priority": "NORMAL",
        }
        mission = client.post("/missions", json=request, headers=AUTH).json()
        run = client.post(
            f"/simulations?mission_id={mission['id']}&seed=42&pace=0.0001", headers=AUTH
        ).json()
        for _ in range(250):
            current = client.get(f"/simulations/{run['id']}").json()
            if current["status"] in {"COMPLETED", "ENDED", "FAILED"}:
                break
            time.sleep(0.01)
        report = client.get(f"/reports/{run['id']}").json()
        assert report["final_drone_states"]["D1"]["position"][:2] == [20.0, 10.0]


async def test_serious_safety_event_creates_automatic_incident():
    state.incidents.clear()
    event = {
        "step": 4, "drone_id": "D2", "original_action": "MOVE_WEST",
        "replacement_action": "CLIMB", "violated_rules": ["COMMUNICATION_CONFLICT"],
        "reason": "Yielding to D1", "rule_version": "1.0.0",
    }
    incident = await create_automatic_incident("simulation-test", event, 42)
    assert incident is not None
    assert incident["title"].startswith("Automatic safety incident")
    assert incident["evidence"]["measured_safety_event"] == event

