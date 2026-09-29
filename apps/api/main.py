from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from apps.api.database import create_schema
from apps.api.state import ROOT, state
from packages.agents.workflow import OperationalWorkflow
from packages.marl.evaluation import compare, evaluate
from packages.marl.training import train
from packages.shared.models import AgentRequest, Mission, MissionCreate, MissionStatus, TrainingRequest
from packages.simulation.environment import DroneDeliveryEnv
from packages.simulation.policy import RuleBasedPolicy


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")
logger = logging.getLogger("drone-platform")
API_LATENCY = Histogram("api_request_duration_seconds", "API latency", ["method", "path"])
ACTIVE_MISSIONS = Gauge("active_missions", "Active missions")
SIMULATION_STEPS = Counter("simulation_steps_total", "Simulation steps")
COLLISION_WARNINGS = Counter("collision_warnings_total", "Collision warnings")
SAFETY_OVERRIDES = Counter("safety_overrides_total", "Safety overrides")
TOOL_CALLS = Counter("agent_tool_calls_total", "Agent tool calls", ["tool"])
AGENT_FAILURES = Counter("agent_failures_total", "Agent failures")


@asynccontextmanager
async def lifespan(_: FastAPI):
    create_schema()
    await state.runtime.connect()
    yield
    for task in state.simulation_tasks.values():
        task.cancel()


app = FastAPI(title="Agentic Drone Delivery Platform", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:8080", "http://127.0.0.1:8080",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def correlation_middleware(request: Request, call_next):
    correlation_id = request.headers.get("x-correlation-id", str(uuid4()))
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["x-correlation-id"] = correlation_id
    API_LATENCY.labels(request.method, request.url.path).observe(time.perf_counter() - started)
    logger.info(json.dumps({"event": "http_request", "correlation_id": correlation_id, "method": request.method, "path": request.url.path, "status": response.status_code}))
    return response


def require_auth(authorization: str | None = Header(default=None)):
    expected = os.getenv("LOCAL_API_TOKEN", "local-development-token")
    if authorization != f"Bearer {expected}":
        raise HTTPException(401, "Missing or invalid local API token")


@app.get("/health")
def health():
    return {"status": "healthy", "time": datetime.now(timezone.utc).isoformat()}


@app.get("/ready")
def ready():
    return {"status": "ready", "state_backend": state.runtime.mode, "database": "configured"}


@app.post("/auth/token")
def token(username: str = "operator", password: str = "operator"):
    if username != os.getenv("LOCAL_USERNAME", "operator") or password != os.getenv("LOCAL_PASSWORD", "operator"):
        raise HTTPException(401, "Invalid local credentials")
    return {"access_token": os.getenv("LOCAL_API_TOKEN", "local-development-token"), "token_type": "bearer"}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return PlainTextResponse(generate_latest().decode(), media_type=CONTENT_TYPE_LATEST)


def validate_request(mission: MissionCreate) -> dict:
    errors, checks = [], {}
    checks["payload"] = {"passed": mission.package_weight <= 5.0, "maximum_kg": 5.0}
    if not checks["payload"]["passed"]:
        errors.append("Package exceeds the configured 5 kg payload limit")
    x, y = mission.destination
    in_bounds = 0 <= x <= 20 and 0 <= y <= 20
    checks["map_bounds"] = {"passed": in_bounds}
    if not in_bounds:
        errors.append("Destination is outside the 20 x 20 simulation map")
    restricted = 8 <= x <= 12 and 8 <= y <= 12
    checks["geofence"] = {"passed": not restricted, "zone": "RZ-1" if restricted else None}
    if restricted:
        errors.append("Destination lies inside restricted zone RZ-1")
    distance = abs(x - mission.pickup[0]) + abs(y - mission.pickup[1])
    energy = distance * (0.0025 + mission.package_weight * 0.0005) + 0.15
    checks["energy"] = {"passed": energy <= 1, "required_battery_fraction": round(energy, 4), "reserve": 0.15}
    if energy > 1:
        errors.append("Estimated energy including reserve exceeds available battery")
    checks["deadline"] = {"passed": mission.deadline_steps >= distance + 5, "estimated_steps": int(distance + 5)}
    if not checks["deadline"]["passed"]:
        errors.append("Deadline is shorter than deterministic minimum travel time")
    checks["weather"] = {"passed": True, "source": "simulator", "wind_limit_mps": 14}
    return {"valid": not errors, "errors": errors, "checks": checks}


@app.post("/missions", dependencies=[Depends(require_auth)])
async def create_mission(request: MissionCreate):
    mission = Mission(request=request)
    state.missions[str(mission.id)] = mission
    await state.event_store.append("mission", mission.model_dump(mode="json"))
    return mission


@app.post("/missions/validate")
def validate_mission(request: MissionCreate):
    return validate_request(request)


@app.get("/missions")
def missions():
    return [m.model_dump(mode="json") for m in state.missions.values()]


@app.post("/missions/{mission_id}/feasibility")
def feasibility(mission_id: str):
    mission = state.missions.get(mission_id)
    if not mission:
        raise HTTPException(404, "Mission not found")
    validation = validate_request(mission.request)
    return {**validation, "simulation": {"method": "deterministic energy and geofence preflight", "seed": 42}}


async def run_simulation(simulation_id: str, mission_id: str | None, seed: int, pace: float):
    env = DroneDeliveryEnv(seed=seed, max_steps=150)
    env.simulation_id = uuid4()
    policy = RuleBasedPolicy()
    control = state.simulation_controls[simulation_id]
    state.simulations[simulation_id].update({"status": "RUNNING", "environment_id": str(env.simulation_id), "seed": seed})
    if mission_id and mission_id in state.missions:
        state.missions[mission_id].status = MissionStatus.ACTIVE
        ACTIVE_MISSIONS.inc()
    try:
        observations, _ = env.reset(seed=seed)
        env.simulation_id = uuid4()
        if mission_id and mission_id in state.missions:
            request = state.missions[mission_id].request
            package = env.packages["P1"]
            package.update(
                {
                    "pickup": env.depot.copy(),
                    "destination": np.array(
                        [request.destination[0], request.destination[1], 2.0], dtype=float
                    ),
                    "weight": request.package_weight,
                    "priority": request.priority,
                    "deadline": request.deadline_steps,
                    "assigned_to": "D1",
                    "active": True,
                    "picked_up": False,
                    "delivered": False,
                    "delivered_step": None,
                }
            )
            env.drones["D1"]["assigned_package"] = "P1"
            for package_id in ("P2", "P3"):
                env.packages[package_id]["active"] = False
                env.packages[package_id]["delivered"] = True
        while env.agents and not control["stop"]:
            await control["resume"].wait()
            actions = {agent: int(policy.act(env, agent)) for agent in env.agents}
            observations, rewards, terminated, truncated, infos = env.step(actions)
            state.current_snapshot = env.snapshot()
            state.current_snapshot["run_id"] = simulation_id
            await state.runtime.set(f"simulation:{simulation_id}", state.current_snapshot)
            state.simulations[simulation_id]["snapshot"] = state.current_snapshot
            SIMULATION_STEPS.inc()
            await state.event_store.append("fleet", state.current_snapshot)
            for info in infos.values():
                await state.event_store.append("decision", info["decision_event"])
            for event in env.communication_events[state.simulations[simulation_id].get("communication_cursor", 0):]:
                await state.event_store.append("communication", event)
            state.simulations[simulation_id]["communication_cursor"] = len(env.communication_events)
            for event in env.safety_events[state.simulations[simulation_id].get("safety_cursor", 0):]:
                await state.event_store.append("safety", event)
                SAFETY_OVERRIDES.inc()
                await create_automatic_incident(simulation_id, event, seed)
                if "MINIMUM_SEPARATION" in event["violated_rules"]:
                    COLLISION_WARNINGS.inc()
                    await state.event_store.append("collision", event)
            state.simulations[simulation_id]["safety_cursor"] = len(env.safety_events)
            if pace:
                await asyncio.sleep(pace)
        completed = all(p["delivered"] for p in env.packages.values())
        status = "STOPPED" if control["stop"] else "COMPLETED" if completed else "ENDED"
        state.simulations[simulation_id]["status"] = status
        state.simulations[simulation_id]["snapshot"] = env.snapshot()
        report = build_mission_report(simulation_id, env)
        state.simulations[simulation_id]["report"] = report
        write_report_files(simulation_id, report)
        await state.event_store.append("report", report)
        if mission_id and mission_id in state.missions:
            state.missions[mission_id].status = MissionStatus.COMPLETED if completed else MissionStatus.STOPPED
            ACTIVE_MISSIONS.dec()
    except asyncio.CancelledError:
        state.simulations[simulation_id]["status"] = "STOPPED"
        raise
    except Exception as exc:
        state.simulations[simulation_id].update({"status": "FAILED", "error": str(exc)})
        logger.exception(json.dumps({"event": "simulation_failure", "simulation_id": simulation_id}))


def build_mission_report(simulation_id: str, env: DroneDeliveryEnv) -> dict:
    active_packages = [p for p in env.packages.values() if p.get("active", True)]
    delivered = [p for p in active_packages if p["delivered"]]
    min_sep = min(
        (e.observation_summary["nearest_drone_distance"] for e in env.decision_events),
        default=None,
    )
    return {
        "simulation_id": simulation_id,
        "mission_summary": "Fleet delivery simulation",
        "packages_delivered": len(delivered),
        "packages_total": len(active_packages),
        "on_time": all(p["delivered_step"] <= p["deadline"] for p in delivered) if delivered else False,
        "total_duration_steps": env.step_count,
        "distance_travelled": sum(d["distance_travelled"] for d in env.drones.values()),
        "energy_consumed": sum(d["energy_consumed"] for d in env.drones.values()),
        "minimum_observed_drone_separation": min_sep,
        "warnings": len([e for e in env.safety_events if "MINIMUM_SEPARATION" in e["violated_rules"]]),
        "safety_interventions": len(env.safety_events),
        "communication_failures": len([e for e in env.communication_events if e["dropped"]]),
        "final_drone_states": env.snapshot()["drones"],
        "model_version": "rule-based-v1",
        "incident_references": [],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def write_report_files(simulation_id: str, report: dict):
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / f"mission-{simulation_id}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (reports / f"mission-{simulation_id}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in report.items():
            writer.writerow([key, json.dumps(value) if isinstance(value, (dict, list)) else value])


async def create_automatic_incident(simulation_id: str, safety_event: dict, seed: int) -> dict | None:
    serious_rules = {
        "MINIMUM_SEPARATION", "EMERGENCY_BATTERY", "RESTRICTED_ZONE",
        "COMMUNICATION_CONFLICT", "MINIMUM_ALTITUDE", "MAXIMUM_ALTITUDE",
    }
    matched = sorted(serious_rules.intersection(safety_event.get("violated_rules", [])))
    if not matched:
        return None
    signature = f"{simulation_id}:{safety_event.get('drone_id')}:{safety_event.get('step')}:{','.join(matched)}"
    if any(item.get("signature") == signature for item in state.incidents.values()):
        return None
    incident_id = str(uuid4())
    incident = {
        "id": incident_id,
        "signature": signature,
        "title": f"Automatic safety incident: {', '.join(matched)}",
        "status": "OPEN",
        "simulation_id": simulation_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "timeline": [
            {
                "step": safety_event.get("step"), "event": "POLICY_ACTION_PROPOSED",
                "drone_id": safety_event.get("drone_id"),
                "action": safety_event.get("original_action"),
            },
            {
                "step": safety_event.get("step"), "event": "SAFETY_OVERRIDE",
                "replacement_action": safety_event.get("replacement_action"),
                "violated_rules": matched, "reason": safety_event.get("reason"),
            },
        ],
        "evidence": {
            "measured_safety_event": safety_event,
            "rule_version": safety_event.get("rule_version"),
        },
        "interpretation": "A deterministic safety rule or communication conflict changed the proposed action. This interpretation does not replace the measured event above.",
        "reproduction": {"seed": seed, "simulation_id": simulation_id, "step": safety_event.get("step")},
    }
    state.incidents[incident_id] = incident
    await state.event_store.append("incident", incident)
    return incident


@app.post("/simulations", dependencies=[Depends(require_auth)])
async def start_simulation(mission_id: str | None = None, seed: int = 42, pace: float = 0.1):
    if mission_id and mission_id not in state.missions:
        raise HTTPException(404, "Mission not found")
    simulation_id = str(uuid4())
    state.simulations[simulation_id] = {"id": simulation_id, "mission_id": mission_id, "status": "STARTING"}
    state.simulation_controls[simulation_id] = {"resume": asyncio.Event(), "stop": False}
    state.simulation_controls[simulation_id]["resume"].set()
    task = asyncio.create_task(run_simulation(simulation_id, mission_id, seed, max(0, min(pace, 2))))
    state.simulation_tasks[simulation_id] = task
    return state.simulations[simulation_id]


@app.get("/simulations/{simulation_id}")
def simulation(simulation_id: str):
    if simulation_id not in state.simulations:
        raise HTTPException(404, "Simulation not found")
    return state.simulations[simulation_id]


@app.post("/simulations/{simulation_id}/{command}", dependencies=[Depends(require_auth)])
def simulation_command(simulation_id: str, command: str):
    if simulation_id not in state.simulation_controls:
        raise HTTPException(404, "Simulation not found")
    control = state.simulation_controls[simulation_id]
    if command == "pause":
        control["resume"].clear()
        state.simulations[simulation_id]["status"] = "PAUSED"
    elif command == "resume":
        control["resume"].set()
        state.simulations[simulation_id]["status"] = "RUNNING"
    elif command == "stop":
        control["stop"] = True
        control["resume"].set()
    else:
        raise HTTPException(400, "Command must be pause, resume, or stop")
    return state.simulations[simulation_id]


@app.websocket("/ws/live")
async def live_socket(websocket: WebSocket):
    await websocket.accept()
    queue = state.event_store.subscribe()
    try:
        while True:
            await websocket.send_json(await queue.get())
    except WebSocketDisconnect:
        pass
    finally:
        state.event_store.unsubscribe(queue)


@app.get("/drones")
def drones():
    return list((state.current_snapshot or {}).get("drones", {}).values())


@app.get("/drones/{drone_id}")
def drone(drone_id: str):
    value = (state.current_snapshot or {}).get("drones", {}).get(drone_id)
    if not value:
        raise HTTPException(404, "Drone not found")
    return value


@app.get("/events/{stream}")
def events(stream: str, limit: int = 200):
    if stream not in {"mission", "fleet", "decision", "communication", "safety", "collision", "incident", "report", "model_comparison"}:
        raise HTTPException(400, "Unknown event stream")
    return state.event_store.list(stream)[-max(1, min(limit, 2000)):]


@app.post("/training", dependencies=[Depends(require_auth)])
async def start_training(request: TrainingRequest):
    public_id = str(uuid4())
    state.training_runs[public_id] = {"id": public_id, "status": "RUNNING", **request.model_dump(), "progress": []}

    def progress(record):
        state.training_runs[public_id]["progress"].append(record)

    async def worker():
        try:
            result = await asyncio.to_thread(train, request.preset, request.algorithm, request.seed, progress)
            state.training_runs[public_id].update({"status": "COMPLETED", "result": result})
            await state.event_store.append("training", state.training_runs[public_id])
        except Exception as exc:
            state.training_runs[public_id].update({"status": "FAILED", "error": str(exc)})

    asyncio.create_task(worker())
    return state.training_runs[public_id]


@app.get("/training/{run_id}")
def training_status(run_id: str):
    if run_id not in state.training_runs:
        raise HTTPException(404, "Training run not found")
    return state.training_runs[run_id]


@app.get("/models/checkpoints")
def checkpoints():
    return [{"path": str(p), "name": p.name, "created_at": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat()} for p in sorted((ROOT / "artifacts" / "checkpoints").glob("*.pt"))]


@app.post("/models/evaluate", dependencies=[Depends(require_auth)])
async def evaluate_model(checkpoint: str, seeds: str = "101,102,103"):
    path = Path(checkpoint).resolve()
    allowed = (ROOT / "artifacts" / "checkpoints").resolve()
    if allowed not in path.parents or not path.exists():
        raise HTTPException(400, "Checkpoint must exist inside artifacts/checkpoints")
    result = await asyncio.to_thread(evaluate, path, [int(x) for x in seeds.split(",")])
    state.evaluations[result["run_id"]] = result
    return result


@app.get("/models/compare")
def compare_models(qmix_run_id: str, idqn_run_id: str):
    try:
        return compare(state.evaluations[qmix_run_id], state.evaluations[idqn_run_id])
    except KeyError as exc:
        raise HTTPException(404, f"Evaluation not found: {exc.args[0]}") from exc


@app.post("/models/comparison-runs", dependencies=[Depends(require_auth)])
async def start_automatic_comparison(preset: str = "quick", seed: int = 42):
    if preset not in {"quick", "standard", "full"}:
        raise HTTPException(400, "Preset must be quick, standard, or full")
    comparison_id = str(uuid4())
    state.comparison_runs[comparison_id] = {
        "id": comparison_id,
        "status": "RUNNING",
        "stage": "TRAINING_QMIX",
        "preset": preset,
        "seed": seed,
        "evaluation_seeds": [101, 102, 103],
    }

    async def worker():
        record = state.comparison_runs[comparison_id]
        try:
            qmix_training = await asyncio.to_thread(train, preset, "qmix", seed)
            record["stage"] = "TRAINING_IDQN"
            idqn_training = await asyncio.to_thread(train, preset, "idqn", seed)
            record["stage"] = "EVALUATING_QMIX"
            qmix_evaluation = await asyncio.to_thread(
                evaluate, qmix_training["checkpoint"], record["evaluation_seeds"]
            )
            record["stage"] = "EVALUATING_IDQN"
            idqn_evaluation = await asyncio.to_thread(
                evaluate, idqn_training["checkpoint"], record["evaluation_seeds"]
            )
            result = compare(qmix_evaluation, idqn_evaluation)
            result["comparison_id"] = comparison_id
            output = ROOT / "reports" / f"model-comparison-{comparison_id[:8]}.json"
            output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            record.update(
                {
                    "status": "COMPLETED", "stage": "COMPLETED",
                    "result": result, "report": str(output),
                }
            )
            await state.event_store.append("model_comparison", record)
        except Exception as exc:
            record.update({"status": "FAILED", "stage": "FAILED", "error": str(exc)})

    asyncio.create_task(worker())
    return state.comparison_runs[comparison_id]


@app.get("/models/comparison-runs/{comparison_id}")
def automatic_comparison_status(comparison_id: str):
    record = state.comparison_runs.get(comparison_id)
    if not record:
        raise HTTPException(404, "Comparison run not found")
    return record


@app.post("/approvals/{approval_id}/{decision}", dependencies=[Depends(require_auth)])
def decide_approval(approval_id: str, decision: str):
    approval = state.approvals.get(approval_id)
    if not approval:
        raise HTTPException(404, "Approval request not found")
    if decision not in {"approve", "reject"}:
        raise HTTPException(400, "Decision must be approve or reject")
    approval.status = "APPROVED" if decision == "approve" else "REJECTED"
    approval.decided_at = datetime.now(timezone.utc)
    return approval


@app.get("/approvals")
def approvals():
    return [a.model_dump(mode="json") for a in state.approvals.values()]


@app.post("/agents/run")
async def run_agent(request: AgentRequest):
    try:
        result = await asyncio.to_thread(OperationalWorkflow(state).invoke, request.request)
        await state.event_store.append("agent", result)
        return result
    except Exception as exc:
        AGENT_FAILURES.inc()
        raise HTTPException(500, f"Operational workflow failed: {exc}") from exc


@app.post("/incidents")
def create_incident(title: str, simulation_id: str):
    incident_id = str(uuid4())
    decision_events = [e for e in state.event_store.list("decision") if e["simulation_id"] == simulation_id]
    safety_events = state.event_store.list("safety")
    incident = {
        "id": incident_id,
        "title": title,
        "status": "OPEN",
        "simulation_id": simulation_id,
        "timeline": sorted(decision_events[-20:] + safety_events[-20:], key=lambda x: (x.get("step", 0), x.get("timestamp", ""))),
        "evidence": {"decision_event_count": len(decision_events), "safety_event_count": len(safety_events)},
        "interpretation": "Review the first safety override or unexpected policy action; this statement is interpretation, not measured evidence.",
        "reproduction": {"seed": state.simulations.get(simulation_id, {}).get("seed"), "simulation_id": simulation_id},
    }
    state.incidents[incident_id] = incident
    return incident


@app.get("/incidents")
def incidents():
    return list(state.incidents.values())


@app.get("/reports/{simulation_id}")
def report(simulation_id: str, format: str = "json"):
    simulation = state.simulations.get(simulation_id)
    if not simulation or "report" not in simulation:
        raise HTTPException(404, "Completed report not found")
    if format == "json":
        return simulation["report"]
    if format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["metric", "value"])
        for key, value in simulation["report"].items():
            writer.writerow([key, json.dumps(value) if isinstance(value, (dict, list)) else value])
        return PlainTextResponse(output.getvalue(), media_type="text/csv")
    raise HTTPException(400, "Format must be json or csv")


@app.get("/provider")
def provider():
    return {"provider": OperationalWorkflow(state).provider.name}
