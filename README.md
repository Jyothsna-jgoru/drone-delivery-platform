# Agentic Multi-Agent Reinforcement Learning Platform for Autonomous Drone Delivery

A working local simulation and operations platform for cooperative delivery drones. Independent drone agents act from local observations, exchange range-limited messages, and are always mediated by deterministic safety rules. A separate operational-agent workflow plans missions, analyzes the fleet and incidents, and requests human approval for protected actions.

The default implementation is a simulation platform. It is not certified real-world autonomous-flight software and must not be used to control aircraft.

## What runs end to end

- A seeded PettingZoo parallel environment with packages, depots, destinations, chargers, altitude, wind, payload, battery, deadlines, static/restricted obstacles, failures, noisy local sensing, and communication loss/delay.
- Predictive collision checking based on relative position, velocity, time to closest approach, and horizontal/vertical separation. Unsafe actions are replaced by the safety shield and both actions are recorded.
- A local drone message network and deterministic conflict-priority protocol.
- A GRU agent network, monotonic QMIX mixer, replay buffer, target networks, epsilon-greedy action masking, checkpoints, evaluation, and an independent-DQN baseline.
- FastAPI REST and WebSocket services with local authentication, validation, simulation control, training, evaluation, approvals, incident timelines, reports, Prometheus metrics, and correlation IDs.
- A React control room whose map, alerts, decision scores, charts, message monitor, reports, and model registry are populated only by backend events or completed model runs.
- PostgreSQL/Alembic durable schemas, Redis runtime state with an explicit in-memory fallback, local MLflow, and optional Ollama.

## Architecture

```text
React/Vite ── REST + WebSocket ── FastAPI
                                  ├─ PettingZoo simulator
                                  │  ├─ local observations → QMIX/IDQN/rule policy
                                  │  ├─ communication channel
                                  │  └─ deterministic safety shield → executed action
                                  ├─ JSONL replay events + PostgreSQL records
                                  ├─ Redis live state/fanout (memory fallback)
                                  ├─ MLflow + checkpoint artifacts
                                  └─ operational workflow
                                     ├─ Ollama or deterministic provider
                                     ├─ typed read/simulation tools
                                     └─ approval-gated protected tools
```

Drone agents and operational agents are deliberately separate. Drone agents produce high-level discrete action proposals from local observations. The safety shield decides whether each proposal may execute. Operational agents read state, validate plans, and create approval requests. LLM output never controls motors, modifies arbitrary database rows, runs a shell, sends unrestricted HTTP requests, or overrides a safety rule.

## Fastest start: Docker

Requirements: Docker Desktop with Compose.

On Windows, `RUN_PROJECT.ps1` is the one-command local launcher. It creates the isolated environment, installs the declared dependencies, starts the API, and serves the dashboard; it does not publish anything externally.

```powershell
.\RUN_PROJECT.ps1
```

```bash
cp .env.example .env
docker compose up --build -d
docker compose exec api alembic -c infra/alembic.ini upgrade head
```

Open:

- Control room: <http://localhost:8080>
- API documentation: <http://localhost:8000/docs>
- Prometheus metrics: <http://localhost:8000/metrics>
- MLflow: <http://localhost:5000>

The browser uses the local development token. For non-local use, change all credentials and place the services behind an authenticated reverse proxy.

Ollama is optional:

```bash
docker compose --profile ollama up -d ollama
docker compose exec ollama ollama pull llama3.2:3b
```

Without Ollama, the UI reports `deterministic-fallback` and operational workflows continue using typed parsers and fixed rules.

## Local development

Python 3.12 and Node 22 are recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\python -m pip install -e ".[dev]"
# macOS/Linux
.venv/bin/python -m pip install -e ".[dev]"

cd apps/web
npm install
cd ../..
alembic -c infra/alembic.ini upgrade head
python -m uvicorn apps.api.main:app --reload
```

In a second terminal:

```bash
cd apps/web
npm run dev
```

The backend defaults to SQLite and an in-memory runtime-state provider when PostgreSQL or Redis is unavailable. It reports the active state backend at `/ready`; fallback is visible, never silent. Docker uses PostgreSQL and Redis.

## Training and measured comparison

Quick CPU validation:

```bash
python -m packages.marl.training --preset quick --algorithm qmix --seed 42
python -m packages.marl.training --preset quick --algorithm idqn --seed 42
```

Evaluate a saved checkpoint:

```bash
python -m packages.marl.evaluation artifacts/checkpoints/qmix-quick-<run>.pt --seeds 101,102,103
```

Run both training jobs, evaluate three held-out seeds, and write the actual comparison:

```bash
python scripts/train_and_compare.py
```

The **Model Comparison** page performs this complete quick workflow with one button; users do not need to copy run IDs. It displays the measured mean and standard deviation for each metric and saves a JSON report.

`quick` has eight short episodes for pipeline validation. `standard` and `full` live in `configs/training.yaml`. Evaluation reports mean and standard deviation across the supplied seeds. The application does not claim a fixed improvement or fabricate missing results. Checkpoints are written to `artifacts/checkpoints/`; evaluation and comparison reports go to `reports/`.

## Simulator and decentralized execution

Each drone observation is a normalized 32-value local vector plus an action-availability mask. It includes own kinematics, battery and temperature, payload/assignment data, deadline and target bearing, nearby obstacles and drones, local wind, charger distance, communication health, previous action, and mission state. The concatenated global state is exposed only to the QMIX mixer during centralized training.

Actions are high-level commands from `MOVE_NORTH` through `EMERGENCY_LAND`. The environment first masks actions that are impossible in the current state, then predicts the result and applies the shield. Pickup, delivery, charge, battery, energy, deadline, and reward changes come from the executed state transition.

### Collision detection

For every nearby peer, the shield computes relative position `r`, relative velocity `v`, clamps `-r·v / |v|²` to the prediction horizon, and measures horizontal and vertical separation at that time. A proposal is unsafe only when both configured separation limits are violated. The shield can hover, climb, return to base, or land depending on the violated rules. `configs/safety_rules.yaml` is versioned and every decision records its rule version.

### Communication

Each broadcast contains position, velocity, waypoint, planned altitude, battery, priority, mission status, and timestamp. Only in-range recipients are queued. The seeded network simulates latency, loss, stale state, and recovery. Conflict priority is deterministic: package priority, battery/maneuver cost, then drone ID. WebSockets and Redis distribute telemetry; they are not part of collision avoidance.

When two intended waypoints conflict, both messages are acknowledged, the priority protocol chooses a winner, and the lower-priority drone proposes a validated climb or hover. The Communication Monitor exposes these actual simulator messages and acknowledgements in plain columns.

## Operational agents and tools

The supervisor routes requests to Mission Planning, Fleet Operations, Maintenance, Incident Analysis, or Model Evaluation. The provider abstraction selects Ollama when reachable and otherwise uses deterministic structured classification. Tools have Pydantic inputs and explicit categories:

- Read: fleet/drone/health/missions/weather/geofences/training/incidents/models.
- Simulation: mission validation, energy estimation, feasibility and route comparison.
- Protected: pause mission, return to base, create incident, or request model promotion. These create `PENDING` approval records only.

Model promotion is never automatic. Maintenance conclusions use measured thresholds/anomaly evidence; language-model text alone cannot mark a vehicle safe.

## Decision inspection and replay

Every step records the local observation summary, received messages, action mask, model scores, proposed action, safety decision, executed action, detailed reward, model version, timestamp, and resulting position. The Decision Inspector provides previous/next controls. A completed run can be replayed deterministically from the `decision`, `communication`, `collision`, and `safety` event streams in `data/events`; replay does not rerun random logic.

## Reports and API

Mission reports contain deliveries, on-time status, duration, distance, energy, minimum separation, warnings, interventions, communication failures, final states, model version, and incident references. They are available from `GET /reports/{simulation_id}?format=json|csv` and saved under `reports/`.

Important routes:

- `POST /missions`, `POST /missions/validate`, `POST /missions/{id}/feasibility`
- `POST /simulations`, `POST /simulations/{id}/pause|resume|stop`
- `GET /drones`, `GET /events/decision|communication|safety|collision`
- `POST /training`, `GET /training/{id}`, `GET /models/checkpoints`
- `POST /models/evaluate`, `GET /models/compare`
- `POST /agents/run`, `GET /approvals`, `POST /approvals/{id}/approve|reject`
- `POST /incidents`, `GET /incidents`, `GET /reports/{simulation_id}`
- `WS /ws/live`

Interactive schemas and examples are at `/docs`.

## Tests

```bash
pytest -q
cd apps/web && npm run lint && npm run build
cd apps/web && npx playwright install chromium && npm run test:e2e
```

Tests cover reset/step, normalized observations, masks, pickup/delivery, battery/charging, deadlines, collision prediction, action replacement, restricted zones, range/loss/delay, QMIX shapes and monotonicity, replay, checkpoint reload, typed-tool authorization, API auth/readiness, metrics, and live WebSocket events. The Playwright workflow validates a mission, submits it, starts a real simulation, receives fleet and decision data, and opens the Decision Inspector.

## Troubleshooting

- `/ready` says `in-memory-fallback`: start Redis or confirm `REDIS_URL`. This is supported for development; durable live state needs Redis.
- No training chart: start a run and wait for the first simulator episode. Values are not prefilled.
- Ollama is unavailable: this is expected unless the optional profile and model are running.
- PyTorch is slow: use `quick`; it is CPU-sized. A GPU is optional and not required.
- PostgreSQL connection errors: use Docker health checks, or omit `DATABASE_URL` to use local SQLite.
- Frontend is disconnected: confirm API port 8000, CORS origin, and `VITE_API_URL`.

## Known limitations and future integration

The grid simulator uses discrete high-level movement and simplified aerodynamics, so results do not transfer directly to aircraft. The ROS 2 and PX4/MAVLink adapters are locked integration boundaries: they fail closed until their runtimes and an explicit bridge are configured. Future work should add PX4 SITL scenario validation, ROS 2 QoS testing, richer wind/vehicle dynamics, larger map curricula, PostgreSQL-backed event querying at scale, and certified flight-controller integration processes.

Built by Jyothsna Devi Goru
