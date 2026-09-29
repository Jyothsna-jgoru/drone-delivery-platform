import { FormEvent, useEffect, useMemo, useState } from "react";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";
const WS = API.replace(/^http/, "ws") + "/ws/live";
const auth = { Authorization: "Bearer local-development-token", "Content-Type": "application/json" };

type Envelope = { type: string; data: any };
type Drone = { id: string; position: number[]; battery: number; package_id?: string; status: string; communication_healthy: boolean; path: number[][]; };
type Snapshot = { simulation_id: string; run_id?: string; step: number; grid_size: number; drones: Record<string, Drone>; packages: Record<string, any>; depot: number[]; chargers: number[][]; obstacles: any[]; restricted_zones: any[]; };

const tabs = ["Mission Center", "Live Fleet", "Decision Inspector", "Communication Monitor", "Safety Center", "Training Lab", "Model Comparison", "Incidents", "Model Registry", "Reports"];

async function api(path: string, init?: RequestInit) {
  const response = await fetch(`${API}${path}`, init);
  if (!response.ok) throw new Error((await response.text()) || response.statusText);
  return response.json();
}

function App() {
  const [tab, setTab] = useState("Mission Center");
  const [events, setEvents] = useState<Record<string, any[]>>({});
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [provider, setProvider] = useState("checking");
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    api("/provider").then((x) => setProvider(x.provider)).catch(() => setProvider("unavailable"));
    const socket = new WebSocket(WS);
    socket.onopen = () => setConnected(true);
    socket.onclose = () => setConnected(false);
    socket.onmessage = ({ data }) => {
      const event: Envelope = JSON.parse(data);
      setEvents((current) => ({ ...current, [event.type]: [...(current[event.type] || []).slice(-499), event.data] }));
      if (event.type === "fleet") setSnapshot(event.data);
    };
    return () => socket.close();
  }, []);

  return <div className="app">
    <aside>
      <div className="brand"><span className="mark">A</span><div><b>AERIAL OPS</b><small>FLEET INTELLIGENCE</small></div></div>
      <nav>{tabs.map((name) => <button key={name} className={tab === name ? "active" : ""} onClick={() => setTab(name)}>{name}</button>)}</nav>
      <div className="system"><span className={connected ? "dot live" : "dot"}/><div><b>{connected ? "Live link" : "Disconnected"}</b><small>AI: {provider}</small></div></div>
    </aside>
    <main>
      <header><div><p>Autonomous delivery control</p><h1>{tab}</h1></div><div className="status">{snapshot ? `SIM ${snapshot.simulation_id.slice(0, 8)} · STEP ${snapshot.step}` : "NO ACTIVE SIMULATION"}</div></header>
      {tab === "Mission Center" && <MissionCenter />}
      {tab === "Live Fleet" && <LiveFleet snapshot={snapshot} safety={events.safety || []} />}
      {tab === "Decision Inspector" && <DecisionInspector decisions={events.decision || []} />}
      {tab === "Communication Monitor" && <CommunicationMonitor messages={events.communication || []} />}
      {tab === "Safety Center" && <SafetyCenter events={events.safety || []} />}
      {tab === "Training Lab" && <TrainingLab />}
      {tab === "Model Comparison" && <ModelComparison />}
      {tab === "Incidents" && <Incidents snapshot={snapshot} />}
      {tab === "Model Registry" && <ModelRegistry />}
      {tab === "Reports" && <Reports snapshot={snapshot} />}
    </main>
    <footer>Built by Jyothsna Devi Goru</footer>
  </div>;
}

function MissionCenter() {
  const [form, setForm] = useState({ description: "Deliver a critical 1.5 kg medical package", x: 17, y: 16, weight: 1.5, deadline: 100, priority: "CRITICAL" });
  const [result, setResult] = useState<any>(null);
  const [mission, setMission] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const payload = () => ({ description: form.description, pickup: [1, 1], destination: [Number(form.x), Number(form.y)], package_weight: Number(form.weight), deadline_steps: Number(form.deadline), priority: form.priority });
  async function validate(e: FormEvent) { e.preventDefault(); setBusy(true); try { setResult(await api("/missions/validate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload()) })); } finally { setBusy(false); } }
  async function submit() { const created = await api("/missions", { method: "POST", headers: auth, body: JSON.stringify(payload()) }); setMission(created); }
  async function start() { if (!mission) return; const sim = await api(`/simulations?mission_id=${mission.id}&seed=42&pace=0.12`, { method: "POST", headers: auth }); setResult({ ...result, simulation: sim }); }
  return <section className="grid two">
    <form className="panel" onSubmit={validate}><div className="eyebrow">MISSION INTAKE</div><h2>Plan a delivery</h2>
      <label>Natural-language request<textarea value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })}/></label>
      <div className="form-grid"><label>Destination X<input type="number" value={form.x} onChange={(e) => setForm({ ...form, x: +e.target.value })}/></label><label>Destination Y<input type="number" value={form.y} onChange={(e) => setForm({ ...form, y: +e.target.value })}/></label><label>Weight (kg)<input type="number" step="0.1" value={form.weight} onChange={(e) => setForm({ ...form, weight: +e.target.value })}/></label><label>Deadline (steps)<input type="number" value={form.deadline} onChange={(e) => setForm({ ...form, deadline: +e.target.value })}/></label><label>Priority<select value={form.priority} onChange={(e) => setForm({ ...form, priority: e.target.value })}><option>NORMAL</option><option>HIGH</option><option>CRITICAL</option></select></label></div>
      <div className="actions"><button className="primary" disabled={busy}>{busy ? "Checking…" : "Validate mission"}</button><button type="button" disabled={!result?.valid} onClick={submit}>Submit</button><button type="button" disabled={!mission} onClick={start}>Launch simulation</button></div>
    </form>
    <div className="panel"><div className="eyebrow">PREFLIGHT EVIDENCE</div><h2>{result ? (result.valid ? "Ready for approval" : "Mission blocked") : "Awaiting validation"}</h2>{result && <><div className={`verdict ${result.valid ? "pass" : "fail"}`}>{result.valid ? "All deterministic checks passed" : result.errors.join(" · ")}</div><div className="checks">{Object.entries(result.checks || {}).map(([name, value]: any) => <div key={name}><span>{name.replace("_", " ")}</span><b className={value.passed ? "good" : "bad"}>{value.passed ? "PASS" : "FAIL"}</b></div>)}</div><pre>{JSON.stringify(mission || payload(), null, 2)}</pre></>}</div>
  </section>;
}

function LiveFleet({ snapshot, safety }: { snapshot: Snapshot | null; safety: any[] }) {
  const [selected, setSelected] = useState<string | null>(null);
  if (!snapshot) return <Empty text="Start a simulation from Mission Center to receive live positions."/>;
  const scale = 30;
  const drone = selected ? snapshot.drones[selected] : null;
  return <section className="grid fleet-layout"><div className="panel map-panel"><div className="map-key"><span>Live simulator telemetry · Orange = destination</span><span>{Object.keys(snapshot.drones).length} aircraft</span></div><svg className="map" viewBox="-10 -10 640 640">
    <defs><pattern id="grid" width="31" height="31" patternUnits="userSpaceOnUse"><path d="M 31 0 L 0 0 0 31" fill="none" stroke="#1a3941" strokeWidth="1"/></pattern></defs><rect width="620" height="620" fill="url(#grid)"/>
    {snapshot.restricted_zones.map((z: any) => <rect key={z.id} x={z.min_x*scale} y={(20-z.max_y)*scale} width={(z.max_x-z.min_x)*scale} height={(z.max_y-z.min_y)*scale} className="restricted"/>)}
    {snapshot.obstacles.map((o: any, i) => <rect key={i} x={o.x*scale} y={(20-o.y-o.height)*scale} width={o.width*scale} height={o.height*scale} className="building"/>)}
    {snapshot.chargers.map((c, i) => <g key={i} transform={`translate(${c[0]*scale},${(20-c[1])*scale})`}><circle r="12" className="charger"/><text y="5">⚡</text></g>)}
    {Object.values(snapshot.packages).filter((p: any) => p.active !== false).map((p: any) => <g key={p.id}><circle cx={p.destination[0]*scale} cy={(20-p.destination[1])*scale} r="7" className="destination"/><text x={p.destination[0]*scale+(p.destination[0]>16?-10:10)} y={(20-p.destination[1])*scale+4} textAnchor={p.destination[0]>16?"end":"start"} fill="#ffb45b" fontSize="10">DEST ({p.destination[0]}, {p.destination[1]})</text></g>)}
    {Object.values(snapshot.drones).map((d) => <g key={d.id} onClick={() => setSelected(d.id)} className="drone" transform={`translate(${d.position[0]*scale},${(20-d.position[1])*scale})`}><circle r="22" className="risk"/><circle r="9" className="craft"/><text x={d.position[0]>18?-13:13} textAnchor={d.position[0]>18?"end":"start"} y="-11">{d.id}</text></g>)}
  </svg></div><div className="stack"><div className="panel"><div className="eyebrow">FLEET</div>{Object.values(snapshot.drones).map((d) => <button className="drone-row" key={d.id} onClick={() => setSelected(d.id)}><b>{d.id}</b><span>{d.status}</span><span>{Math.round(d.battery*100)}%</span><i style={{ width: `${d.battery*100}%` }}/></button>)}</div><div className="panel"><div className="eyebrow">SELECTED AIRCRAFT</div>{drone ? <dl><dt>Position</dt><dd>{drone.position.map(x => x.toFixed(1)).join(", ")}</dd><dt>Altitude</dt><dd>{drone.position[2].toFixed(1)} m</dd><dt>Package</dt><dd>{drone.package_id || "None"}</dd><dt>Communication</dt><dd>{drone.communication_healthy ? "Healthy" : "Degraded"}</dd></dl> : <p>Select a drone on the map.</p>}</div><div className="panel alert"><b>{safety.length}</b><span>Safety interventions this session</span></div></div></section>;
}

function DecisionInspector({ decisions }: { decisions: any[] }) {
  const [index, setIndex] = useState(-1);
  useEffect(() => { if (decisions.length) setIndex(decisions.length - 1); }, [decisions.length]);
  const event = decisions[index];
  if (!event) return <Empty text="Decision events appear here as the simulator advances."/>;
  return <section className="panel"><div className="decision-head"><div><div className="eyebrow">RECORDED DECISION · {event.event_id?.slice(0,8)}</div><h2>{event.drone_id} · step {event.step}</h2></div><div className="actions"><button onClick={() => setIndex(Math.max(0,index-1))}>Previous</button><button onClick={() => setIndex(Math.min(decisions.length-1,index+1))}>Next</button></div></div><div className="decision-grid"><DataCard title="Local observation" value={event.observation_summary}/><DataCard title="Received messages" value={event.received_messages}/><DataCard title="Available actions" value={event.available_actions}/><DataCard title="Model action scores (Q-values)" value={event.policy_q_values}/><DataCard title="Policy → executed" value={{ policy_action: event.policy_action, executed_action: event.executed_action, model_version: event.model_version }}/><DataCard title="Deterministic safety result" value={event.safety_result}/><DataCard title={`Reward ${event.reward.toFixed(3)}`} value={event.reward_breakdown}/></div></section>;
}

function CommunicationMonitor({ messages }: { messages: any[] }) { const rows=messages.map(m=>({sender:m.sender,receiver:m.receiver,position:m.payload?.position?.join(", "),next_waypoint:m.payload?.next_waypoint?.join(", "),planned_altitude:m.payload?.planned_altitude,battery:m.payload?`${Math.round(m.payload.battery*100)}%`:"—",priority:m.payload?.package_priority,latency_ms:m.latency_ms,dropped:m.dropped,conflict_acknowledged:m.conflict_acknowledgement})); return <section className="grid"><div className="panel"><div className="eyebrow">SIMULATED LOCAL RADIO CHANNEL</div><h2>What the drones are broadcasting</h2><p>Each in-range drone sends its position, intended waypoint, altitude, battery and mission priority. Delay and message loss are simulated. A conflict acknowledgement means two intended paths overlapped and the deterministic priority protocol selected which drone should yield.</p></div><Table title="Actual messages generated by the simulator" rows={rows} columns={["sender","receiver","position","next_waypoint","planned_altitude","battery","priority","latency_ms","dropped","conflict_acknowledged"]}/></section>; }
function SafetyCenter({ events }: { events: any[] }) { return <section className="grid two"><div className="panel"><div className="eyebrow">ACTIVE RULES · VERSION 1.0.0</div><h2>Hard safety shield</h2><div className="rule-list">{["Minimum battery reserve: 15%","Payload maximum: 5 kg","Horizontal separation: 1.5 m","Vertical separation: 1.0 m","Altitude: 1–12 m","Restricted zone RZ-1","Emergency landing: 5% battery"].map(x=><span key={x}>{x}</span>)}</div></div><Table title="Rejected and replaced actions" rows={events} columns={["step","drone_id","original_action","replacement_action","violated_rules","reason"]}/></section>; }

function TrainingLab() {
  const [algorithm, setAlgorithm] = useState("qmix"), [preset, setPreset] = useState("quick"), [run, setRun] = useState<any>(null);
  useEffect(() => { if (!run?.id || run.status !== "RUNNING") return; const timer=setInterval(()=>api(`/training/${run.id}`).then(setRun),1000); return()=>clearInterval(timer); },[run?.id,run?.status]);
  async function start(){ setRun(await api("/training",{method:"POST",headers:auth,body:JSON.stringify({algorithm,preset,seed:42})})); }
  const data = run?.progress || run?.result?.history || [];
  return <section className="grid two"><div className="panel"><div className="eyebrow">EXPERIMENT CONTROL</div><h2>Train on real simulator episodes</h2><label>Algorithm<select value={algorithm} onChange={e=>setAlgorithm(e.target.value)}><option value="qmix">QMIX</option><option value="idqn">Independent DQN</option></select></label><label>Configuration<select value={preset} onChange={e=>setPreset(e.target.value)}><option>quick</option><option>standard</option><option>full</option></select></label><button className="primary" onClick={start} disabled={run?.status==="RUNNING"}>Start training</button>{run&&<pre>{JSON.stringify({id:run.id,status:run.status,checkpoint:run.result?.checkpoint,error:run.error},null,2)}</pre>}</div><div className="panel chart"><div className="eyebrow">ACTUAL TRAINING TELEMETRY</div><h2>Team reward</h2>{data.length?<ResponsiveContainer width="100%" height={320}><LineChart data={data}><XAxis dataKey="episode"/><YAxis/><Tooltip/><Line type="monotone" dataKey="reward" stroke="#56e0b1" dot={false}/><Line type="monotone" dataKey="loss" stroke="#ffb45b" dot={false}/></LineChart></ResponsiveContainer>:<p>No metrics until a training run starts.</p>}</div></section>;
}

function ModelComparison() { const [run,setRun]=useState<any>(null); useEffect(()=>{if(!run?.id||run.status!=="RUNNING")return;const timer=setInterval(()=>api(`/models/comparison-runs/${run.id}`).then(setRun),1000);return()=>clearInterval(timer)},[run?.id,run?.status]); async function start(){setRun(await api("/models/comparison-runs?preset=quick&seed=42",{method:"POST",headers:auth}))} const result=run?.result; const rows=result?Object.keys(result.qmix.summary).map(metric=>({metric,qmix:`${result.qmix.summary[metric].mean.toFixed(4)} ± ${result.qmix.summary[metric].std.toFixed(4)}`,idqn:`${result.idqn.summary[metric].mean.toFixed(4)} ± ${result.idqn.summary[metric].std.toFixed(4)}`,winner:result.winner_by_metric[metric]})):[]; return <section className="grid"><div className="panel"><div className="eyebrow">AUTOMATIC MEASURED COMPARISON</div><h2>QMIX vs independent DQN</h2><p>This runs both algorithms with the same quick configuration, evaluates them on seeds 101, 102 and 103, and compares only measured results.</p><button className="primary" onClick={start} disabled={run?.status==="RUNNING"}>{run?.status==="RUNNING"?`Working: ${run.stage.replaceAll("_"," ")}`:"Train, evaluate and compare both models"}</button>{run?.status==="FAILED"&&<div className="verdict fail">{run.error}</div>}{result&&<p>QMIX run {result.qmix.run_id.slice(0,8)} · IDQN run {result.idqn.run_id.slice(0,8)} · {result.qmix.episodes} evaluation episodes each</p>}</div>{rows.length>0&&<Table title="Real evaluation metrics — no invented results" rows={rows} columns={["metric","qmix","idqn","winner"]}/>}</section>; }
function Incidents({snapshot}:{snapshot:Snapshot|null}) { const [items,setItems]=useState<any[]>([]); useEffect(()=>{let active=true;const load=()=>api("/incidents").then(x=>{if(active)setItems(x)});load();const timer=setInterval(load,2000);return()=>{active=false;clearInterval(timer)}},[snapshot?.run_id]); return <section className="panel"><div className="eyebrow">AUTOMATIC SAFETY EVIDENCE</div><h2>Incident review</h2><p>Incidents are created automatically only when a real safety override, collision risk, restricted-zone attempt, battery emergency or communication conflict occurs.</p>{items.length?items.map(x=><DataCard key={x.id} title={x.title} value={x}/>):<div className="verdict pass">No safety incidents have occurred in this session.</div>}</section>; }
function ModelRegistry() { const [models,setModels]=useState<any[]>([]); useEffect(()=>{api("/models/checkpoints").then(setModels)},[]); return <Table title="Saved model checkpoints" rows={models} columns={["name","path","created_at"]}/>; }
function Reports({snapshot}:{snapshot:Snapshot|null}) { const [report,setReport]=useState<any>(null),[error,setError]=useState(""); useEffect(()=>{if(snapshot?.run_id)api(`/reports/${snapshot.run_id}`).then(setReport).catch(e=>setError(String(e)))},[snapshot?.run_id,snapshot?.step]); if(!snapshot?.run_id)return <Empty text="Complete a mission to generate its JSON and CSV reports."/>; return <section className="panel"><div className="eyebrow">GENERATED FROM STORED EVENTS</div><h2>Mission report</h2>{report?<><div className="actions"><a href={`${API}/reports/${snapshot.run_id}?format=json`} target="_blank">Open JSON</a><a href={`${API}/reports/${snapshot.run_id}?format=csv`} target="_blank">Open CSV</a></div><pre>{JSON.stringify(report,null,2)}</pre></>:<p>{error||"Waiting for the simulation to complete…"}</p>}</section>; }
function DataCard({title,value}:{title:string;value:any}) { return <div className="data-card"><h3>{title}</h3><pre>{JSON.stringify(value,null,2)}</pre></div>; }
function Empty({text}:{text:string}) { return <div className="panel empty"><div className="radar"/><h2>Waiting for operational data</h2><p>{text}</p></div>; }
function Table({title,rows,columns}:{title:string;rows:any[];columns:string[]}) { return <div className="panel table-wrap"><div className="eyebrow">{title}</div><table><thead><tr>{columns.map(c=><th key={c}>{c.replaceAll("_"," ")}</th>)}</tr></thead><tbody>{rows.slice().reverse().slice(0,100).map((r,i)=><tr key={r.event_id||i}>{columns.map(c=><td key={c}>{typeof r[c]==="object"?JSON.stringify(r[c]):String(r[c]??"—")}</td>)}</tr>)}</tbody></table>{!rows.length&&<p>No events recorded.</p>}</div>; }

export default App;

