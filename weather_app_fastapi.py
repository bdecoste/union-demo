"""FastAPI + passthrough-auth app for the weather-route agent (hosted Union).

Why this exists: on a hosted cluster, tasks must run with the *caller's* identity.
FastAPIPassthroughAuthMiddleware auto-extracts the caller's Authorization/Cookie
headers on every request and forwards them to Flyte, so each run executes as that
user — and the organization is resolved from their token. That fixes the empty
`project_id.organization` the hand-rolled Gradio header capture couldn't supply.

Deploy (hosted):
    export FLYTE_ENDPOINT=dns:///<your-endpoint.hosted.unionai.cloud>
    export FLYTE_INSECURE=false
    # FLYTE_ORG is optional — usually resolved from the caller's token:
    # export FLYTE_ORG=<org-slug>
    flyte deploy weather_app_fastapi.py app_env

Prereq: the workflow must be deployed on the SAME hosted cluster/project/domain:
    flyte deploy weather_route_agent.py data_env      # against your hosted config

Call it (JSON API):
    flyte get api-key weather-key
    curl -X POST -H "Authorization: Bearer <key>" -H "Content-Type: application/json" \
         -d '{"start":"San Diego, CA","end":"Las Vegas, NV","use_judge":true}' \
         https://<app-host>/briefing
Or just open the app URL in a browser — you're already authenticated via the gateway.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from starlette import status

import flyte
import flyte.errors
import flyte.remote as remote
from flyte.app.extras import FastAPIAppEnvironment, FastAPIPassthroughAuthMiddleware

FLYTE_PROJECT = os.getenv("FLYTE_PROJECT", "billdecoste")
FLYTE_DOMAIN = os.getenv("FLYTE_DOMAIN", "development")
FLYTE_ORG = os.getenv("FLYTE_ORG", "")
FLYTE_ENDPOINT = os.getenv("FLYTE_ENDPOINT", "")
FLYTE_INSECURE = os.getenv("FLYTE_INSECURE", "false").lower() == "true"

TASK_NAME = "weather_route.main"  # {environment_name}.{task_name}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize passthrough auth on startup.

    In-cluster, init_passthrough auto-configures the endpoint from the injected
    environment, so we pass endpoint=None and don't require FLYTE_ENDPOINT. Set
    FLYTE_ENDPOINT only when running the app OUTSIDE the cluster, where there's
    nothing to infer from.
    """
    await flyte.init_passthrough.aio(
        endpoint=os.getenv("FLYTE_ENDPOINT") or None,   # None -> infer from in-cluster injection
        org=os.getenv("FLYTE_ORG") or None,             # usually resolved from the caller's token
        project=FLYTE_PROJECT,
        domain=FLYTE_DOMAIN,
        insecure=os.getenv("FLYTE_INSECURE", "false").lower() == "true",
    )
    yield


app = FastAPI(title="Weather Along the Route", lifespan=lifespan)
# Auto-extracts Authorization + Cookie per request and sets them in the Flyte context.
app.add_middleware(FastAPIPassthroughAuthMiddleware, excluded_paths={"/health"})


class BriefingRequest(BaseModel):
    start: str
    end: str
    use_judge: bool = True


@app.get("/health")
async def health():
    """Liveness check (excluded from auth)."""
    return {"status": "healthy"}


@app.post("/briefing")
async def start_briefing(req: BriefingRequest):
    """Start the agent as the calling user; return the run name + URL immediately.

    The run is not awaited here (a long route would exceed the gateway timeout);
    poll GET /result/{run_name} for the briefing.
    """
    try:
        task = remote.Task.get(
            TASK_NAME, project=FLYTE_PROJECT, domain=FLYTE_DOMAIN, auto_version="latest"
        )
        run = await flyte.run.aio(task, start=req.start, end=req.end, use_judge=req.use_judge)
        return {"name": run.name, "url": run.url}
    except flyte.errors.RemoteTaskNotFoundError:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Task {TASK_NAME} not found in {FLYTE_PROJECT}/{FLYTE_DOMAIN} — "
            "is the workflow deployed on this cluster?",
        )
    except flyte.errors.RemoteTaskUsageError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(e))


@app.get("/result/{run_name}")
async def result(run_name: str):
    """Poll a run: returns its phase, and the briefing once it has succeeded."""
    try:
        run = await remote.Run.get.aio(run_name)
        action = getattr(run, "action", None)
        phase = str(getattr(action, "phase", ""))
        done = bool(getattr(action, "done", lambda: False)())
        briefing = None
        if done:
            try:
                outputs = await run.outputs.aio()
                briefing = outputs[0] if outputs else None
            except Exception as exc:  # failed run -> no readable outputs
                briefing = f"(run finished without readable output: {exc})"
        return {"name": run_name, "phase": phase, "done": done, "briefing": briefing}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(e))


INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Weather Along the Route</title>
<style>
  body{font-family:system-ui,sans-serif;max-width:780px;margin:40px auto;padding:0 16px}
  input[type=text]{padding:8px;width:260px} button{padding:8px 16px}
  #out{white-space:pre-wrap;border:1px solid #ddd;padding:16px;margin-top:16px;border-radius:8px}
  .row{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:12px 0}
</style></head>
<body>
  <h1>Weather Along the Route</h1>
  <p>Enter two US locations (place names or ZIP codes).</p>
  <div class="row">
    <input type="text" id="start" value="San Diego, CA" placeholder="Start">
    <input type="text" id="end" value="Las Vegas, NV" placeholder="End">
  </div>
  <div class="row">
    <label><input type="checkbox" id="judge" checked> Run hallucination judge</label>
    <button onclick="go()">Get briefing</button>
  </div>
  <div id="link"></div>
  <div id="out"></div>
<script>
async function go(){
  const out=document.getElementById('out'), link=document.getElementById('link');
  out.textContent=''; link.innerHTML='Starting run…';
  const body={start:start.value,end:end.value,use_judge:judge.checked};
  const r=await fetch('/briefing',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(!r.ok){link.innerHTML='Error: '+await r.text();return;}
  const d=await r.json();
  link.innerHTML='<a href="'+d.url+'" target="_blank">View run on Flyte</a>';
  poll(d.name,out);
}
async function poll(name,out){
  const r=await fetch('/result/'+encodeURIComponent(name));
  const d=await r.json();
  if(d.done){out.textContent=d.briefing||'(no output)';}
  else{out.textContent='Status: '+(d.phase||'running')+'…';setTimeout(()=>poll(name,out),3000);}
}
</script>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return INDEX_HTML


# ---- Deployment config -------------------------------------------------------
image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
    "flyte>=2.1.4", "fastapi", "uvicorn", "pydantic",
)

# Only forward org/endpoint when set, so empty strings don't override injected values.
_env_vars = {"FLYTE_PROJECT": FLYTE_PROJECT, "FLYTE_DOMAIN": FLYTE_DOMAIN,
             "FLYTE_INSECURE": str(FLYTE_INSECURE).lower()}
if FLYTE_ENDPOINT:
    _env_vars["FLYTE_ENDPOINT"] = FLYTE_ENDPOINT
if FLYTE_ORG:
    _env_vars["FLYTE_ORG"] = FLYTE_ORG

app_env = FastAPIAppEnvironment(
    name="weather-route-webhook",
    app=app,
    description="Weather-route agent behind FastAPI with passthrough auth",
    image=image,
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    requires_auth=True,   # platform handles auth at the gateway
    env_vars=_env_vars,
)
