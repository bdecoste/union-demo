"""Gradio UI for the weather-route agent — runs the agent as a Flyte task.

One codebase, two backends (same pattern as the Flyte devbox workshop app):

- No-auth devbox: the app runs tasks as its own in-cluster identity.
- Auth'd cluster (e.g. Union tryv2): the logged-in user's credentials ride in
  on the request and each run executes AS THAT USER (passthrough).

Progression:
  1. Local app + local task:   RUN_MODE=local ANTHROPIC_API_KEY=sk-... python weather_app.py
  2. Local app + remote task:  python weather_app.py        (needs: flyte deploy weather_route_agent.py data_env)
  3. Deploy app to cluster:    flyte deploy weather_app.py serving_env
"""

import os
from contextlib import nullcontext

from dotenv import load_dotenv
import gradio as gr

import flyte
import flyte.app
import flyte.remote as remote
from flyte.remote import auth_metadata

load_dotenv()

RUN_MODE = os.getenv("RUN_MODE", "remote")

# Set APP_REQUIRES_AUTH=true on a cluster with auth in front; leave false on the devbox.
REQUIRES_AUTH = os.getenv("APP_REQUIRES_AUTH", "false").lower() == "true"

# The app looks for the task in the SAME project/domain the workflow was deployed to.
FLYTE_PROJECT = os.getenv("FLYTE_PROJECT", "flytesnacks")
FLYTE_DOMAIN = os.getenv("FLYTE_DOMAIN", "development")

# Devbox is plaintext (FLYTE_INSECURE=true); a real cluster has TLS.
FLYTE_INSECURE = os.getenv("FLYTE_INSECURE", "false").lower() == "true"

# Hosted Union clusters scope projects under an organization; the devbox has none.
# Set FLYTE_ORG to your org slug (visible in the Union console URL, or `flyte get project`).
FLYTE_ORG = os.getenv("FLYTE_ORG", "tryv2")

# Passthrough may need the endpoint explicitly when in-cluster injection isn't present.
FLYTE_ENDPOINT = os.getenv("FLYTE_ENDPOINT", "")

# Flipped on by the server once passthrough auth is initialized.
_PASSTHROUGH = False

serving_env = flyte.app.AppEnvironment(
    name="weather-route-ui",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "flyte>=2.1.4", "gradio", "python-dotenv",
    ),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    requires_auth=REQUIRES_AUTH,
    # Carry deploy-time config into the running container so the runtime task
    # lookup matches where the workflow was deployed.
    env_vars={
        "FLYTE_PROJECT": FLYTE_PROJECT,
        "FLYTE_DOMAIN": FLYTE_DOMAIN,
        "FLYTE_INSECURE": str(FLYTE_INSECURE).lower(),
        "FLYTE_ORG": FLYTE_ORG,
        "FLYTE_ENDPOINT": FLYTE_ENDPOINT,
    },
    port=7860,
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=300),
)

# Pre-registered task reference, fetched from the control plane at runtime.
# Deploy tasks first: flyte deploy weather_route_agent.py data_env
weather_route_task = remote.Task.get(
    "weather_route.main",
    project=FLYTE_PROJECT,
    domain=FLYTE_DOMAIN,
    auto_version="latest",
)


def _resolve_task():
    """Local task object (in-process) vs the deployed remote handle."""
    if RUN_MODE == "local":
        from weather_route_agent import main  # module name must be import-safe (no hyphens)
        return main
    return weather_route_task


def _auth_tuples(request):
    """Pull the caller's auth headers off the Gradio request (passthrough only)."""
    if not (_PASSTHROUGH and request is not None):
        return []
    tuples = []
    for header in ("authorization", "cookie"):
        value = request.headers.get(header)
        if value:
            tuples.append((header, value))
    return tuples


def _auth_ctx(auth):
    """Forward `auth` to Flyte for one call, or a no-op when there's none."""
    return auth_metadata(*auth) if auth else nullcontext()


def _browser_url(run) -> str:
    """run.url points at the in-cluster host on the devbox; rewrite for the browser."""
    url = str(getattr(run, "url", "") or "")
    return url.replace("flyte-binary-http.flyte:8090", "localhost:30080")


def run_query(start, end, use_judge, request: gr.Request = None):
    """Kick off the agent as a Flyte task; stream the run link, then the briefing."""
    task = _resolve_task()
    auth = _auth_tuples(request)

    with _auth_ctx(auth):
        run = flyte.run(task, start=start, end=end, use_judge=bool(use_judge))

    url = _browser_url(run)
    link_html = f'<a href="{url}" target="_blank">View run on Flyte</a>' if url else ""
    yield "", (link_html or "Running…")  # show the link immediately

    with _auth_ctx(auth):
        run.wait()
        try:
            briefing = run.outputs()[0]
        except Exception as exc:  # failed run -> surface the error rather than crashing
            phase = getattr(getattr(run, "action", None), "phase", "?")
            yield f"Run {run.name} did not succeed (phase={phase}): {exc}", link_html
            return

    yield briefing, link_html


def create_demo():
    with gr.Blocks(title="Weather Along the Route") as demo:
        gr.Markdown(
            "# Weather Along the Route\n"
            "Enter two US locations (place names or ZIP codes); the agent samples "
            "the NWS forecast along the driving route and writes a briefing."
        )
        with gr.Row():
            start = gr.Textbox(label="Start", value="San Diego, CA", scale=2)
            end = gr.Textbox(label="End", value="Las Vegas, NV", scale=2)
            submit = gr.Button("Get briefing", variant="primary", scale=1)
        use_judge = gr.Checkbox(label="Run hallucination judge", value=True)
        run_link = gr.HTML()
        report = gr.Markdown(label="Briefing")

        inputs = [start, end, use_judge]
        submit.click(fn=run_query, inputs=inputs, outputs=[report, run_link])

        gr.Examples(
            examples=[
                ["San Diego, CA", "Las Vegas, NV"],
                ["92101", "Reno, NV"],
                ["Seattle, WA", "Portland, OR"],
            ],
            inputs=[start, end],
        )
    return demo


@serving_env.server
def app_server():
    """Launched by Flyte when the app is deployed to the cluster."""
    global _PASSTHROUGH
    if REQUIRES_AUTH:
        # Auth'd cluster: forward the caller's identity via passthrough. Hosted
        # clusters require an organization, which passthrough won't infer.
        flyte.init_passthrough(
#            endpoint=FLYTE_ENDPOINT or None,   # None -> use in-cluster injected endpoint
#            org=FLYTE_ORG or None,
            org="tryv2",
            project=FLYTE_PROJECT,
            domain=FLYTE_DOMAIN,
            insecure=FLYTE_INSECURE,
        )
        _PASSTHROUGH = True
    else:
        # No-auth devbox: run as the app's own in-cluster identity. This sets up
        # the cluster transport (incl. plaintext) on its own — no endpoint needed.
        flyte.init_in_cluster(project=FLYTE_PROJECT, domain=FLYTE_DOMAIN)
    create_demo().launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    # Local run: in-process task (RUN_MODE=local) or trigger the devbox (remote).
    if RUN_MODE != "local":
        flyte.init_from_config()
    create_demo().launch()
