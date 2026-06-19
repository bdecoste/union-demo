"""
Serve weather_route_agent as a Flyte app (web UI) on the devbox.

This app is a thin Gradio front end: the user enters two locations and the
handler triggers the already-deployed `weather_route.main` task on the devbox,
then shows the briefing. The heavy lifting (and the Anthropic secret) stays in
the deployed task, so this app needs no secret of its own.

1) Deploy the workflow tasks first so this app has something to call:
       flyte deploy weather_route_agent.py data_env     # pulls llm_env via depends_on

2) Iterate / serve on the devbox (development serving):
       python weather_app.py                 # calls flyte.serve(), prints the URL

3) Deploy it as a managed app (production):
       flyte deploy weather_app.py serving_env
       flyte get app                         # list apps + find the assigned URL
       flyte get app weather-route-app       # details for this one
       flyte update app weather-route-app --deactivate   # stop serving
"""

import flyte
import flyte.app

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
    "flyte", "gradio"
)

serving_env = flyte.app.AppEnvironment(
    name="weather-route-app",
    image=image,
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    requires_auth=False,   # reachable without login on the devbox; True to require auth
    port=7860,
)


def make_briefing(start: str, end: str, use_judge: bool) -> str:
    """Trigger the deployed weather_route.main task and return its output."""
    main = flyte.remote.Task.get("weather_route.main", auto_version="latest")
    run = flyte.run(main, start=start, end=end, use_judge=use_judge)
    run.wait()
    try:
        return run.outputs()[0]
    except Exception as exc:  # failed run -> show the error in the UI instead of crashing
        return f"Run {run.name} did not succeed (phase={run.action.phase}): {exc}\n{run.url}"


@serving_env.server
def server():
    import os
    import gradio as gr

#    if os.getenv("KUBERNETES_SERVICE_HOST"):
#        flyte.init_in_cluster(
#            project="flytesnacks",
#            domain="development",
#            endpoint="dns:///flyte-binary-http.flyte.svc.cluster.local:8090",
#            insecure=True,   # devbox is plaintext; use http://, not https://
#        )
#    else:
#        flyte.init_from_config()  # local `flyte serve` from your laptop

    endpoint = os.getenv("FLYTE_ENDPOINT", "dns:///flyte-binary-grpc.flyte:8089")  # verify name/port
    flyte.init_passthrough(
        endpoint=endpoint,
        project=os.getenv("FLYTE_INTERNAL_EXECUTION_PROJECT", "flytesnacks"),
        domain=os.getenv("FLYTE_INTERNAL_EXECUTION_DOMAIN", "development"),
        insecure=True,   # devbox is insecure; drop/flip if your hosted cluster uses TLS
    )

    gr.Interface(
        fn=make_briefing,
        inputs=[
            gr.Textbox(label="Start", value="San Diego, CA"),
            gr.Textbox(label="End", value="Las Vegas, NV"),
            gr.Checkbox(label="Run hallucination judge", value=True),
        ],
        outputs=gr.Textbox(label="Weather briefing", lines=20),
        title="Weather Along the Route",
        description="Enter two US locations (place names or ZIP codes).",
        flagging_mode="never",
    ).launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    flyte.init_from_config()
    app = flyte.serve(serving_env)
    print(f"App URL: {app.url}")
