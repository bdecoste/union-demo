"""
Weather-along-the-route agent, built with Flyte v2 (the `flyte` 2.x SDK).

Given two US locations (place names, addresses, or ZIP codes) it:
  1. Geocodes each endpoint            -> Open-Meteo (names) / Zippopotam (ZIPs)
  2. Computes the driving route        -> OSRM public server         (free, no key)
  3. Samples waypoints along the route -> MORE waypoints for LONGER trips
  4. Pulls the NWS forecast per point  -> api.weather.gov            (free, no key)
     ...fanned out in parallel, one Flyte action per waypoint.
  5. Asks Claude for a driver briefing -> length scales with route distance.
  6. Verifies the briefing             -> deterministic checks + LLM judge,
     flagging any claim not supported by the sampled NWS data (--no-judge to skip).

Why these extra APIs: the endpoints in the brief only return weather. To get a
*route* you need (a) geocoding to turn "San Diego, CA" / "92101" into lat,lon and
(b) a routing engine. api.weather.gov/points takes only `lat,lon`, so ZIPs and
place names are geocoded first (this also handles the /points/{zipcode} intent).
Geocoding uses Open-Meteo (place names) and Zippopotam.us (ZIPs) rather than
Nominatim, whose public server rejects datacenter / pod requests with HTTP 403.

Setup:
    pip install flyte anthropic requests
    flyte create config --endpoint localhost:30080 --project flytesnacks \
        --domain development --builder local --insecure        # devbox
    flyte create secret anthropic_api_key                      # paste key when prompted

Run (in-process, no cluster):
    python weather_route_agent.py --start "San Diego, CA" --end "Las Vegas, NV" --local

Run on the devbox / remote cluster:
    python weather_route_agent.py --start "92101" --end "Reno, NV"
    # or via the CLI:
    flyte run --local weather_route_agent.py main --start "San Diego, CA" --end "Phoenix, AZ"

Note: NWS only covers the US and its territories; points outside it return no data
(the agent reports that gracefully rather than failing the run).
"""

from __future__ import annotations

import argparse
import asyncio
import re
from dataclasses import dataclass, field

from kubernetes.client import V1PodSpec, V1Container

import flyte
import requests

# api.weather.gov requires a descriptive User-Agent; replace the contact with
# your own before any real use. (Open-Meteo / Zippopotam / OSRM don't need it.)
HEADERS = {"User-Agent": "flyte-weather-route-agent (contact@example.com)"}

# US state abbreviation -> full name, used to disambiguate place-name geocoding
# (Open-Meteo matches on the city name only and ignores a trailing ", CA").
US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}

DEFAULT_MODEL = "claude-sonnet-4-6"  # swap for "claude-opus-4-8" or "claude-haiku-4-5-20251001"

# One shared image so both environments are image-compatible.
image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
    "requests", "anthropic", "kubernetes"
)

# LLM task environment: gets the Anthropic key injected as an env var.
# Defined first so data_env can depend on it.
llm_env = flyte.TaskEnvironment(
    name="weather_route_llm",
    image=image,
    secrets=[flyte.Secret(key="anthropic_api_key", as_env_var="ANTHROPIC_API_KEY")],
)

# Data tasks: network calls, no secret needed. `main` lives here and calls
# `summarize` in llm_env, so we declare that cross-environment dependency.
data_env = flyte.TaskEnvironment(
    name="weather_route",
    image=image,
    depends_on=[llm_env],
    pod_template=flyte.PodTemplate(
        primary_container_name="primary",
        labels={"app": "weather-route", "component": "data"},
        annotations={"owner": "bill"},
        pod_spec=V1PodSpec(containers=[V1Container(name="primary")]),   # <-- required
    )
)


# --------------------------------------------------------------------------- #
# Typed data passed between tasks
# --------------------------------------------------------------------------- #
@dataclass
class Coordinate:
    name: str
    latitude: float
    longitude: float


@dataclass
class RouteInfo:
    distance_miles: float
    duration_hours: float
    waypoints: list[Coordinate]


@dataclass
class WeatherPoint:
    label: str
    latitude: float
    longitude: float
    period_name: str
    short_forecast: str
    detailed_forecast: str
    temperature: int
    temperature_unit: str
    wind: str
    error: str = ""


@dataclass
class VerificationReport:
    """Faithfulness check on the briefing against the NWS data we supplied."""
    flagged: bool
    issues: list[str] = field(default_factory=list)        # deterministic findings
    judge_verdict: str = ""                                # "" if judge not run
    judge_unsupported: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Pure helpers (run inside tasks, not orchestrated themselves)
# --------------------------------------------------------------------------- #
def _sample_count(distance_miles: float, max_samples: int) -> int:
    """Roughly one extra check per 75 miles, on top of the two endpoints."""
    n = int(distance_miles // 75) + 2
    return max(2, min(max_samples, n))


def _sample_waypoints(
    coords: list[list[float]], n: int, start: Coordinate, end: Coordinate
) -> list[Coordinate]:
    """Pick n evenly spaced points along the OSRM geometry ([lon, lat] pairs)."""
    if n <= 2 or len(coords) <= 2:
        return [start, end]
    idxs = [round(i * (len(coords) - 1) / (n - 1)) for i in range(n)]
    points: list[Coordinate] = []
    for j, idx in enumerate(idxs):
        if j == 0:
            points.append(start)
        elif j == n - 1:
            points.append(end)
        else:
            lon, lat = coords[idx]
            points.append(Coordinate(name=f"En-route stop {j}", latitude=lat, longitude=lon))
    return points


def _build_digest(weather: list["WeatherPoint"]) -> str:
    """Compact, factual digest of the sampled forecasts (the ground truth that
    both the briefing and the verifier are anchored to)."""
    lines = []
    for w in weather:
        if w.error:
            lines.append(
                f"- {w.label} ({w.latitude:.2f},{w.longitude:.2f}): no NWS data available"
            )
        else:
            lines.append(
                f"- {w.label}: {w.period_name} — {w.short_forecast}, "
                f"{w.temperature}°{w.temperature_unit}, wind {w.wind}. {w.detailed_forecast}"
            )
    return "\n".join(lines)


# Hazard words that are unambiguous enough to check deterministically. "wind"/
# "rain" are excluded: too common in forecasts to flag reliably.
_HAZARDS = ("snow", "ice", "sleet", "freezing", "fog", "thunder", "flood", "hail", "blizzard")


def _verify_against_digest(briefing: str, weather: list["WeatherPoint"]) -> list[str]:
    """High-precision deterministic checks: every number/hazard asserted in the
    briefing must trace back to data we actually supplied. Returns a list of
    issue strings (empty == nothing suspicious)."""
    issues: list[str] = []
    good = [w for w in weather if not w.error]
    known_temps = {w.temperature for w in good}

    # Temperatures: "72°F", "72 °", or "72 degrees". Allow ±1 for rounding.
    quoted = {int(t) for t in re.findall(r"(-?\d+)\s*(?:°|degrees?\b)", briefing, re.I)}
    for t in quoted:
        if known_temps and not any(abs(t - k) <= 1 for k in known_temps):
            issues.append(
                f"Temperature {t}° in briefing matches no sampled forecast "
                f"(sampled: {sorted(known_temps)})"
            )

    # Hazards: a hazard named in the briefing should appear in some forecast text.
    corpus = " ".join(
        (w.short_forecast + " " + w.detailed_forecast).lower() for w in good
    )
    low = briefing.lower()
    for hazard in _HAZARDS:
        if hazard in low and hazard not in corpus:
            issues.append(f"Briefing mentions '{hazard}' not present in any sampled forecast")

    # If every point lacked data, the briefing shouldn't be asserting conditions.
    if not good and quoted:
        issues.append("Briefing cites temperatures although no point returned NWS data")

    return issues


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #
def _geocode_zip(zipcode: str) -> Coordinate:
    """US ZIP -> Coordinate via Zippopotam.us (free, no key, server-friendly)."""
    resp = requests.get(f"https://api.zippopotam.us/us/{zipcode}", timeout=30)
    if resp.status_code == 404:
        raise ValueError(f"Unknown US ZIP code: {zipcode!r}")
    resp.raise_for_status()
    place = resp.json()["places"][0]
    return Coordinate(
        name=f'{place["place name"]}, {place["state abbreviation"]} {zipcode}',
        latitude=float(place["latitude"]),
        longitude=float(place["longitude"]),
    )


def _geocode_place(location: str) -> Coordinate:
    """Place name -> Coordinate via Open-Meteo geocoding (free, no key).

    Open-Meteo matches on the city name only, so we split off a trailing state
    (", CA" / ", California") and use it to pick the right result client-side.
    """
    city, _, region = location.partition(",")
    city, region = city.strip(), region.strip()

    resp = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 10, "language": "en", "format": "json"},
        timeout=30,
    )
    resp.raise_for_status()
    results = resp.json().get("results") or []
    if not results:
        raise ValueError(f"Could not geocode location: {location!r}")

    us = [r for r in results if r.get("country_code") == "US"] or results

    chosen = us[0]
    if region:
        want = US_STATES.get(region.upper(), region).lower()
        chosen = next(
            (r for r in us if (r.get("admin1") or "").lower() == want), us[0]
        )

    state_full = chosen.get("admin1", "")
    state_abbr = next((a for a, f in US_STATES.items() if f == state_full), state_full)
    label = f'{chosen["name"]}, {state_abbr}' if state_abbr else chosen["name"]
    return Coordinate(
        name=label,
        latitude=float(chosen["latitude"]),
        longitude=float(chosen["longitude"]),
    )


@data_env.task(
    pod_template=flyte.PodTemplate(
        primary_container_name="primary",
        labels={"app": "weather-route", "component": "geocode"},
        pod_spec=V1PodSpec(containers=[V1Container(name="primary")]),
    )
)
def geocode(location: str) -> Coordinate:
    """Place name / "City, ST" / ZIP -> Coordinate.

    Uses Zippopotam.us for ZIPs and Open-Meteo for place names. Both are free,
    keyless, and (unlike Nominatim's public server) tolerant of requests coming
    from datacenter / Kubernetes-pod egress IPs.
    """
    loc = location.strip()
    if re.fullmatch(r"\d{5}(-\d{4})?", loc):
        return _geocode_zip(loc[:5])
    return _geocode_place(loc)


@data_env.task(
    pod_template=flyte.PodTemplate(
        primary_container_name="primary",
        labels={"app": "weather-route", "component": "get_route"},
        pod_spec=V1PodSpec(containers=[V1Container(name="primary")]),
    )
)
def get_route(start: Coordinate, end: Coordinate, max_samples: int = 8) -> RouteInfo:
    """Driving route via the public OSRM server; samples waypoints by distance."""
    url = (
        "https://router.project-osrm.org/route/v1/driving/"
        f"{start.longitude},{start.latitude};{end.longitude},{end.latitude}"
    )
    resp = requests.get(
        url,
        params={"overview": "full", "geometries": "geojson"},
        headers=HEADERS,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "Ok" or not data.get("routes"):
        raise ValueError(f"No drivable route found between {start.name} and {end.name}")

    route = data["routes"][0]
    distance_miles = route["distance"] / 1609.344
    duration_hours = route["duration"] / 3600.0
    coords = route["geometry"]["coordinates"]  # list of [lon, lat]

    n = _sample_count(distance_miles, max_samples)
    waypoints = _sample_waypoints(coords, n, start, end)
    return RouteInfo(
        distance_miles=distance_miles,
        duration_hours=duration_hours,
        waypoints=waypoints,
    )


@data_env.task(
    pod_template=flyte.PodTemplate(
        primary_container_name="primary",
        labels={"app": "weather-route", "component": "get_weather"},
        pod_spec=V1PodSpec(containers=[V1Container(name="primary")]),
    )
)
def get_weather(point: Coordinate) -> WeatherPoint:
    """NWS forecast for one coordinate. Errors are captured, not raised, so a
    single bad point never sinks the whole run."""
    try:
        meta = requests.get(
            f"https://api.weather.gov/points/{point.latitude},{point.longitude}",
            headers=HEADERS,
            timeout=30,
        )
        meta.raise_for_status()
        props = meta.json()["properties"]

        loc = props.get("relativeLocation", {}).get("properties", {})
        city, state = loc.get("city"), loc.get("state")
        label = f"{city}, {state}" if city and state else point.name

        forecast = requests.get(props["forecast"], headers=HEADERS, timeout=30)
        forecast.raise_for_status()
        period = forecast.json()["properties"]["periods"][0]

        return WeatherPoint(
            label=label,
            latitude=point.latitude,
            longitude=point.longitude,
            period_name=period.get("name", ""),
            short_forecast=period.get("shortForecast", ""),
            detailed_forecast=period.get("detailedForecast", ""),
            temperature=period.get("temperature", 0),
            temperature_unit=period.get("temperatureUnit", "F"),
            wind=f'{period.get("windSpeed", "")} {period.get("windDirection", "")}'.strip(),
        )
    except Exception as exc:  # noqa: BLE001 - we deliberately degrade gracefully
        return WeatherPoint(
            label=point.name,
            latitude=point.latitude,
            longitude=point.longitude,
            period_name="",
            short_forecast="",
            detailed_forecast="",
            temperature=0,
            temperature_unit="F",
            wind="",
            error=str(exc),
        )


@flyte.trace
async def call_claude(model: str, max_tokens: int, prompt: str) -> str:
    """Traced LLM call: if the run hits a system failure, Flyte can replay from
    the last successful trace instead of re-running everything upstream."""
    import anthropic

    client = anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY from the env var
    msg = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in msg.content if block.type == "text")


@llm_env.task
async def summarize(
    start_name: str,
    end_name: str,
    route: RouteInfo,
    weather: list[WeatherPoint],
    model: str,
) -> str:
    """Turn the sampled forecasts into a driver briefing. Verbosity scales with
    distance via both the target word count and max_tokens."""
    miles = route.distance_miles
    target_words = int(min(900, max(120, miles * 1.2)))
    max_tokens = int(min(2000, max(400, target_words * 2)))

    digest = _build_digest(weather)

    prompt = f"""You are a travel-weather briefer writing for someone about to drive.

Trip: {start_name} -> {end_name}
Driving distance: {miles:.0f} miles (about {route.duration_hours:.1f} hours)

Forecasts sampled along the route, in travel order:
{digest}

Write a clear, friendly weather briefing for the driver. Guidance:
- Target about {target_words} words. A short hop should be brief and to the point;
  a long haul deserves a more detailed, segment-by-segment walkthrough.
- Flag anything that affects driving: rain, snow, ice, wind, fog, heat, and large
  temperature swings between segments.
- Describe how conditions change along the way, using the place names above.
- Finish with one practical takeaway (what to pack, when to leave, what to watch).
- Use only the data provided; do not invent forecasts. If some segments lack data,
  note that briefly rather than guessing."""

    return await call_claude(model=model, max_tokens=max_tokens, prompt=prompt)


# Cross-vendor option: using a different model family as the judge reduces the
# chance the writer and grader share the same blind spot. To use OpenAI instead
# of Claude here: add "openai" to the image's with_pip_packages, declare a
# Secret(key="openai_api_key", as_env_var="OPENAI_API_KEY") on llm_env, and swap
# the client below for `openai.AsyncOpenAI()`. IMPORTANT: create that secret
# (flyte create secret + flyte-binary rollout restart) BEFORE deploying — an
# env that requests a missing secret gets its pods denied by the webhook.
@flyte.trace
async def judge_faithfulness(model: str, digest: str, briefing: str) -> str:
    """LLM-as-judge constrained to the digest. Returns a JSON string:
    {"verdict": "...", "unsupported_claims": [...]}. The judge checks claims
    against the SOURCE only, never its own world knowledge."""
    import json

    import anthropic

    client = anthropic.AsyncAnthropic()
    prompt = f"""You are a strict fact-checker. You are given SOURCE weather data and a
BRIEFING written from it. Decompose the briefing into individual factual claims
(temperatures, conditions, hazards, place names, timing) and check each ONLY
against the SOURCE. Do not use outside knowledge. A claim is "unsupported" if the
SOURCE does not state or directly imply it. General travel advice (e.g. "bring a
jacket") is fine and not a factual claim.

SOURCE:
{digest}

BRIEFING:
{briefing}

Respond with ONLY a JSON object, no prose, no markdown fences:
{{"verdict": "supported" | "has_unsupported_claims",
  "unsupported_claims": ["<short quote or paraphrase>", ...]}}"""

    msg = await client.messages.create(
        model=model,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text").strip()
    # Be tolerant if the model wraps the JSON in fences anyway.
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        json.loads(text)  # validate; verify() re-parses
    except ValueError:
        return json.dumps({"verdict": "judge_parse_error", "unsupported_claims": [], "raw": text})
    return text


@llm_env.task
async def verify(
    weather: list[WeatherPoint],
    briefing: str,
    use_judge: bool = True,
    model: str = DEFAULT_MODEL,
) -> VerificationReport:
    """Flag possible hallucinations in the briefing.

    Two layers: (1) always-on deterministic checks against the supplied data,
    and (2) an optional LLM faithfulness judge for qualitative/fabricated claims
    the deterministic pass can't see."""
    import json

    issues = _verify_against_digest(briefing, weather)

    judge_verdict = ""
    judge_unsupported: list[str] = []
    if use_judge:
        digest = _build_digest(weather)
        raw = await judge_faithfulness(model=model, digest=digest, briefing=briefing)
        try:
            parsed = json.loads(raw)
            judge_verdict = parsed.get("verdict", "")
            judge_unsupported = list(parsed.get("unsupported_claims", []))
        except ValueError:
            judge_verdict = "judge_parse_error"

    flagged = bool(issues) or bool(judge_unsupported) or judge_verdict == "has_unsupported_claims"
    return VerificationReport(
        flagged=flagged,
        issues=issues,
        judge_verdict=judge_verdict,
        judge_unsupported=judge_unsupported,
    )


def _format_report(report: VerificationReport) -> str:
    """Render a short verification footer appended to the briefing."""
    if not report.flagged:
        note = "✓ Verification: no unsupported claims detected"
        if report.judge_verdict and report.judge_verdict != "supported":
            note += f" (judge: {report.judge_verdict})"
        return note

    lines = ["⚠️  Verification flagged possible hallucinations:"]
    for issue in report.issues:
        lines.append(f"  • [check] {issue}")
    for claim in report.judge_unsupported:
        lines.append(f"  • [judge] unsupported: {claim}")
    if report.judge_verdict == "judge_parse_error":
        lines.append("  • [judge] could not parse judge response; review manually")
    return "\n".join(lines)


@data_env.task
async def main(
    start: str, end: str, model: str = DEFAULT_MODEL, use_judge: bool = True
) -> str:
    # Geocode both endpoints in parallel.
    start_coord, end_coord = await asyncio.gather(geocode.aio(start), geocode.aio(end))

    # Route, then sample waypoints (count depends on distance).
    route = await get_route.aio(start_coord, end_coord)

    # Fan out: one parallel Flyte action per waypoint.
    weather = list(await asyncio.gather(*[get_weather.aio(p) for p in route.waypoints]))

    # Distance-aware LLM briefing.
    briefing = await summarize.aio(start, end, route, weather, model)

    # Flag anything in the briefing not supported by the sampled NWS data.
    report = await verify.aio(weather, briefing, use_judge, model)

    return briefing + "\n\n" + ("-" * 70) + "\n" + _format_report(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weather along a driving route.")
    parser.add_argument("--start", required=True, help='e.g. "San Diego, CA" or "92101"')
    parser.add_argument("--end", required=True, help='e.g. "Las Vegas, NV"')
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="skip the LLM faithfulness judge (deterministic checks still run)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="run in-process instead of on the configured cluster/devbox",
    )
    args = parser.parse_args()

    if args.local:
        flyte.init()
    else:
        flyte.init_from_config()

    run = flyte.run(
        main, start=args.start, end=args.end, model=args.model, use_judge=not args.no_judge
    )
    print(f"Run: {run.name}")
    if run.url:
        print(f"UI:  {run.url}")

    run.wait()  # block until the run reaches a terminal state

    # On a failed run there is no outputs.pb, so run.outputs() raises a 404-style
    # storage error that hides the real failure. Read the error info instead.
    try:
        outputs = run.outputs()
        print("\n" + "=" * 70 + "\n")
        print(outputs[0] if outputs else "(no output)")
    except Exception as exc:  # noqa: BLE001
        print(f"\nRun did not produce outputs (phase={run.action.phase}).")
        print(f"Reading outputs failed with: {exc}")
        try:
            details = flyte.remote.ActionDetails.get(run_name=run.name, name="a0")
            print("\nUnderlying error:\n", details.pb2.error_info)
        except Exception as exc2:  # noqa: BLE001
            print(f"Could not fetch error details ({exc2}). Open the UI: {run.url}")
