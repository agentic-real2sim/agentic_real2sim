#!/usr/bin/env python3
"""
droid_sample_episodes.py  (v4)
==============================

Build a diverse candidate pool of DROID episodes and — optionally — call an
LLM to automatically prune it to a small maximally-diverse subset for a
real-to-sim pipeline.

Joins two small JSON files from the `KarlP/droid` HuggingFace repo:

  * episode_id_to_path.json          (~7 MB)   -- the EPISODE UNIVERSE
        episode_id -> "<LAB>/<success|failure>/<DATE>/<TIMESTAMP_FOLDER>"
        (contains BOTH successes and failures)
  * droid_language_annotations.json  (~21 MB)  -- LANGUAGE LABELS
        episode_id -> {"language_instruction1/2/3": str}
        (~75k SUCCESS episodes; failures are unlabeled -> empty instructions)

Both are keyed by an id like "IRIS+7dfa2da3+2023-04-25-11h-42m-28s" =
"<LAB>+<collector/setup hash>+<datetime>".

IMPORTANT about granularity (this is what makes the diversity numbers behave):
  * <LAB>  = institution. There are 13 of them.
  * <hash> = a DATA-COLLECTOR / setup id, NOT a scene. There are only ~50-66
             distinct lab+hash values total. A single collector recorded MANY
             scenes over the 12-month collection, so lab+hash is far coarser
             than DROID's 564 "scenes". The 564 scene ids are a separate
             annotation that is NOT present in these two files.
  * To approximate scenes we use DROID's protocol: a collector works one scene
    at a time (~100 trajectories / ~20 min) before moving on, so each
    collector's episodes cluster in time into bursts. We split each collector's
    timeline wherever there's a gap > --session-gap-min minutes and call each
    burst a "session" (~= one scene). This is a heuristic proxy, not ground
    truth; tune --session-gap-min (smaller -> more, finer sessions).

Fields per episode: episode_id, lab, status, collector_group (lab+hash),
session_group (scene proxy), date, instructions, path.

De-dup keys (--dedup):
  session+instruction (default) one per (scene-proxy, task). Keeps different
                      tasks in a scene AND the same task across scenes; drops
                      in-scene exact-instruction repeats. Best task+scene spread.
  collector+instruction         one per (collector, task) -- coarse (~66 groups).
  instruction                   one per task across the whole dataset.
  session                       one per scene-proxy (ignores task).
  none                          no de-duplication.

LLM diversity pruning (--api-key):
  After building the candidate pool, optionally call an LLM to select
  --n-select episodes that maximize diversity of skills, objects, and scenes.

  --api-key anthropic   uses ANTHROPIC_API_KEY env var  (claude-sonnet-4-6)
  --api-key openai      uses OPENAI_API_AR2S_KEY env var      (gpt-5.5)
  --api-key deepseek    uses DEEPSEEK_API_KEY env var    (deepseek-v4-flash)
  --api-key none        skip LLM call (default); write candidate pool only

  If the chosen provider's key is missing the script falls back to --api-key none
  and writes droid_episode_pool.json as usual.

  LLM output is written to --out-selected
  (default: outputs/droid_sample_episodes/droid_selected_<timestamp>.json).

Usage:
    python droid_sample_episodes.py
    python droid_sample_episodes.py --x 1000 --status success --session-gap-min 6
    python droid_sample_episodes.py --dedup instruction --seed 7 --out pool.json
    python droid_sample_episodes.py --api-key anthropic --n-select 20
    python droid_sample_episodes.py --api-key openai --n-select 10 --out-selected selected.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from huggingface_hub import hf_hub_download

HF_REPO = "KarlP/droid"
HF_FILES = {
    "annotations": "droid_language_annotations.json",
    "id2path": "episode_id_to_path.json",
}
HF_RESOLVE = "https://huggingface.co/{repo}/resolve/main/{name}?download=true"
RAW_GCS_PREFIX = "gs://gresearch/robotics/droid_raw/1.0.1/"
TS_FMT = "%Y-%m-%d-%Hh-%Mm-%Ss"
DEFAULT_X = 1200            # ~95-110k tokens at ~1200 records; ~half of 200k ctx
DEFAULT_GAP_MIN = 8         # timestamp gap that starts a new "session"/scene
DEFAULT_N_SELECT = 20       # number of episodes the LLM should select
DEFAULT_SELECTED_OUTPUT_DIR = Path("outputs/droid_sample_episodes")

# LLM provider config: (env-var, api-url, model-name)
LLM_PROVIDERS: dict[str, tuple[str, str, str]] = {
    "anthropic": (
        "ANTHROPIC_API_KEY",
        "https://api.anthropic.com/v1/messages",
        "claude-sonnet-4-6",
    ),
    "openai": (
        "OPENAI_API_AR2S_KEY",
        "https://api.openai.com/v1/chat/completions",
        "gpt-5.5",
    ),
    "deepseek": (
        "DEEPSEEK_API_KEY",
        "https://api.deepseek.com/v1/chat/completions",
        "deepseek-v4-flash",
    ),
}


def default_selected_output_path() -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return str(DEFAULT_SELECTED_OUTPUT_DIR / f"droid_selected_{timestamp}.json")


# --------------------------------------------------------------------------- #
# Download                                                                    #
# --------------------------------------------------------------------------- #
def fetch_input(kind: str, explicit_path: str | None, cache_dir: Path) -> Path:
    if explicit_path:
        p = Path(explicit_path).expanduser()
        if not p.exists():
            sys.exit(f"--{kind}: file not found: {p}")
        return p
    name = HF_FILES[kind]
    try:
        return Path(hf_hub_download(repo_id=HF_REPO, filename=name, repo_type="model"))
    except Exception as e:  # noqa: BLE001
        print(f"[info] huggingface_hub failed ({e}); using urllib.")
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / name
    if not dest.exists():
        print(f"[info] downloading {name} -> {dest}")
        urllib.request.urlretrieve(HF_RESOLVE.format(repo=HF_REPO, name=name), dest)  # noqa: S310
    return dest


# --------------------------------------------------------------------------- #
# Records                                                                     #
# --------------------------------------------------------------------------- #
_norm_re = re.compile(r"[^a-z0-9 ]+")


def normalize_instruction(text: str) -> str:
    return _norm_re.sub("", text.lower()).strip()


def instructions_from(ann: dict | None) -> list[str]:
    if not ann:
        return []
    out: list[str] = []
    for k in ("language_instruction1", "language_instruction2", "language_instruction3"):
        v = (ann.get(k) or "").strip()
        if v and v.lower() != "do something" and v not in out:
            out.append(v)
    return out


def parse_dt(episode_id: str):
    try:
        return datetime.strptime(episode_id.split("+")[-1], TS_FMT)
    except (ValueError, IndexError):
        return None


def build_records(id2path: dict, annotations: dict, statuses: set[str]) -> list[dict]:
    records: list[dict] = []
    for episode_id, path in id2path.items():
        parts = path.split("/")
        status = parts[1] if len(parts) > 1 else "unknown"
        if status not in statuses:
            continue
        date = parts[2] if len(parts) > 2 else ""
        idp = episode_id.split("+")
        lab = idp[0] if idp else episode_id
        hsh = idp[1] if len(idp) > 1 else ""
        records.append(
            {
                "episode_id": episode_id,
                "lab": lab,
                "status": status,
                "collector_group": f"{lab}+{hsh}",
                "session_group": "",  # filled by assign_sessions()
                "date": date,
                "instructions": instructions_from(annotations.get(episode_id)),
                "path": path,
                "_dt": parse_dt(episode_id),
            }
        )
    return records


def assign_sessions(records: list[dict], gap_min: int) -> None:
    """Cluster each collector's episodes into time-contiguous sessions (~scenes)."""
    gap = gap_min * 60
    by_collector: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_collector[r["collector_group"]].append(r)

    for cg, recs in by_collector.items():
        dated = sorted([r for r in recs if r["_dt"]], key=lambda r: r["_dt"])
        undated = [r for r in recs if not r["_dt"]]
        idx, prev = -1, None
        for r in dated:
            if prev is None or (r["_dt"] - prev).total_seconds() > gap:
                idx += 1
            r["session_group"] = f"{cg}#s{idx:03d}"
            prev = r["_dt"]
        for j, r in enumerate(undated):  # no timestamp -> its own session
            r["session_group"] = f"{cg}#u{j:03d}"


# --------------------------------------------------------------------------- #
# De-dup                                                                      #
# --------------------------------------------------------------------------- #
def dedup_key(rec: dict, mode: str) -> tuple:
    instr = normalize_instruction(rec["instructions"][0]) if rec["instructions"] else ""
    return {
        "session+instruction": (rec["session_group"], instr),
        "collector+instruction": (rec["collector_group"], instr),
        "instruction": (instr,),
        "session": (rec["session_group"],),
        "none": (rec["episode_id"],),
    }[mode]


def deduplicate(records: list[dict], mode: str, rng: random.Random) -> list[dict]:
    if mode == "none":
        return list(records)
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        buckets[dedup_key(r, mode)].append(r)
    return [rng.choice(v) for v in buckets.values()]


# --------------------------------------------------------------------------- #
# LLM diversity selection                                                     #
# --------------------------------------------------------------------------- #
DIVERSITY_SYSTEM = (
    "You are helping me curate a maximally DIVERSE subset of DROID robot-manipulation\n"
    "episodes for a real-to-sim pipeline: each selected episode's real-world scene\n"
    "will be reconstructed in MuJoCo and its task replayed, so I want broad coverage\n"
    "of distinct skills, objects, and scenes — not many variants of the same thing."
)

DIVERSITY_PROMPT_TMPL = """\
I've attached a JSON file shaped like:
```
{{ "meta": {{...}},
  "episodes": [
    {{"episode_id": "...", "lab": "...", "status": "success|failure",
     "collector_group": "...", "session_group": "...", "date": "...",
     "instructions": ["...", "..."], "path": "..."}},
    ...
  ] }}
```

Notes on the data:
- An episode's `instructions` are up to three crowd-sourced paraphrases of the
  SAME task — treat them as one task, not three.
- `lab` is the collecting institution (13 total). `collector_group` is one data
  collector / robot setup (coarse, ~70 total). `session_group` is an approximate
  single physical SCENE (a ~20-minute collection burst); it's the best available
  scene id, so use it as the scene axis.
- `status` is success or failure; failure episodes have an empty `instructions`
  list (they were never language-labeled).

Select EXACTLY {n_select} episodes that together maximize diversity, prioritizing:
1. Manipulation skill / verb (pick, place, pour, open, close, stack, wipe, push,
   fold, rotate, insert, press, …) — cover as many distinct skills as possible.
2. Manipulated objects and object categories.
3. Scene and lab spread — avoid taking many picks from the same `session_group`,
   `collector_group`, or `lab`.

Rules:
- Judge similarity SEMANTICALLY, not by string overlap. "Move the bottle to the
  right" and "take the green bottle and move it to the right" are the same task;
  pick at most one. No two selected episodes should share the same skill+object
  combination.
- Copy each `path` VERBATIM from the input.
- Before finalizing, sanity-check that your set spans many different verbs and
  several different labs/scenes; revise if it clusters.

Return ONLY valid JSON, no prose before or after, in exactly this form:
```
{{
  "selected": [
    {{
      "path": "<verbatim path from input>",
      "task": "<short canonical task, or 'failure / no label'>",
      "why_distinct": "<one short phrase: the unique skill/object/scene it adds>"
    }}
  ]
}}
```

Here is the episode pool:

{pool_json}"""


def _sanitize_pool(pool: dict) -> dict:
    """
    Return a copy of the pool with instruction strings scrubbed of control
    characters (raw newlines, tabs, etc.) that would break JSON-inside-a-string
    embedding when the pool is pasted into the prompt.
    """
    _ctrl_re = re.compile(r"[\x00-\x1f\x7f]+")

    def scrub(v):
        if isinstance(v, str):
            return _ctrl_re.sub(" ", v).strip()
        if isinstance(v, list):
            return [scrub(i) for i in v]
        if isinstance(v, dict):
            return {k: scrub(val) for k, val in v.items()}
        return v

    return scrub(pool)


def _build_prompt(pool: dict, n_select: int) -> str:
    # ensure_ascii=True so every non-ASCII / control char is \uXXXX-escaped;
    # this prevents the embedded JSON from breaking the outer HTTP JSON body.
    pool_json = json.dumps(_sanitize_pool(pool), ensure_ascii=True, indent=2)
    return DIVERSITY_PROMPT_TMPL.format(n_select=n_select, pool_json=pool_json)


def _http_post(url: str, headers: dict, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:  # noqa: S310
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        # Read the response body so we can show the actual API error message.
        raw = exc.read().decode(errors="replace")
        try:
            detail = json.loads(raw)
            msg = detail.get("error", {})
            if isinstance(msg, dict):
                msg = msg.get("message", raw)
        except (json.JSONDecodeError, AttributeError):
            msg = raw[:300] or "(empty response body)"
        raise RuntimeError(f"HTTP {exc.code} from API: {msg}") from exc


def call_anthropic(api_key: str, pool: dict, n_select: int) -> dict:
    prompt = _build_prompt(pool, n_select)
    body = {
        "model": LLM_PROVIDERS["anthropic"][2],
        "max_tokens": 4096,
        "system": DIVERSITY_SYSTEM,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    resp = _http_post(LLM_PROVIDERS["anthropic"][1], headers, body)
    text = resp["content"][0]["text"]
    return _parse_json_response(text)


def call_openai_compat(provider: str, api_key: str, pool: dict, n_select: int) -> dict:
    """Shared caller for OpenAI-compatible APIs (OpenAI, DeepSeek)."""
    prompt = _build_prompt(pool, n_select)
    _, url, model = LLM_PROVIDERS[provider]
    body = {
        "model": model,
        "max_tokens": 4096,
        "messages": [
            {"role": "system", "content": DIVERSITY_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    resp = _http_post(url, headers, body)
    text = resp["choices"][0]["message"]["content"]
    return _parse_json_response(text)


def _parse_json_response(text: str) -> dict:
    """Strip optional markdown fences and parse JSON from LLM response."""
    text = text.strip()
    # Strip ```json ... ``` or ``` ... ``` fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first line (``` or ```json) and last line (```)
        inner = lines[1:] if lines[-1].strip() == "```" else lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner).strip()
    return json.loads(text)


def run_llm_selection(provider: str, pool: dict, n_select: int) -> dict | None:
    """
    Call the chosen LLM provider. Returns parsed JSON dict on success,
    or None if the API key is missing or the call fails.
    """
    env_var, _, model = LLM_PROVIDERS[provider]
    api_key = os.environ.get(env_var, "").strip()
    if not api_key:
        print(f"[warn] {env_var} not set; falling back to pool-only mode.")
        return None

    print(f"[info] calling {provider} ({model}) to select {n_select} diverse episodes …")
    try:
        if provider == "anthropic":
            return call_anthropic(api_key, pool, n_select)
        else:
            return call_openai_compat(provider, api_key, pool, n_select)
    except (urllib.error.URLError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
        # RuntimeError is raised by _http_post with the decoded API error body;
        # URLError covers network-level failures (DNS, timeout, etc.).
        print(f"[warn] LLM call failed ({exc}); falling back to pool-only mode.")
        return None


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--annotations", help="droid_language_annotations.json (else auto-download).")
    ap.add_argument("--id2path", help="episode_id_to_path.json (else auto-download).")
    ap.add_argument("--cache-dir", default="./droid_meta")
    ap.add_argument("--x", type=int, default=DEFAULT_X, help=f"Episodes to sample (default {DEFAULT_X}).")
    ap.add_argument("--dedup",
                    choices=["session+instruction", "collector+instruction", "instruction", "session", "none"],
                    default="session+instruction")
    ap.add_argument("--status", choices=["both", "success", "failure"], default="both")
    ap.add_argument("--session-gap-min", type=int, default=DEFAULT_GAP_MIN,
                    help=f"Minutes-gap that starts a new session/scene (default {DEFAULT_GAP_MIN}).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="droid_episode_pool.json",
                    help="Output path for the candidate pool JSON (default: droid_episode_pool.json).")
    # --- LLM selection args ---
    ap.add_argument(
        "--api-key",
        choices=["anthropic", "openai", "deepseek", "none"],
        default="none",
        help=(
            "LLM provider to use for automatic diversity selection. "
            "Reads ANTHROPIC_API_KEY / OPENAI_API_AR2S_KEY / DEEPSEEK_API_KEY from env. "
            "Falls back to 'none' (pool-only) if the key is missing. (default: none)"
        ),
    )
    ap.add_argument(
        "--n-select",
        type=int,
        default=DEFAULT_N_SELECT,
        help=f"Number of diverse episodes the LLM should select (default {DEFAULT_N_SELECT}).",
    )
    ap.add_argument(
        "--out-selected",
        default=default_selected_output_path(),
        help=(
            "Output path for the LLM-selected episodes JSON "
            "(default: outputs/droid_sample_episodes/droid_selected_<timestamp>.json)."
        ),
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cache_dir = Path(args.cache_dir)
    with open(fetch_input("id2path", args.id2path, cache_dir)) as f:
        id2path = json.load(f)
    with open(fetch_input("annotations", args.annotations, cache_dir)) as f:
        annotations = json.load(f)
    print(f"[info] {len(id2path)} episodes in universe, {len(annotations)} language-annotated.")

    statuses = {"success", "failure"} if args.status == "both" else {args.status}
    records = build_records(id2path, annotations, statuses)
    assign_sessions(records, args.session_gap_min)

    n_coll = len({r["collector_group"] for r in records})
    n_sess = len({r["session_group"] for r in records})
    n_lang = sum(1 for r in records if r["instructions"])
    print(f"[info] {len(records)} episodes ({args.status}); {n_lang} with language. "
          f"Universe granularity: {n_coll} collector-groups -> {n_sess} session-groups "
          f"(gap={args.session_gap_min}min).")

    deduped = deduplicate(records, args.dedup, rng)
    print(f"[info] {len(deduped)} episodes after dedup ('{args.dedup}').")

    if args.x >= len(deduped):
        print(f"[info] requested x={args.x} >= available {len(deduped)}; using all.")
        sampled = list(deduped)
    else:
        sampled = rng.sample(deduped, args.x)
    rng.shuffle(sampled)

    for r in sampled:  # drop the private datetime before serializing
        r.pop("_dt", None)

    by_status: dict[str, int] = defaultdict(int)
    for r in sampled:
        by_status[r["status"]] += 1
    meta = {
        "source_repo": HF_REPO,
        "raw_gcs_prefix": RAW_GCS_PREFIX,  # full raw path = prefix + record["path"]
        "n_requested": args.x,
        "n_returned": len(sampled),
        "dedup": args.dedup,
        "status": args.status,
        "session_gap_min": args.session_gap_min,
        "seed": args.seed,
        "by_status": dict(by_status),
        "n_unique_labs": len({r["lab"] for r in sampled}),
        "n_unique_collectors": len({r["collector_group"] for r in sampled}),
        "n_unique_sessions": len({r["session_group"] for r in sampled}),
        "note": "lab=institution(13); collector_group=~66 setups; session_group=time-gap scene proxy, not an official scene id.",
    }

    with open(args.out, "w") as f:
        f.write("{\n")
        f.write(f'  "meta": {json.dumps(meta)},\n')
        f.write('  "episodes": [\n')
        for i, r in enumerate(sampled):
            comma = "," if i < len(sampled) - 1 else ""
            f.write("    " + json.dumps(r, ensure_ascii=False) + comma + "\n")
        f.write("  ]\n}\n")

    print(f"[done] wrote {len(sampled)} episodes -> {args.out}")
    print(f"[info] sample spans {meta['n_unique_labs']} labs, "
          f"{meta['n_unique_collectors']} collectors, {meta['n_unique_sessions']} sessions; "
          f"by_status={dict(by_status)}")

    # ----------------------------------------------------------------------- #
    # Optional LLM diversity selection                                         #
    # ----------------------------------------------------------------------- #
    if args.api_key == "none":
        return

    pool = {"meta": meta, "episodes": sampled}
    result = run_llm_selection(args.api_key, pool, args.n_select)
    if result is None:
        # Key missing or call failed — pool JSON is already written above.
        return

    selected = result.get("selected", [])
    print(f"[info] LLM returned {len(selected)} selected episodes.")

    out_selected = Path(args.out_selected)
    out_selected.parent.mkdir(parents=True, exist_ok=True)
    with open(out_selected, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[done] wrote {len(selected)} selected episodes -> {out_selected}")


if __name__ == "__main__":
    main()
