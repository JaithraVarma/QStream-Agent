"""Request and locally evaluate up to three Kimi K3 decode configurations."""
import argparse
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
RUN = HERE / "decode-search"
ROUNDS = 3
HIGHER_CAP = 16384
MODEL = "kimi-k3.1-pro:standard"
CATALOG = "https://api.inco.ai/v1/models"
ENDPOINT = "https://api.inco.ai/v1/chat/completions"
OPTIONS = {
    "n16-k32-w2-s1": (16, 32, 2, 1),
    "n32-k32-w2-s1": (32, 32, 2, 1),
    "n32-k64-w2-s1": (32, 64, 2, 1),
    "n32-k128-w4-s1": (32, 128, 4, 1),
    "n32-k64-w4-s2": (32, 64, 4, 2),
    "n64-k64-w4-s2": (64, 64, 4, 2),
    "n32-k32-w4-s2": (32, 32, 4, 2),
}
FIXED = (32, 64, 4, 1)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def evidence(report, baseline, trial, wins_key, projection=None):
    if report["status"] != "PASS":
        raise ValueError("Expected a completed benchmark")
    timings = report["medians_ms_per_step"]
    return dict(
        baseline_wall_ms=round(timings[baseline]["wall_ms"], 4),
        trial_wall_ms=round(timings[trial]["wall_ms"], 4),
        paired_wall_speedup=round(report["median_paired_speedup"]["wall_ms"], 4),
        wins=report[wins_key],
        pairs=report["pairs"],
        max_logit_relative_l2=max(row["relative_l2"] for row in report["correctness"]),
        compiler=report["projection"][projection or trial],
    )


def attempt_dir(round_number):
    retry = RUN / "round-1-retry"
    if round_number == 1 and retry.exists():
        return retry
    return RUN / f"round-{round_number}"


def retry_dir(round_number):
    if round_number != 1 or attempt_dir(1) != RUN / "round-1":
        raise ValueError("The one-time recovery is for the original round-one request")
    api = read_json(RUN / "round-1/api.json")
    if api["status"] != "http_error" or api["http_status"] != 404:
        raise ValueError("Recovery requires a recorded HTTP 404")
    return RUN / "round-1-retry"


def verify_model_access(key):
    request = urllib.request.Request(CATALOG, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=30) as response:
        models = json.load(response)["data"]
    if not any(entry["id"] == MODEL for entry in models):
        raise ValueError(f"{MODEL} is absent from the authenticated model catalog")


def make_request(round_number):
    manual = read_json(HERE / "tile-width/summary.json")
    fixed = read_json(RUN / "fixed.json")
    if any(manual[key] != fixed[key] for key in
           ("gpu", "kernel_sha256", "quantized_sha256", "input_sha256", "cache_capacity")):
        raise ValueError("Manual and fixed benchmark fingerprints differ")
    history = []
    available = dict(OPTIONS)
    for index in range(1, round_number):
        previous = attempt_dir(index)
        api = read_json(previous / "api.json")
        if api["status"] == "invalid_proposal":
            history.append(dict(round=index, status="invalid_proposal"))
            continue
        if api["status"] != "accepted":
            raise ValueError(f"Inspect round {index} request status before proceeding")
        proposal = read_json(previous / "proposal.json")
        config_id = proposal["config_id"]
        del available[config_id]
        outcome = read_json(previous / "evaluation.json")
        feedback = dict(round=index, config_id=config_id, status=outcome["status"])
        if outcome["status"] == "PASS":
            if outcome["configurations"]["challenger"] != list(OPTIONS[config_id]):
                raise ValueError(f"Round {index} configuration does not match its proposal")
            feedback.update(evidence(outcome, "manual", "challenger", "challenger_wall_wins"))
        history.append(feedback)

    feedback = dict(
        original_vs_manual=evidence(manual, "bn64", "bn32", "narrower_wall_wins", projection="32"),
        manual_vs_fixed=evidence(fixed, "manual", "challenger", "challenger_wall_wins"),
        prior_rounds=history,
    )
    prompt = """Choose one launch configuration for an existing Triton 3.5.1 W4A16
kernel running in Qwen3-0.6B on an RTX 3060 Laptop GPU (SM 8.6, 30 SMs).
Optimize paired full-model CUDA Graph decode wall time for batch 1, prompt
length 128, and 16 fixed cached decode steps. The only changed operation is
the down projection in each of 28 layers (M=1, N=1024, K=3072, G=128).
Gate, up, prefill, packed format, and arithmetic stay fixed. The kernel
loads unique packed bytes, decodes signed INT4, scales in FP32, rounds to
FP16 before tl.dot, and accumulates in FP32. It uses a 16-row tile and
broadcasts one group scale per output channel for each aligned K tile.

Settings are (output tile N, reduction tile K, warps, stages). Original:
(64,32,4,1). Manual baseline: (32,32,4,1). Fixed comparison: (32,64,4,1).
One warmed, isolated Nsight capture of the original had 16 blocks,
81.5% no-eligible-warp cycles, 1.49 long-scoreboard cycles per issued
instruction, 98.88% L2 hit rate, and 0.49% of peak DRAM throughput.
That isolated cache regime may differ from end-to-end replay.
Choose a configuration whose tradeoff could beat the manual baseline.
Use measured feedback as evidence and describe the expected tradeoff.
Do not claim performance for untested settings.

Return exactly one plain JSON object with keys "config_id" and "hypothesis".
Use one allowed ID and a hypothesis of at most 400 characters.
Do not return code or Markdown.
"""
    prompt += "\nMeasured feedback:\n" + json.dumps(feedback, indent=2)
    prompt += "\nAllowed options:\n" + json.dumps(available, indent=2)
    return dict(model=MODEL, max_tokens=8192, stream=False,
                messages=[dict(role="user", content=prompt)]), available


def make_higher_cap_request(round_number):
    path = attempt_dir(round_number)
    receipt = read_json(path / "api.json")
    if (receipt.get("status"), receipt.get("finish_reason")) != ("invalid_proposal", "length"):
        raise ValueError("The higher-cap repeat requires a truncated original round")
    original = read_json(path / "request.json")
    expected, available = make_request(round_number)
    if original != expected:
        raise ValueError("The saved request differs from the current experiment")
    return {**original, "max_tokens": HIGHER_CAP}, available


def parse_proposal(raw, available):
    response = json.loads(raw)
    [choice] = response["choices"]
    if choice["finish_reason"] != "stop":
        raise ValueError("The response did not finish")
    result = json.loads(choice["message"]["content"])
    if not isinstance(result, dict) or set(result) != {"config_id", "hypothesis"}:
        raise ValueError("Expected exactly config_id and hypothesis")
    config, hypothesis = result["config_id"], result["hypothesis"]
    if not isinstance(config, str) or config not in available:
        raise ValueError("Configuration is outside the remaining options")
    if not isinstance(hypothesis, str) or not 1 <= len(hypothesis.strip()) <= 400:
        raise ValueError("Expected a short hypothesis")
    return result


def submit(path, payload, available, key):
    path.mkdir()
    body = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    (path / "request.json").write_bytes(body)
    write_json(path / "api.json", {"status": "started"})
    request = urllib.request.Request(
        ENDPOINT, data=body, method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    start = time.perf_counter()
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=180) as response:
            request_id = response.headers.get("x-request-id")
            raw = response.read()
    except urllib.error.HTTPError as exc:
        receipt = dict(status="http_error", http_status=exc.code,
                       request_id=exc.headers.get("x-request-id") if exc.headers else None)
        try:
            code = json.loads(exc.read(16384))["error"]["code"]
            if isinstance(code, str) and re.fullmatch(r"[a-z0-9_]{1,64}", code):
                receipt["error_code"] = code
        except (OSError, ValueError, KeyError, TypeError):
            pass
    except OSError:
        receipt = dict(status="outcome_unknown")
    else:
        receipt = dict(status="invalid_proposal", request_id=request_id,
                       client_seconds=time.perf_counter() - start)
        (path / "response.json").write_bytes(raw)
        try:
            response = json.loads(raw)
            receipt.update(usage=response.get("usage"),
                           finish_reason=response["choices"][0].get("finish_reason"))
            proposal = parse_proposal(raw, available)
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            pass
        else:
            write_json(path / "proposal.json", proposal)
            receipt["status"] = "accepted"
        usage = receipt.get("usage")
        tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        if type(tokens) is int and receipt["client_seconds"] > 0:
            receipt["completion_tokens_per_client_second"] = tokens / receipt["client_seconds"]
    receipt.setdefault("client_seconds", time.perf_counter() - start)
    write_json(path / "api.json", receipt)
    print(json.dumps(receipt, indent=2))


def evaluate(output, config):
    if os.environ.get("INCO_API_KEY"):
        raise ValueError("Unset INCO_API_KEY before evaluating locally")
    from .tile_width import MANUAL, run

    if output.exists():
        raise FileExistsError(output)
    try:
        run(output, {"manual": MANUAL, "challenger": config})
    except AssertionError:
        write_json(output, dict(status="assertion_failed",
                                configurations={"manual": MANUAL, "challenger": config}))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fixed", "preview", "request", "evaluate"))
    parser.add_argument("--round", type=int, choices=range(1, ROUNDS + 1), default=1)
    parser.add_argument("--retry-404", action="store_true", help="Recover round one after its saved HTTP 404")
    parser.add_argument("--higher-cap", action="store_true", help="Repeat a saved truncated round at 16,384 tokens")
    args = parser.parse_args()
    if args.retry_404 and args.action != "request":
        parser.error("--retry-404 applies only to request")
    if args.higher_cap and (args.round not in (2, 3) or args.action == "fixed" or args.retry_404):
        parser.error("--higher-cap applies only to preview, request, or evaluation for rounds two and three")
    if args.action == "fixed":
        evaluate(RUN / "fixed.json", FIXED)
        return
    if args.higher_cap:
        path = RUN / f"round-{args.round}-higher-cap"
        try:
            payload, available = make_higher_cap_request(args.round)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    else:
        path = attempt_dir(args.round)
        payload, available = make_request(args.round)
    if args.action == "evaluate":
        if read_json(path / "api.json")["status"] != "accepted":
            parser.error("This round has no accepted proposal")
        if read_json(path / "request.json") != payload:
            parser.error("The proposal's stored request differs from the current experiment")
        proposal = parse_proposal((path / "response.json").read_bytes(), available)
        if proposal != read_json(path / "proposal.json"):
            parser.error("The proposal differs from the stored Inco response")
        evaluate(path / "evaluation.json", OPTIONS[proposal["config_id"]])
        return
    if args.action == "preview":
        print(json.dumps(payload, indent=2))
        return
    key = os.environ.get("INCO_API_KEY", "").strip()
    if not key:
        parser.error("Set INCO_API_KEY in the current WSL terminal")
    if args.retry_404:
        try:
            path = retry_dir(args.round)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    if path.exists():
        parser.error("Round already exists. No request was sent.")
    try:
        verify_model_access(key)
    except urllib.error.HTTPError as exc:
        parser.error(f"Authenticated model catalog returned HTTP {exc.code}. No completion request was sent.")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        detail = str(exc) if type(exc) is ValueError else type(exc).__name__
        parser.error(f"Model catalog check failed ({detail}). No completion request was sent.")
    submit(path, payload, available, key)


if __name__ == "__main__":
    main()
