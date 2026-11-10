"""Validate the selected Kimi launch on unseen Qwen contexts and passages."""
import hashlib
import json
from math import exp
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from .all_layers import configure_mlp
from .benchmark import save_report, workload
from .decode_search import OPTIONS, read_json
from .graph_decode import checked_sequence
from .qwen_mlp import load_selected_kernel_module
from .smoke import load_model
from .telemetry import snapshot
from .tile_width import MANUAL, compare_workload, prepare_graphs


HERE = Path(__file__).resolve().parent
OUT = HERE / "heldout" / "summary.json"
LENGTHS = (64, 256, 512)
PREFIX = 32
TARGETS = 32
PASSAGES = {
    "navigation": """A harbor pilot boarded the ship before dawn. The tide was still falling,
so she asked the captain to keep close to the eastern buoy until the current weakened.
The chart showed shallow water on the other side of the channel. From the bridge they
could see the first fishing boats return, their lights reflected in the water. By sunrise
the ship was inside the breakwater and the crew began preparing the lines.""",
    "astronomy": """The telescope collected a series of short exposures instead of a single
long photograph. Small changes in the position of the stars showed how the camera moved
between frames. After aligning the images, the observer combined them to bring out the
faint tail of the comet. A separate dark frame helped remove a repeating pattern from the
sensor. The last image was sharper, though a thin cloud had reduced the brightness.""",
    "repair": """Before replacing the bicycle chain, Mira checked the wear on the rear
sprockets. The old chain had stretched enough that several teeth were beginning to
hook. A new chain might skip under pressure if she kept the worn cassette. She cleaned
the parts, compared the measurements with the service guide, and tested the gears on a
short hill near the shop. The final adjustment made the smallest sprocket shift cleanly.""",
    "music": """The quartet rehearsed the opening slowly, listening for the place where the
cello passed the melody to the viola. The players could agree on the written tempo, but
the phrase still sounded uneven when each person counted separately. They tried again
with the second violin leading the breath before the entrance. After a few repetitions
the transition sounded natural. The conductor then asked them to play the entire movement
without stopping.""",
    "ecology": """Rain collected in the shallow basins along the edge of the field. In early
spring the basins were deep enough for frogs to lay eggs, but the water often disappeared
before summer. The survey team marked the waterline every week and recorded when tadpoles
first appeared. They also measured the shade from nearby trees. The shaded basins held
water longer, although their temperatures rose more slowly in the morning.""",
    "archive": """The librarian found two copies of the letter in different folders. One
carried a date in pencil and the other included a short note at the bottom. Instead of
choosing one as the original, she cataloged both and described the differences. A later
ledger entry placed the sender in another city on the penciled date. That small detail
changed which of the two copies researchers consulted first.""",
    "cooking": """The baker mixed a small portion of the flour with hot water and left it
to cool before adding it to the dough. She weighed each loaf before baking so that
differences in the oven would be easier to compare. The loaf on the lower shelf browned
first, while the one on the upper shelf stayed pale. After the bread cooled, she cut
each loaf through the center and compared the texture of the crumb.""",
    "water": """After several days of cold weather, the town crew noticed a drop in water
pressure on two streets. They closed one valve at a time and watched the reading at a
hydrant. The pressure recovered when they isolated a short branch near the school. A
small leak had softened the ground below the pavement without leaving a puddle. The crew
marked the location and scheduled the repair before the morning buses arrived.""",
}


def quality_inputs(tokenizer):
    rows = {}
    for name, text in PASSAGES.items():
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < PREFIX + TARGETS:
            raise ValueError(f"{name} has only {len(ids)} tokens")
        tokens = torch.tensor([ids[:PREFIX + TARGETS]], dtype=torch.long)
        rows[name] = (tokens.cuda(), hashlib.sha256(tokens.numpy().tobytes()).hexdigest())
    return rows


def nll(logits, labels):
    return F.cross_entropy(logits.float(), labels, reduction="none")


def score_stock(model, inputs):
    scores = {}
    for name, (tokens, _) in inputs.items():
        logits = model(input_ids=tokens[:, :-1], use_cache=False, logits_to_keep=TARGETS).logits[0]
        scores[name] = nll(logits, tokens[0, PREFIX:]).mean().item()
    return scores


def score_packed(model, module, configs, inputs, stock):
    first, _ = next(iter(inputs.values()))
    cache, positions, pointers, stage, methods, _ = prepare_graphs(
        model, module, configs, first[:, :PREFIX].contiguous(),
        first[:, PREFIX:-1].split(1, dim=1)
    )
    rows = []
    for name, (tokens, digest) in inputs.items():
        prefix = tokens[:, :PREFIX].contiguous()
        continuation = tokens[:, PREFIX:-1].split(1, dim=1)
        labels = tokens[0, PREFIX:]
        outputs = {
            mode: torch.cat(checked_sequence(model, cache, prefix, continuation, positions, positions,
                                            stage, replay, pointers), dim=1)[0]
            for mode, replay in methods.items()
        }
        manual, kimi = outputs["manual"], outputs["challenger"]
        delta = kimi.float() - manual.float()
        relative = delta.norm(dim=-1) / manual.float().norm(dim=-1).clamp_min(1e-12)
        rows.append(dict(
            passage=name, token_sha256=digest, targets=TARGETS,
            max_relative_l2=relative.max().item(),
            top1_matches=(kimi.argmax(-1) == manual.argmax(-1)).sum().item(),
            mean_nll=dict(fp16=stock[name], manual=nll(manual, labels).mean().item(),
                          kimi=nll(kimi, labels).mean().item()),
        ))
        assert torch.isfinite(manual).all() and torch.isfinite(kimi).all()
    count = len(rows) * TARGETS
    means = {mode: sum(row["mean_nll"][mode] for row in rows) / len(rows)
             for mode in ("fp16", "manual", "kimi")}
    max_l2 = max(row["max_relative_l2"] for row in rows)
    matches = sum(row["top1_matches"] for row in rows)
    delta_nll = means["kimi"] - means["manual"]
    return dict(passages=rows, targets=count, top1_matches=matches, max_relative_l2=max_l2,
                mean_nll=means, perplexity={mode: exp(value) for mode, value in means.items()},
                kimi_minus_manual_nll=delta_nll,
                passed=max_l2 <= .02 and matches / count >= .99 and abs(delta_nll) <= .01)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if os.environ.get("INCO_API_KEY"):
        raise ValueError("Unset INCO_API_KEY before offline evaluation")
    search = HERE / "decode-search/round-1-retry"
    proposal = read_json(search / "proposal.json")
    result = read_json(search / "evaluation.json")
    config = OPTIONS[proposal["config_id"]]
    assert result["status"] == "PASS" and result["configurations"]["challenger"] == list(config)
    assert 1 - 1 / result["median_paired_speedup"]["wall_ms"] >= .01
    assert result["challenger_wall_wins"] >= 8
    configs = {"manual": MANUAL, "challenger": config}
    module, kernel_hash = load_selected_kernel_module()
    model, tokenizer = load_model()
    report = dict(status="RUNNING", configurations=configs, prompt_lengths=LENGTHS, timing=[],
                  quality_prefix=PREFIX, experiment_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  gpu=torch.cuda.get_device_name(), telemetry=[snapshot("start")])
    previous = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    try:
        with torch.inference_mode():
            inputs = quality_inputs(tokenizer)
            stock = score_stock(model, inputs)
            weight_hash, selected_hash = configure_mlp(model, "packed")
            assert selected_hash == kernel_hash == result["kernel_sha256"]
            assert weight_hash == result["quantized_sha256"]
            report.update(kernel_sha256=kernel_hash, quantized_sha256=weight_hash)
            report["quality"] = score_packed(model, module, configs, inputs, stock)
            print(f"Quality gate: {report['quality']['passed']}", flush=True)
            assert report["quality"]["passed"], "Held-out text quality gate failed"
            for length in LENGTHS:
                prompt, continuation, digest = workload(tokenizer, 1, length)
                case = dict(prompt_tokens=length, input_sha256=digest)
                compare_workload(model, module, case, configs, prompt, continuation)
                report["timing"].append(case)
                save_report(OUT, report)
                print(json.dumps(dict(prompt_tokens=length, wall_ms=case["medians_ms_per_step"],
                                      wins=case["challenger_wall_wins"])), flush=True)
            report["status"] = "PASS"
    except Exception:
        report["status"] = "FAILED"
        raise
    finally:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = previous
        report["telemetry"].append(snapshot("end"))
        save_report(OUT, report)


if __name__ == "__main__":
    main()
