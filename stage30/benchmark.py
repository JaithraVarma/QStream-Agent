"""Build deterministic Qwen workloads and write benchmark receipts."""
import hashlib
import json
import os
DECODE_STEPS = 16
TEXT = (
    "A transformer reads the prompt, then predicts the next token. ",
    "The model stores keys and values for the context it has processed. ",
    "A small GPU can run a compact model for experiments. ",
    "The feed-forward block mixes features at each token position. ",
)


def workload(tokenizer, batch, length):
    import torch
    from .all_layers import fixed_tokens

    tokens = torch.tensor([fixed_tokens(tokenizer, text, length + DECODE_STEPS)
                           for text in TEXT[:batch]], dtype=torch.long)
    fingerprint = hashlib.sha256(tokens.numpy().tobytes()).hexdigest()
    tokens = tokens.cuda()
    prompt = tokens[:, :length].contiguous()
    continuation = [tokens[:, length+i:length+i+1].contiguous() for i in range(DECODE_STEPS)]
    return prompt, continuation, fingerprint


def events():
    import torch

    timers = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    for timer in timers:
        timer.record()
    timers[-1].synchronize()
    return timers


def trial(model, prompt, continuation, timers, check=False):
    import torch

    pre_start, pre_end, dec_start, dec_end = timers
    wall_start = perf_counter()
    pre_start.record()
    output = model(input_ids=prompt, use_cache=True, logits_to_keep=1)
    pre_end.record()
    pre_end.synchronize()
    pre_wall = 1000 * (perf_counter() - wall_start)
    cache = output.past_key_values
    if check:
        assert torch.isfinite(output.logits).all() and cache.get_seq_length() == prompt.shape[1]
    del output

    wall_start = perf_counter()
    dec_start.record()
    for token in continuation:
        output = model(input_ids=token, past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = output.past_key_values
    dec_end.record()
    dec_end.synchronize()
    dec_wall = 1000 * (perf_counter() - wall_start)
    if check:
        assert torch.isfinite(output.logits).all()
        assert cache.get_seq_length() == prompt.shape[1] + len(continuation)
    return dict(prefill_event_ms=pre_start.elapsed_time(pre_end), prefill_wall_ms=pre_wall,
                decode_event_ms=dec_start.elapsed_time(dec_end), decode_wall_ms=dec_wall)


def save_report(output, report):
    temporary = output.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output)
