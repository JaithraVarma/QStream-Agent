"""Run and time cached Qwen decode with static KV storage."""
from time import perf_counter

import torch


def forward(model, tokens, cache, position=None, mask=None):
    return model(input_ids=tokens, past_key_values=cache, use_cache=True,
                 cache_position=position, attention_mask=mask, logits_to_keep=1)


def mask_for(positions, keys):
    return keys[None, None, None, :] <= positions[None, None, :, None]


def cache_pointers(cache):
    return tuple((layer.keys.data_ptr(), layer.values.data_ptr()) for layer in cache.layers)


def prefill(model, cache, prompt, positions, keys):
    cache.reset()
    return forward(model, prompt, cache, positions, mask_for(positions, keys)).logits


def dynamic_reference(model, prompt, continuation):
    output = forward(model, prompt, None)
    logits = [output.logits]
    for token in continuation:
        output = forward(model, token, output.past_key_values)
        logits.append(output.logits)
    assert output.past_key_values.get_seq_length() == prompt.shape[1] + len(continuation)
    return logits


def checked_sequence(model, cache, prompt, continuation, positions, keys,
                     stage_inputs, execute, pointers):
    prompt_length = prompt.shape[1]
    logits = [prefill(model, cache, prompt, positions[:prompt_length], keys)]
    for index, token in enumerate(continuation):
        position = prompt_length + index
        stage_inputs(token, positions[position:position+1])
        logits.append(execute().clone())
    torch.cuda.synchronize()
    assert cache.get_seq_length() == prompt_length + len(continuation)
    assert cache_pointers(cache) == pointers
    return logits


def measure(model, cache, prompt, continuation, positions, keys, stage_inputs, execute, timers):
    prompt_length = prompt.shape[1]
    prefill(model, cache, prompt, positions[:prompt_length], keys)
    torch.cuda.synchronize()
    start, end = timers[:2]
    wall = perf_counter()
    start.record()
    for index, token in enumerate(continuation):
        position = prompt_length + index
        stage_inputs(token, positions[position:position+1])
        execute()
    end.record()
    end.synchronize()
    wall_ms = (perf_counter() - wall) * 1000
    return dict(event_ms=start.elapsed_time(end), wall_ms=wall_ms)
