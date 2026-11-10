import json
import torch
from qstream import quantize, dequantize, unpack

passed = []
with torch.inference_mode():
    for seed in (7, 42, 123):
        torch.manual_seed(seed)
        for group in (16, 32, 64, 128, 256):
            for magnitude in (1e-5, 1, 1000):
                # Transpose is intentionally noncontiguous.
                w = torch.randn(256, 8, device='cuda').t() * magnitude
                q = quantize(w, group)
                assert torch.isfinite(q.scales).all()
                codes = unpack(q)
                assert codes.min() >= -7 and codes.max() <= 7
                error = (w-dequantize(q, torch.float32)).abs().reshape(8, -1, group)
                assert torch.all(error <= q.scales.unsqueeze(-1)/2 + magnitude*1e-6)
                passed.append([seed,group,magnitude])
    invalid = [torch.ones(128, device='cuda'), torch.ones(2,127, device='cuda'),
               torch.full((2,128), float('nan'), device='cuda'), torch.full((2,128), float('inf'), device='cuda')]
    for w in invalid:
        try:
            quantize(w)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid input accepted')
    for group in (0, -2, 3, 256):
        try:
            quantize(torch.ones(2,128,device='cuda'), group)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid group size accepted')
print(json.dumps(dict(correctness='PASS', reconstruction_cases=len(passed), invalid_cases=8), indent=2))
