import json
import math
import torch
from qstream import quantize, unpack, dequantize, swiglu, quantized_swiglu

def main():
    torch.manual_seed(42)
    device = 'cuda'
    results = []
    for group in (16, 32, 128):
        # Every representable nibble, including negative values, and zero groups.
        w = torch.arange(-7, 8, device=device).repeat(math.ceil(256 / 15))[:256].reshape(2, 128).float()
        qw = quantize(w, group)
        assert torch.equal(unpack(qw), w.to(torch.int8))
        assert torch.equal(dequantize(qw, torch.float32), w)
        zero = torch.zeros_like(w)
        assert torch.equal(dequantize(quantize(zero, group)), zero)
        random = torch.randn(8, 256, device=device)
        qr = quantize(random, group)
        error = (random - dequantize(qr, torch.float32)).abs().reshape(8, -1, group)
        assert torch.all(error <= qr.scales.unsqueeze(-1) / 2 + 1e-6)
    for batch, hidden, intermediate in ((1, 128, 256), (16, 256, 512), (64, 512, 1024)):
        x = torch.randn(batch, hidden, device=device, dtype=torch.float16)
        ws = [torch.randn(o, i, device=device, dtype=torch.float16) / math.sqrt(i)
              for o, i in ((intermediate, hidden), (intermediate, hidden), (hidden, intermediate))]
        fp16 = swiglu(x, *ws)
        fp32 = swiglu(x.float(), *(w.float() for w in ws))
        torch.testing.assert_close(fp16.float(), fp32, atol=0.005, rtol=0.02)
        qs = [quantize(w) for w in ws]
        actual = quantized_swiglu(x, *qs)
        reference = swiglu(x, *(dequantize(q) for q in qs))
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        assert torch.isfinite(actual).all()
        rel = ((actual.float() - fp32).norm() / fp32.norm()).item()
        cosine = torch.nn.functional.cosine_similarity(actual.float().flatten(), fp32.flatten(), dim=0).item()
        assert rel < 0.35 and cosine > 0.95, (rel, cosine)
        results.append(dict(shape=[batch, hidden, intermediate], relative_l2=rel, cosine=cosine,
                            packed_bytes=sum(q.packed.numel() + 4*q.scales.numel() for q in qs),
                            fp16_bytes=sum(2*w.numel() for w in ws)))
    print(json.dumps({'correctness': 'PASS', 'seed': 42, 'results': results}, indent=2))

if __name__ == '__main__':
    with torch.inference_mode():
        main()
