"""Kernel breakdown of one force-training step (torch.profiler)."""
import math, sys
import torch
sys.path.insert(0, __import__("os").path.dirname(__file__))
from bench_pet_attn import METHODS, make_step, mt  # noqa: E402

N, L, H, d = (int(x) for x in sys.argv[1:5])
method = sys.argv[5] if len(sys.argv) > 5 else "mlipcpp"
D = H * d
torch.manual_seed(0)
qkv = torch.randn(N, L, 3 * D, device="cuda", requires_grad=True)
bias = (0.3 * torch.randn(N, L, L, device="cuda")).requires_grad_()
u = torch.randn(N, L, D, device="cuda", requires_grad=True)
wq, wb = torch.randn_like(qkv), torch.randn_like(bias)
step = make_step(METHODS[method], "force", qkv, bias, u, wq, wb, H, 1 / math.sqrt(d))
for _ in range(5):
    step()
torch.cuda.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
    for _ in range(10):
        step()
    torch.cuda.synchronize()
print(p.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=70))
