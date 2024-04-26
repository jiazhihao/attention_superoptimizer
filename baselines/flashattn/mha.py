from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
import torch
import numpy as np

Q = torch.rand([8, 256, 32, 64], dtype=torch.float16, device='cuda')
K = torch.rand([8, 4096, 32, 64], dtype=torch.float16, device='cuda')
V = torch.rand([8, 4096, 32, 64], dtype=torch.float16, device='cuda')

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

repetitions = 128
timings=np.zeros((repetitions,1))

with torch.no_grad():
  for rep in range(repetitions):
      starter.record()
      flash_attn_func(Q, K, V)
      ender.record()
      torch.cuda.synchronize()
      curr_time = starter.elapsed_time(ender)
      timings[rep] = curr_time

mean_syn = np.sum(timings) / repetitions

print(mean_syn)

