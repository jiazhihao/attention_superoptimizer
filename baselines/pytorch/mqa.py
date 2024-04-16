import torch
import numpy as np

Q = torch.rand([1, 64, 1, 64], dtype=torch.float16, device='cuda')
K = torch.rand([1, 64, 64, 4096], dtype=torch.float16, device='cuda')
V = torch.rand([1, 64, 4096, 64], dtype=torch.float16, device='cuda')

multihead_attn = torch.nn.MultiheadAttention(embed_dim=32 * 64, num_heads = 32, batch_first=True, device='cuda', dtype=torch.float16)

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

repetitions = 1024
timings=np.zeros((repetitions,1))

with torch.no_grad():
  for rep in range(repetitions):
      starter.record()
      S = torch.matmul(Q, K)
      S = torch.softmax(S, dim=3)
      S = torch.matmul(S, V)
      ender.record()
      torch.cuda.synchronize()
      curr_time = starter.elapsed_time(ender)
      timings[rep] = curr_time

mean_syn = np.sum(timings) / repetitions

print(mean_syn)

