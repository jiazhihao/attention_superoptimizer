import torch
import numpy as np


X = torch.rand([16, 8192], dtype=torch.float16, device='cuda')
A = torch.rand([8192, 8], dtype=torch.float16, device='cuda')
B = torch.rand([8, 8192], dtype=torch.float16, device='cuda')

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

repetitions = 1024
timings=np.zeros((repetitions,1))

with torch.no_grad():
  for rep in range(repetitions):
      starter.record()
      D = torch.matmul(X, A)
      D = torch.relu(D)
      E = torch.matmul(D, B)
      torch.add(X, E)
      ender.record()
      torch.cuda.synchronize()
      curr_time = starter.elapsed_time(ender)
      timings[rep] = curr_time

mean_syn = np.sum(timings) / repetitions

print(mean_syn)

