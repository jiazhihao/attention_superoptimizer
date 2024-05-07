# Mirage: A Multi-level Superoptimizer for Tensor Algebra

Mirage is a tensor algebra superoptimizer that automatically discovers and verifies sophisticated tensor program optimizations, most of which require joint optimization of algebraic transformations, schedule transformations, and discovery of new custom kernels.
For an input DNN model, 

## Installation

See [instructions](INSTALL.md) to install Mirage from source.

## Quickstart

The following example shows how to use Mirage to automatically generate CUDA kernels for group-query attention (GQA) in LLAMA-3-70B. We assume the model is served in half precision and is tensor model parallelized across 4 GPUs to fit in GPU device memory. Therefore, the GQA operator computes attention across 8 query heads and 2 key-value heads.

First, we define the computation graph for GQA, which takes three input tensors `Q`, `K`, and `V`, and produces a single output tensor `O` that contains the attention result.

```python
import mirage as mi
graph = mi.new_graph()
Q = graph.new_input(dims=(2, 256, 64), dtype=mi.float16)
K = graph.new_input(dims=(2, 64, 4096), dtype=mi.float16)
V = graph.new_input(dims=(2, 4096, 64), dtype=mi.float16)
A = graph.matmul(Q, K)
E = graph.exp(A)
S = graph.reduction(E, 2)
D = graph.div(E, S)
O = graph.matmul(D, V)
```

Second, we will use `mi.optimize` to superoptimize an input DNN, which searches the space of potential mugraphs, a hierarchical graph representation that specifies a DNN at the kernel, thread block, and thread levels of the GPU compute hierarchy, to discover highly-optimized tensor programs.
Mirage can automatically discover mugraphs that represent expert-designed GPU optimizations such as FlashAttention and its inference variant FlashDecoding.
In addition, Mirage also finds other mugraphs that outperform these manually designed optimizations for certain use cases.

```python
graphs = mi.optimize(graph, griddims=[(2, 16, 1), (2, 16, 4)])
```
The search is configured by the following parameters (`griddims` is the one you are likely to reset for your problem size).

