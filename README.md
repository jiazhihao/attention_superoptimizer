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
The search is configured by the following parameters (`griddims` is the one you are likely to reset for your problem sizes). Some parameters are highly specific to mugraphs, we `

* `griddims`: the number of thread blocks along each dimension of the grid. (default: `[(16, 1, 1), (16, 2, 1), (16, 4, 1)]`)
* `blockdims`: the number of threads along each dimension of the thread block. (default: `[(128, 1, 1)]`)
* `imaps`: potential mappings between data dimensions of an input tensor and `griddims` (default: `[(0, -1, -1), (0, 1, -1), (0, 2, -1), (0, -1, 1)]`). Note that `-1` indicates the input tensor is replicated across thread blocks along that grid dimension.
* `omaps`: potential mappings between data dimensions of an output tensor and `griddims` (default: `[(0, -1, -1), (0, 1, -1), (0, 2, -1), (0, 2, 1)]`).
* `fmaps`: potential mappings between data dimensions of an input tensor and the for-loop dimension of the thread block (default: `[-1, 1, 2]`). Again, `-1` indicates the input tensor is replicated across different for-loop iterations.
* `franges`: possible for-loop ranges to be considered during the search (default: `1, 4, 8, 16`).

Note that the default values for these parameters are tailored for multi-head/multi-query/group-query attention. Except for `griddims`, which depends on the problem sizes, the default values for other parameters are sufficient to discover FlashAttn, FlashDecoding, and many other optimized attention kernerls.

The above code snippet returns `graphs`, a list of mugraphs discovered by Mirage that are functionally equivalent to the input program. Mirage uses a probabilistic equivalence verification mechanism to ensure that all discovered mugraphs are equivalent to the inpout. Finally, we can generate Triton programs from these mugraphs.

```python
for i, graph in enumerate(graphs):
    graph.generate_triton_program("generated_program_{}.py".format(i))
```

The above search procedure takes around 4 hours and discovers 69 potential tensor programs for implementing group-query attention. To bypass the search and directly check the generated programs, we can start from a previous checkpoint of the search by running
```bash
python demo/demo_group_query_attention_spec_decode.py --checkpoint demo/checkpoint_group_query_attn_spec_decode.json
```
This returns 69 Triton programs. The performance of these programs on a NVIDIA A100 GPU is as follows. Note that some generated programs perform small matrix multiplications within a thread block. These programs cannot be directly supporrted by Triton, which requires a minimum of 16x16x16 matrix multiplication.
