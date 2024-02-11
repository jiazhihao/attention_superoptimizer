# attention_superoptimizer
An Attention Superoptimizer

## Installation

### Install Z3 or Build Z3 from source code

* If the environment does not have a pre-installed Z3, you can build Z3 from source code using the following command lines
```
cd deps/z3
mkdir build
cd build
cmake ..
make -j
```
This will install Z3 in the `build` folder

### Build ASO
Build the ASO runtime library. You will need to set `CUDACXX` and `Z3_DIR` to let cmake find the paths to CUDA and Z3 librarires.
```
export CUDACXX=/usr/local/cuda/bin/nvcc
export Z3_DIR=/path-to-aso/deps/z3/build
mkdir build; cd build; cmake ..
make -j
```
