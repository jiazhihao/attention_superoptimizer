# Mirage Installation

The Mirage python package can be built from source code by following the instructions below. We are also working on providing prebuilt Mirage docker images with all dependencies preinstalled.

### Install Z3 or build Z3 from source code

If the environment does not have a pre-installed Z3, you can build Z3 from source code using the following command lines
```
cd deps/z3
mkdir build
cd build
cmake ..
make -j
```
This will install Z3 in the `build` folder

### Build the Mirage runtime library
Second, you will need to build the Mirage runtime library
You will need to set `CUDACXX` and `Z3_DIR` to let cmake find the paths to CUDA and Z3 librarires.
```
export CUDACXX=/usr/local/cuda/bin/nvcc
export Z3_DIR=/path-to-mirage/deps/z3/build
mkdir build; cd build; cmake ..
make -j
```

### Install the Mirage python package
Finally, you will install the Mirage python package, which allows you to use Mirage's python package to superoptimize DNNs.
```
cd /path-to-mirage/python
python setup.py install
```

### Docker images

We will release the Mirage docker images to quickly try Mirage.
