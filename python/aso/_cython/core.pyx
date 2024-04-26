# Copyright 2024 CMU
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from CCore cimport *
from cpython cimport array
import ctypes
import array
import numpy as np

cdef class PyTensor:
    cdef DTensor c_ptr # hold a pointer to the C++ instance which we're wrapping

    def __cinit__(self):
        pass

    def __init__(self, tensor):
        self.c_ptr.num_dims = tensor.num_dims

cdef class PyGraph:
    cdef Graph *p_graph #Hold a Graph instance

    def __cinit__(self, graph = None):
        cdef unsigned long long ptr
        if graph is None:
            self.p_graph = new Graph()
        else:
            ptr = ctypes.cast(graph, ctypes.c_void_p).value
            self.p_graph = <Graph*>(ptr)

    def new_input(self, *, tuple dims, dtype):
        cdef vector[int] cdims
        cdims.resize(len(dims))
        for i in range(len(dims)):
            cdims[i] = dims[i]
        cdef DTensor handle = self.p_graph.new_input(cdims, DT_FLOAT16, DmemRowMajor)
        return PyTensor(handle)
