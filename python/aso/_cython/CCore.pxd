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

from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "aso/type.h" namespace "aso::type":
    # This must be consistent with aso/type.h
    cdef enum DataType:
        DT_INT8 = 900,
        DT_UINT16 = 910,
        DT_BFLOAT16 = 920,
        DT_FLOAT16 = 921,
        DT_FLOAT32 = 930,
        DT_DOUBLE = 940,
        DT_UNKNOWN = 999,
  
cdef extern from "aso/layout.h" namespace "aso::layout":
    # This must be consistent with aso/layout.h
    cdef enum DmemLayout:
        DmemRowMajor = 100,
        DmemColumnMajor = 101,
        DmemUnknowLayout = 199,

cdef extern from "aso/kernel/graph.h" namespace "aso::kernel":
    cdef cppclass KNOperator:
        pass
    ctypedef struct DTensor:
        DataType data_type
        DmemLayout layout
        int num_dims
        int dim[4]
        size_t guid
        #KNOperator *owner_op
        #void *data_ptr
        int owner_ts_idx
        pass

    cdef cppclass Graph:
        Graph()
        DTensor* new_input_ptr(vector[int] dims,
                               DataType data_type,
                               DmemLayout layout)
        DTensor* matmul(const DTensor* A, const DTensor* B)
        DTensor* exp(const DTensor* input)
        DTensor* add(const DTensor* op1, const DTensor* op2)
        DTensor* mul(const DTensor* op1, const DTensor* op2)
        DTensor* div(const DTensor* op1, const DTensor* op2)
