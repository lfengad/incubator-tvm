# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""External function interface to BLAS libraries."""
from __future__ import absolute_import as _abs

from .. import api as _api, intrin as _intrin

import tvm

def create(key_dtype, value_dtype, dtype, **kwargs):
    """Create an extern op that compute matrix mult of A and rhs with CrhsLAS

    This function serves as an example on how to call external libraries.

    Parameters
    ----------
    lhs : Tensor
        The left matrix operand
    rhs : Tensor
        The right matrix operand
    transa : bool
        Whether transpose lhs
    transb : bool
        Whether transpose rhs

    Returns
    -------
    C : Tensor
        The result tensor.
    """
    #key_example = tvm.const(1, key_dtype)
    #value_example = tvm.const(1, key_dtype)
    return _api.extern(
        (1,),
        [],
        lambda ins, outs: _intrin.call_packed(
            "tvm.contrib.hashtable_handler.create", outs[0], key_dtype, value_dtype
        ),
        dtype = "custom[hashtable]64",
        name="C",
        **kwargs
    )


def find(table_reference, key_to_check, default_value, dtype, **kwargs):
    """Create an extern op that compute batched matrix mult of A and rhs with CBLAS
     This function serves as an example on how to call external libraries.
     Parameters
    ----------
    lhs : Tensor
        The left matrix operand
    rhs : Tensor
        The right matrix operand
    transa : bool
        Whether transpose lhs
    transb : bool
        Whether transpose rhs
     Returns
    -------
    C : Tensor
        The result tensor.
    """
    return _api.extern(
        key_to_check.shape,
        [table_reference, key_to_check, default_value],
        lambda ins, outs: _intrin.call_packed(
            "tvm.contrib.hashtable_handler.find",
            ins[0],
            ins[1],
            ins[2],
            outs[0]
        ),
        name="C",
        dtype=dtype,
        **kwargs
    )

def init(table_reference, keys, values, **kwargs):
    """Create an extern op that compute batched matrix mult of A and rhs with CBLAS
     This function serves as an example on how to call external libraries.
     Parameters
    ----------
    lhs : Tensor
        The left matrix operand
    rhs : Tensor
        The right matrix operand
    transa : bool
        Whether transpose lhs
    transb : bool
        Whether transpose rhs
     Returns
    -------
    C : Tensor
        The result tensor.
    """
    return _api.extern(
        (1,),
        [table_reference, keys, values],
        lambda ins, outs: _intrin.call_packed(
            "tvm.contrib.hashtable_handler.init",
            ins[0],
            ins[1],
            ins[2],
            outs[0]
        ),
        name="C",
        dtype="int32",
        **kwargs
    )



def initfromtxt(table_reference, files, vocab_size, key_index, value_index, delim, **kwargs):
    """Create an extern op that compute batched matrix mult of A and rhs with CBLAS
     This function serves as an example on how to call external libraries.
     Parameters
    ----------
    lhs : Tensor
        The left matrix operand
    rhs : Tensor
        The right matrix operand
    transa : bool
        Whether transpose lhs
    transb : bool
        Whether transpose rhs
     Returns
    -------
    C : Tensor
        The result tensor.
    """
    return _api.extern(
        (1,),
        [table_reference, files],
        lambda ins, outs: _intrin.call_packed(
            "tvm.contrib.hashtable_handler.initfromtxt",
            ins[0],
            ins[1],
            vocab_size,
            key_index,
            value_index,
            delim,
            outs[0]
        ),
        name="C",
        dtype="int32",
        **kwargs
    )
