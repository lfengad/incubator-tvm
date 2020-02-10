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
"""External function interface to HashTable libraries."""
from __future__ import absolute_import as _abs

from .. import api as _api, intrin as _intrin

import tvm

def create(key_dtype, value_dtype, dtype, **kwargs):
    """Create a Hash Table object.

    Parameters
    ----------
    key_dtype : string
        Data type of the keys in the table.

    value_dtype : string
        Data type of the values in the table.

    dtype: string
        Data type of the output tensor for the hashtable, usually as custom[hashtable]64.

    Returns
    -------
    C : Tensor
        A tensor with the hashtable pointer as its element.
    """
    return _api.extern(
        (1,),
        [],
        lambda ins, outs: _intrin.call_packed(
            "tvm.contrib.hashtable_handler.create", outs[0], key_dtype, value_dtype
        ),
        dtype="custom[hashtable]64",
        name="C",
        **kwargs
    )


def find(table_reference, key_to_check, default_value, dtype, **kwargs):
    """Find the corresponding values to given keys from a given hashtable.

    Parameters
    ----------
    table_reference : Tensor
        A tensor with hashtable pointer as its element.
        To specify the given hashtable for lookup.

    key_to_check : Tensor
        A tensor of a set of keys for checking.

    default_value : Tensor
        A tensor to specify the default value when no corresponding key in the hashtable.

    key_dtype : string
        Data type of the keys in the table.

    value_dtype : string
        Data type of the values in the table.

    dtype : string
        Data type of the output from the op, usually same as value_dtype.

    Returns
    -------
    C : Tensor
        A tensor with the same shape as key_to_check.
        The corresponding values returned for the checked keys.
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
    """Initialize the hash table using given keys and values tensors

    Parameters
    ----------
    table_refernece : Tensor
        A tensor with hashtable pointer as its element.
        To specify the given hashtable for to initialize.
               
    keys : Tensor
        A tensor to specify the keys in the key-value pairs for initialization.

    values : Tensor
        A tensor to specify the values in the key-value pairs for initialization.

    key_dtype : string
        Data type of the keys in the table.

    value_dtype : string
        Data type of the values in the table.

    Returns
    -------
    C : Tensor
        A dummy node as return.
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
    """Initialize the hash table using a given text file

    Parameters
    ----------
    table_refernece : Tensor
        A tensor with hashtable pointer as its element.
        To specify the given hashtable for to initialize.
               
    files : Tensor
        A tensor of a string to specify the path fo the text file for initialization.

    vocab_size : int
        The number of valid elements in the text file, if known. 
vocab_size: The number of elements in the file, if known.

    key_index: int 
        The column index from the text file to get the key values from. The default is to use the whole line content.

    value_index: int 
        The column index from the text file to get the value values from. The default is to use the line number, starting from zero.

    delimiter: string 
        The delimiter to separate fields in a line.

    Returns
    -------
    C : Tensor
        A dummy node as return.
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
