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
# pylint: disable=import-error, invalid-name, no-member, too-many-locals, too-many-arguments, undefined-variable, too-many-nested-blocks, too-many-branches, too-many-statements, too-many-function-args
"""Non-maximum suppression operator"""
import tvm

from tvm import hybrid
from ..sort import argsort
from tvm.contrib import hashtable_handler
from .. import *

@tvm.target.generic_func
def hash_table(key_dtype, value_dtype, dtype):
    """Get valid count of bounding boxes given a score threshold.
    Also moves valid boxes to the top of input data.

    Parameters
    ----------
    data : tvm.Tensor
        Input data. 3-D tensor with shape [batch_size, num_anchors, 6]
        or [batch_size, num_anchors, 5].

    score_threshold : optional, float
        Lower limit of score for valid bounding boxes.

    id_index : optional, int
        index of the class categories, -1 to disable.

    score_index: optional, int
        Index of the scores/confidence of boxes.

    Returns
    -------
    out_tensor : tvm.Tensor
        Rearranged data tensor.

    valid_count : tvm.Tensor
        1-D tensor for valid number of boxes.
    """
    table_instance = hashtable_handler.create(key_dtype, value_dtype, dtype)
    return table_instance



@tvm.target.generic_func
def lookup_table_find(table_reference, key_to_check, default_value, key_dtype, value_dtype, dtype):
    """Get valid count of bounding boxes given a score threshold.
    Also moves valid boxes to the top of input data.

    Parameters
    ----------
    data : tvm.Tensor
        Input data. 3-D tensor with shape [batch_size, num_anchors, 6]
        or [batch_size, num_anchors, 5].

    score_threshold : optional, float
        Lower limit of score for valid bounding boxes.

    id_index : optional, int
        index of the class categories, -1 to disable.

    score_index: optional, int
        Index of the scores/confidence of boxes.

    Returns
    -------
    out_tensor : tvm.Tensor
        Rearranged data tensor.

    valid_count : tvm.Tensor
        1-D tensor for valid number of boxes.
    """
    #default_value_ = topi.cast(default_value, dtype=dtype)
    found_values = hashtable_handler.find(table_reference, key_to_check, default_value, dtype)
    return found_values

@tvm.target.generic_func
def lookup_table_import(table_reference, keys, values, key_dtype, value_dtype):
    """Get valid count of bounding boxes given a score threshold.
    Also moves valid boxes to the top of input data.

    Parameters
    ----------
    data : tvm.Tensor
        Input data. 3-D tensor with shape [batch_size, num_anchors, 6]
        or [batch_size, num_anchors, 5].

    score_threshold : optional, float
        Lower limit of score for valid bounding boxes.

    id_index : optional, int
        index of the class categories, -1 to disable.

    score_index: optional, int
        Index of the scores/confidence of boxes.

    Returns
    -------
    out_tensor : tvm.Tensor
        Rearranged data tensor.

    valid_count : tvm.Tensor
        1-D tensor for valid number of boxes.
    """
    fake_out = hashtable_handler.init(table_reference, keys, values)
    return fake_out

@tvm.target.generic_func
def initialize_table_from_text_file(table_reference, files, vocab_size, key_index, value_index, delim):
    """Get valid count of bounding boxes given a score threshold.
    Also moves valid boxes to the top of input data.

    Parameters
    ----------
    data : tvm.Tensor
        Input data. 3-D tensor with shape [batch_size, num_anchors, 6]
        or [batch_size, num_anchors, 5].

    score_threshold : optional, float
        Lower limit of score for valid bounding boxes.

    id_index : optional, int
        index of the class categories, -1 to disable.

    score_index: optional, int
        Index of the scores/confidence of boxes.

    Returns
    -------
    out_tensor : tvm.Tensor
        Rearranged data tensor.

    valid_count : tvm.Tensor
        1-D tensor for valid number of boxes.
    """
    fake_out = hashtable_handler.initfromtxt(table_reference, files, vocab_size, key_index, value_index, delim)
    return fake_out
