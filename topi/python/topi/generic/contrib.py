
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
# pylint: disable=invalid-name, no-member
"""Generic contrib hashtable operators"""
from __future__ import absolute_import as _abs
import tvm
from .. import cpp
from .extern import *

@tvm.target.generic_func
def schedule_hash_table(outs):
    """Schedule for reorg

    Parameters
    ----------
    outs: Array of Tensor
      The computation graph description of reorg
      in the format of an array of tensors.

    Returns
    -------
    s: Schedule
      The computation schedule for the op.
    """
    return schedule_extern(outs)

@tvm.target.generic_func
def schedule_lookup_table_find(outs):
    """Schedule for get_valid_counts

    Parameters
    ----------
    outs: Array of Tensor
      The computation graph description of nms
      in the format of an array of tensors.

    Returns
    -------
    s: Schedule
      The computation schedule for the op.
    """
    return schedule_extern(outs)

@tvm.target.generic_func
def schedule_lookup_table_import(outs):
    """Schedule for non-maximum suppression

    Parameters
    ----------
    outs: Array of Tensor
      The computation graph description of nms
      in the format of an array of tensors.

    Returns
    -------
    s: Schedule
      The computation schedule for the op.
    """
    return schedule_extern(outs)
