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
# pylint: disable=invalid-name, unused-argument
"""Definition of hashtable ops"""
import topi
from .. import op as reg
from ..op import OpPattern


@reg.register_schedule("contrib.hash_table")
def schedule_hash_table(_, outs, target):
    """Schedule definition of hash_table"""
    with target:
        return topi.generic.schedule_hash_table(outs)


@reg.register_compute("contrib.hash_table")
def compute_hash_table(attrs, inputs, _, target):
    """Compute definition of hash_table"""
    key_dtype = attrs.key_dtype
    value_dtype = attrs.value_dtype
    return [topi.hash_table(key_dtype, value_dtype)]


reg.register_pattern("contrib.hash_table", OpPattern.OPAQUE)


@reg.register_schedule("contrib.lookup_table_find")
def schedule_lookup_table_find(_, outs, target):
    """Schedule definition of lookup_table_find"""
    with target:
        return topi.generic.schedule_lookup_table_find(outs)


@reg.register_compute("contrib.lookup_table_find")
def compute_lookup_table_find(attrs, inputs, _, target):
    """Compute definition of lookup_table_find"""
    dtype = attrs.dtype
    return [topi.lookup_table_find(
        inputs[0], inputs[1], inputs[2], dtype)]


reg.register_pattern("contrib.lookup_table_find", OpPattern.OPAQUE)


@reg.register_schedule("contrib.lookup_table_import")
def schedule_lookup_table_import(_, outs, target):
    """Schedule definition of lookup_table_import"""
    with target:
        return topi.generic.schedule_lookup_table_import(outs)


@reg.register_compute("contrib.lookup_table_import")
def compute_lookup_table_import(attrs, inputs, _, target):
    """Compute definition of lookup_table_import"""
    return [topi.lookup_table_import(
        inputs[0], inputs[1], inputs[2])]

reg.register_pattern("contrib.lookup_table_import", OpPattern.OPAQUE)


@reg.register_schedule("contrib.initialize_table_from_text_file")
def schedule_initialize_table_from_text_file(_, outs, target):
    """Schedule definition of initialize_table_from_text_file"""
    with target:
        return topi.generic.schedule_initialize_table_from_text_file(outs)


@reg.register_compute("contrib.initialize_table_from_text_file")
def compute_initialize_table_from_text_file(attrs, inputs, _, target):
    """Compute definition of initialize_table_from_text_file"""
    vocab_size = attrs.vocab_size
    key_index = attrs.key_index
    value_index = attrs.value_index
    delim = attrs.delim
    return [topi.initialize_table_from_text_file(
        inputs[0], inputs[1], vocab_size, key_index, value_index, delim)]

reg.register_pattern("contrib.initialize_table_from_text_file", OpPattern.OPAQUE)
