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
"""Hash-table related operations."""
from __future__ import absolute_import as _abs
from . import _make
from ...expr import TupleWrapper

def hash_table(key_dtype="int32",
               value_dtype="int32",
               dtype="int32"):
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
    out_tensor : relay.Expr
        A tensor with the hashtable pointer as its element.
    """
    return _make.hash_table(key_dtype, value_dtype, dtype)


def lookup_table_find(table_reference,
                      key_to_check,
                      default_value,
                      key_dtype="int32",
                      value_dtype="int32",
                      dtype="int32"):
    """Find the corresponding values to given keys from a given hashtable.

    Parameters
    ----------
    table_reference : relay.Expr
        A tensor with hashtable pointer as its element.
        To specify the given hashtable for lookup.

    key_to_check : relay.Expr
        A tensor of a set of keys for checking.

    default_value : relay.Expr
        A tensor to specify the default value when no corresponding key in the hashtable.

    key_dtype : string
        Data type of the keys in the table.

    value_dtype : string
        Data type of the values in the table.

    dtype : string
        Data type of the output from the op, usually same as value_dtype.

    Returns
    -------
    out : relay.Expr
        A tensor with the same shape as key_to_check.
        The corresponding values returned for the checked keys.
    """
    return _make.lookup_table_find(table_reference,
                                   key_to_check,
                                   default_value,
                                   key_dtype,
                                   value_dtype,
                                   dtype)


def lookup_table_import(table_reference,
                        keys,
                        values,
                        key_dtype="int32",
                        value_dtype="int32"):
    """Initialize the hash table using given keys and values tensors

    Parameters
    ----------
    table_refernece : relay.Expr
        A tensor with hashtable pointer as its element.
        To specify the given hashtable for to initialize.
               
    keys : relay.Expr
        A tensor to specify the keys in the key-value pairs for initialization.

    values : relay.Expr
        A tensor to specify the values in the key-value pairs for initialization.

    key_dtype : string
        Data type of the keys in the table.

    value_dtype : string
        Data type of the values in the table.

    Returns
    -------
    out : relay.Expr
        A dummy node as return.
    """
    return _make.lookup_table_import(table_reference,
                                     keys,
                                     values,
                                     key_dtype,
                                     value_dtype)


def initialize_table_from_text_file(table_reference,
                                    files,
                                    vocab_size,
                                    key_index,
                                    value_index,
                                    delim):
    """Initialize the hash table using a given text file

    Parameters
    ----------
    table_refernece : relay.Expr
        A tensor with hashtable pointer as its element.
        To specify the given hashtable for to initialize.
               
    files : relay.Expr
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
    out : relay.Expr
        A dummy node as return.
    """
    return _make.initialize_table_from_text_file(table_reference,
                                                 files,
                                                 vocab_size,
                                                 key_index,
                                                 value_index,
                                                 delim)


