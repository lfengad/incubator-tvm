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
import tvm
import numpy as np
import tensorflow as tf
from tvm import relay
from tensorflow.python.framework import graph_util

def test_stringtovalue():
    g=tf.Graph()
    with g.as_default(): 
        input_tensor = tf.placeholder(tf.string,shape=(5,), name='input')
        keys = tf.constant(np.array(['1', '2', '3', '4', '5', '6']),  dtype=tf.string, name='keys')
        values = tf.constant(np.array([4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),  dtype=tf.float32, name='values')
        table_init =tf.contrib.lookup.KeyValueTensorInitializer(keys, values)
        table = tf.contrib.lookup.HashTable(table_init, 0.0)
        res = table.lookup(input_tensor)
        out = tf.identity(res, name='sum')
    data = np.array(['6','3','2','1','5'])
    with tf.Session(graph=out.graph) as sess:
        sess.run([tf.tables_initializer(),tf.global_variables_initializer()])
        tf_out = sess.run(out, feed_dict={input_tensor:data})
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['sum','init_all_tables'])
    

    layout = None
    target = 'llvm'
    ctx=tvm.cpu(0)
    mod, params = relay.frontend.from_tensorflow(constant_graph, layout=layout, outputs=['sum'])
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod,
                                     target=target,
                                     target_host = target,
                                     params=params)
    from tvm.contrib import graph_runtime
    m = graph_runtime.create(graph, lib, ctx)
    m.set_input(**params)
    m.set_input('input', data)
    m.run()
    tvm_out=m.get_output(0)
    print(tvm_out)
    print(tf_out)
    tvm.testing.assert_allclose(tvm_out.asnumpy(), tf_out.astype(tvm_out.dtype), rtol=1e-5)

def test_valuetostring():
    g=tf.Graph()
    with g.as_default(): 
        input_tensor = tf.placeholder(tf.int32,shape=(5,), name='input')
        keys = tf.constant(np.array([4, 5, 6, 7, 8, 9]),  dtype=tf.int32, name='keys')
        values = tf.constant(np.array(['1', '2', '3', '4', '5', '6']),  dtype=tf.string, name='values')
        table_init =tf.contrib.lookup.KeyValueTensorInitializer(keys, values)
        table = tf.contrib.lookup.HashTable(table_init, ' ')
        res = table.lookup(input_tensor)
        out = tf.identity(res, name='sum')
    data = np.array([9, 7, 6, 5, 4])
    with tf.Session(graph=out.graph) as sess:
        sess.run([tf.tables_initializer(),tf.global_variables_initializer()])
        tf_out = sess.run(out, feed_dict={input_tensor:data})
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['sum','init_all_tables'])
    

    layout = None
    target = 'llvm'
    ctx=tvm.cpu(0)
    mod, params = relay.frontend.from_tensorflow(constant_graph, layout=layout, outputs=['sum'])
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod,
                                     target=target,
                                     target_host = target,
                                     params=params)
    from tvm.contrib import graph_runtime
    m = graph_runtime.create(graph, lib, ctx)
    m.set_input(**params)
    m.set_input('input', data)
    m.run()
    tvm_out=m.get_output(0)
    print(tvm_out)
    print(tf_out.astype('str'))
    np.testing.assert_array_equal(tvm_out.asnumpy(), tf_out.astype('str'))

def test_valuetovalue():
    g=tf.Graph()
    with g.as_default(): 
        input_tensor = tf.placeholder(tf.int64,shape=(5,), name='input')
        keys = tf.constant(np.array([4, 5, 6, 7, 8, 9]),  dtype=tf.int64, name='keys')
        values = tf.constant(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),  dtype=tf.float64, name='values')
        table_init =tf.contrib.lookup.KeyValueTensorInitializer(keys, values)
        table = tf.contrib.lookup.HashTable(table_init, 0.0)
        res = table.lookup(input_tensor)
        out = tf.identity(res, name='sum')
    data = np.array([9, 7, 6, 5, 4])
    with tf.Session(graph=out.graph) as sess:
        sess.run([tf.tables_initializer(),tf.global_variables_initializer()])
        tf_out = sess.run(out, feed_dict={input_tensor:data})
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['sum','init_all_tables'])
    

    layout = None
    target = 'llvm'
    ctx=tvm.cpu(0)
    mod, params = relay.frontend.from_tensorflow(constant_graph, layout=layout, outputs=['sum'])
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod,
                                     target=target,
                                     target_host = target,
                                     params=params)
    from tvm.contrib import graph_runtime
    m = graph_runtime.create(graph, lib, ctx)
    m.set_input(**params)
    m.set_input('input', data)
    m.run()
    tvm_out=m.get_output(0)
    print(tvm_out)
    print(tf_out)
    tvm.testing.assert_allclose(tvm_out.asnumpy(), tf_out.astype(tvm_out.dtype), rtol=1e-5)

def test_stringtostring():
    g=tf.Graph()
    with g.as_default(): 
        input_tensor = tf.placeholder(tf.string,shape=(5,), name='input')
        keys = tf.constant(np.array(['4', '5', '6', '7', '8', '9']),  dtype=tf.string, name='keys')
        values = tf.constant(np.array(['1', '2', '3', '4', '5', '6']),  dtype=tf.string, name='values')
        table_init =tf.contrib.lookup.KeyValueTensorInitializer(keys, values)
        table = tf.contrib.lookup.HashTable(table_init, ' ')
        res = table.lookup(input_tensor)
        out = tf.identity(res, name='sum')
    data = np.array(['9', '7', '6', '5', '4'])
    with tf.Session(graph=out.graph) as sess:
        sess.run([tf.tables_initializer(),tf.global_variables_initializer()])
        tf_out = sess.run(out, feed_dict={input_tensor:data})
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['sum','init_all_tables'])
    

    layout = None
    target = 'llvm'
    ctx=tvm.cpu(0)
    mod, params = relay.frontend.from_tensorflow(constant_graph, layout=layout, outputs=['sum'])
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod,
                                     target=target,
                                     target_host = target,
                                     params=params)
    from tvm.contrib import graph_runtime
    m = graph_runtime.create(graph, lib, ctx)
    m.set_input(**params)
    m.set_input('input', data)
    m.run()
    tvm_out=m.get_output(0)
    print(tvm_out)
    print(tf_out.astype('str'))
    np.testing.assert_array_equal(tvm_out.asnumpy(), tf_out.astype('str'))

def test_fromtext():
    g=tf.Graph()
    with g.as_default(): 
        input_tensor = tf.placeholder(tf.string,shape=(5,), name='input')
        table_init =tf.contrib.lookup.TextFileIdTableInitializer('init.txt')
        table = tf.contrib.lookup.HashTable(table_init, 0)
        res = table.lookup(input_tensor)
        out = tf.identity(res, name='sum')
    """
    In init.txt there are different characters on different lines.
    "character-linenumber" are the key-value pairs to initialize the hash table.
    Given a set of string characters corresponding line numbers in the init.txt will be returned.
    """

    data = np.array(['9', '7', '6', '5', '4'])
    with tf.Session(graph=out.graph) as sess:
        sess.run([tf.tables_initializer(),tf.global_variables_initializer()])
        tf_out = sess.run(out, feed_dict={input_tensor:data})
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['sum','init_all_tables'])
    

    layout = None
    target = 'llvm'
    ctx=tvm.cpu(0)
    mod, params = relay.frontend.from_tensorflow(constant_graph, layout=layout, outputs=['sum'])
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod,
                                     target=target,
                                     target_host = target,
                                     params=params)
    from tvm.contrib import graph_runtime
    m = graph_runtime.create(graph, lib, ctx)
    m.set_input(**params)
    m.set_input('input', data)
    m.run()
    tvm_out=m.get_output(0)
    print(tvm_out)
    print(tf_out)
    tvm.testing.assert_allclose(tvm_out.asnumpy(), tf_out.astype(tvm_out.dtype), rtol=1e-5)

if __name__ == "__main__":
    test_stringtovalue()
    test_valuetostring()
    test_valuetovalue()
    test_stringtostring()
    test_fromtext()

