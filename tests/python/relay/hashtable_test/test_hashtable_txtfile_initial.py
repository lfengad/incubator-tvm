import numpy as np
import tensorflow as tf
import tvm
import datetime
from tvm import relay
from tensorflow.python.framework import graph_util

import tvm.relay.testing.tf as tf_testing

#tf.compat.v1.disable_eager_execution()

g=tf.Graph()

with g.as_default(): 

#tf.compat.v1.disable_eager_execution()
    input_tensor = tf.placeholder(tf.string,shape=(5,), name='input-0')
    #input_tensor = tf.placeholder(tf.int64, shape=(2,) name='input-0')

#input_tensor = tf.constant([1, 2, 3], dtype=tf.int64)
    
    #keys = tf.constant(np.array(['149675741656895691659826758', '23461658765961956198659814658946754591495', '3', '4', '5', '6']),  dtype=tf.string, name='keys')
    #keys = tf.constant(np.array([1, 2, 3, 4, 5, 6]),  dtype=tf.int32, name='keys')
    #values = tf.constant(np.array(['1', '5', '6', '7', '8', '9']),  dtype=tf.string, name='values')

    table_init =tf.contrib.lookup.TextFileIdTableInitializer('init.txt')
    table = tf.contrib.lookup.HashTable(
        #tf.contrib.lookup.KeyValueTensorInitializer(keys, values),
        table_init,
        2)
    
    
    out = table.lookup(input_tensor, name = 'sum')
    #out1 = table.lookup(input_tensor)
    #out2 = table.lookup(input_tensor)
    
    """
    out0 = input_tensor
    out1 = input_tensor
    out2 = input_tensor
    """
    #sum_tmp = tf.add(out0, out1, name='sum_temp')
    #out = tf.add(sum_tmp, out2, name='sum')
    

LOGDIR='./try_dir'
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(g)
#data = np.random.uniform(1, 9, size=(128,)).astype("string")
data = np.array(['1','2','3','4','5'])
#data = np.array(['1234567890100e3290479757143','24326846236748067462346312','3','4','5'])
#print(data.dtype)
#print(np.dtype(object))
#print(data)

with tf.Session(graph=out.graph) as sess:
    g0 = sess.graph_def 
    #table.init.run()
    sess.run(tf.tables_initializer())
    g1 = sess.graph_def 
    sess.run(tf.global_variables_initializer())
    g2 = sess.graph_def 
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['sum','init_all_tables'])
    #constant_graph = sess.graph_def
    train_writer = tf.summary.FileWriter('tf_graph', sess.graph)
    print(sess.run(out, feed_dict={input_tensor:data}))
    with tf.gfile.FastGFile('./model.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())
    start_time = datetime.datetime.now()
    for i in range (20):
        sess.run(out, feed_dict={input_tensor:data})
    end_time = datetime.datetime.now()
    t_tf = end_time - start_time
    

with tf.gfile.GFile('./model.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    coming_string = f.read()
    graph_def.ParseFromString(coming_string)
    writefile = open("graph_def","w")
    print(graph_def, file=writefile)
    graph = tf.import_graph_def(graph_def, name='')
    # Call the utility to import the graph definition into default graph.
    graph_def = tf_testing.ProcessGraphDefParam(graph_def)
    # Add shapes to the graph.
    #with tf.Session() as sess:
    #    graph_def = tf_testing.AddShapesToGraphDef(sess, 'sum')

layout = None
target = 'llvm'
target_host = 'llvm'
ctx=tvm.cpu(0)



mod, params = relay.frontend.from_tensorflow(graph_def, layout=layout, outputs=['sum'])


writefile__ = open("mod_def_tvm","w")
print(mod.astext(show_meta_data=False), file=writefile__)


with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(mod,
                                     target=target,
                                     target_host = target_host,
                                     params=params)

writefile_ = open("graph_def_tvm","w")
print(graph, file=writefile_)

from tvm.contrib import graph_runtime

dtype = 'uint64'
m = graph_runtime.create(graph, lib, ctx)

#data = np.random.uniform(1,4, size=(2,2)).astype("int64")
#data = np.array([1,2])

print(params)
m.set_input(**params)
m.set_input('input-0', data)

m.run()

start_time = datetime.datetime.now()
for i in range (20):
    m.set_input('input-0', data)
    m.run()
end_time = datetime.datetime.now()
t_tvm = end_time - start_time

print("tftime is {} while tvmtime is {}".format(t_tf, t_tvm))

tvm_output=m.get_output(0)

print(tvm_output)

"""
with tf.Session() as sess:
    LOGDIR='./check_dir'
    train_writer = tf.summary.FileWriter(LOGDIR)
    train_writer.add_graph(sess.graph)
    table.init.run()
    for i in range(5):
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        print(sess.run(out, feed_dict={input_tensor : i},options=run_options, run_metadata=run_metadata))
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
    train_writer.close()
"""
