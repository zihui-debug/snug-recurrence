import tensorflow as tf
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat
from tensorflow.python.framework import ops



import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import warnings
warnings.filterwarnings('ignore')

model = 'models/SNUG-Tshirt/saved_model.pb'
with tf.Session(graph=ops.Graph()) as sess:
    with tf.gfile.GFile(model, "rb") as f:
        data = compat.as_bytes(f.read())
        sm = saved_model_pb2.SavedModel()
        sm.ParseFromString(data)
        g_in = tf.import_graph_def(sm.meta_graphs[0].graph_def)
        train_writer = tf.summary.FileWriter("./log")
        train_writer.add_graph(sess.graph)
        train_writer.flush()
        train_writer.close()