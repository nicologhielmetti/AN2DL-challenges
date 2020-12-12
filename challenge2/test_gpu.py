import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
