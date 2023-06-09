import tensorflow as tf

# This is so we don't have to change all the legacy conversion calls to casts.
tf.to_float = lambda x: tf.cast(x, tf.float32)
tf.to_int32 = lambda x: tf.cast(x, tf.int32)