import tensorflow as tf
import tensorflow.keras.layers as layers

x = tf.constant(3.0)
print(x)

with tf.GradientTape() as g:
    g.watch(x)
    y = x*x
    dy = g.gradient(y,x)
print(dy)

X = tf.zeros([1,2,3], tf.float32)
print(X.shape)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.1, epsilon=1e-7)