import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

dtype = tf.float32
batch_size = 1
seq_length = 2
m = 3
eps = 0

u = tf.cast( tf.range( seq_length + 2 ), dtype=dtype) + 0.5

init =  tf.zeros((batch_size, m), dtype)

j = tf.slice( u, [0], [ seq_length+1 ] )

# gbk_t = tf.constant([ [0.1, 10, 1], [0.5, 1, 0.5], [0.3, 0.1, 0.1] ])
gbk_t = tf.constant([ [0.1, 0.5, 0.3, 10, 1, 0.1, 1, 0.5, 0.1] ])
g_t, b_t, k_t = tf.split( gbk_t, num_or_size_splits=3, axis=1 )


mu_t = 0 + tf.math.softplus(k_t)
# print(mu_t)

sig_t = tf.math.softplus(b_t) + eps
# print(sig_t)

g_t = tf.nn.softmax( g_t, axis=1 ) + eps
# print(g_t)
print(j)
x = (j-tf.expand_dims(mu_t, -1))/ tf.expand_dims(sig_t, -1)
print(x)
p = tf.nn.sigmoid(x)
print(p)
#   phi_t = tf.expand_dims(g_t, -1) * tf.nn.sigmoid( x )
phi_t = tf.expand_dims(g_t, -1) * p

# phi_t = x/( tf.expand_dims(sig_t, -1)*x*x )
alpha_t = tf.reduce_sum( phi_t, 1 )

# discretize
# print(alpha_t)
a = tf.slice( alpha_t, [0, 1], [batch_size, seq_length] )
b = tf.slice( alpha_t, [0, 0], [batch_size, seq_length] )
alpha_t = a-b

print(alpha_t)

alpha_t = _maybe_mask_score(alpha_t, )