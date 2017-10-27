import tensorflow as tf

def build_mlp(
        input_placeholder,
        output_size,
        scope,
        n_layers=2,
        size=64,
        activation=tf.tanh,
        output_activation=None
        ):

    with tf.variable_scope(scope):
        prev_layer = tf.layers.dense(inputs=input_placeholder, units=size, activation=activation)
        for _ in range(n_layers):
            prev_layer = tf.layers.dense(inputs=prev_layer, units=size, activation=activation)
        return tf.layers.dense(inputs=prev_layer, units=output_size, activation=output_activation)
