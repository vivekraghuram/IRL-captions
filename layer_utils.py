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


def affine_transform(input_placeholder, output_size, scope):
    with tf.variable_scope(scope):
        out = tf.layers.dense(inputs=input_placeholder, units=output_size, activation=None)
    return out


def difference_over_time(xs, scope):
    with tf.variable_scope(scope):
        time_dim = tf.shape(xs)[1]
        initial = tf.expand_dims(xs[:, 0, :], dim=1)
        diff = xs[:, 1: time_dim, :] - xs[:, 0:time_dim - 1, :]
        return tf.concat([initial, diff], axis=1)
