import tensorflow as tf

def model(input_tensor, num_features, num_separations):
    separator_weight = tf.get_variable('separator_weight', (num_features, num_separations), initializer=tf.random_normal_initializer())
    fitter_weight = tf.get_variable('fitter_weight', (num_features, num_separations), initializer=tf.random_normal_initializer())
    output_weight = tf.get_variable('output_weight', (num_separations*2, 1), initializer=tf.random_normal_initializer())
    output_bias = tf.get_variable('output_bias', (1), initializer=tf.random_normal_initializer())
    separator_tensor = tf.matmul(input_tensor, separator_weight) # None * num_separations, (0-1)
    separator_tensor = tf.nn.softmax(separator_tensor) # None * num_separations, row sum = 1
    fitter_tensor = tf.matmul(input_tensor, fitter_weight)
    fitter_tensor = tf.sigmoid(fitter_tensor) # None * num_separations, (0-1)
    new_features = tf.concat([fitter_tensor, separator_tensor], axis=1)
    result_tensor = tf.sigmoid(tf.matmul(new_features, output_weight) + output_bias)
    return result_tensor
