import tensorflow as tf

def model(input_tensor, num_features, num_separations):
    separator_weight = tf.get_variable('separator_weight', (num_features, num_separations), initializer=tf.random_normal_initializer())
    fitter_weight = tf.get_variable('fitter_weight', (num_features, num_separations), initializer=tf.random_normal_initializer())
    separator_tensor = tf.matmul(input_tensor, separator_weight) # None * num_separations, (0-1)
    separator_tensor = tf.nn.softmax(separator_tensor) # None * num_separations, row sum = 1
    fitter_tensor = tf.matmul(input_tensor, fitter_weight)
    fitter_tensor = tf.sigmoid(fitter_tensor) # None * num_separations, (0-1)
    result_tensor = tf.multiply(separator_tensor, fitter_tensor) # Element-wise multiplication
    result_tensor = tf.reduce_sum(result_tensor, 1, keepdims=True) # Equivalent of row-wise dot product
    return result_tensor

