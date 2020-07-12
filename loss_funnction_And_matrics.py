import tensorflow as tf

####---Loss
@tf.function
def macro_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.

    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    cost = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost) # average on all labels
    return macro_cost


###Matrics
@tf.function
def macro_f1(y, y_hat, thresh=0.5):
    """Compute the macro F1-score on a batch of observations (average F1 across labels)

    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive

    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1



@tf.function
def Weighted_BCTL(y_true, y_pred):

    # Manually calculate the weighted cross entropy.
    # Formula is qz * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
    # where z are labels, x is logits, and q is the weight.
    # Since the values passed are from sigmoid (assuming in this case)
    # sigmoid(x) will be replaced by y_pred
    # qz * -log(sigmoid(x)) 1e-6 is added as an epsilon to stop passing a zero into the log

    ##get the positive labels

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred , tf.float32)

    P=tf.cast(tf.math.count_nonzero(y_true), tf.float32)
    N=tf.cast(len(tf.where(y_true==0)),tf.float32)

    BP1=P+N/P
    BP=tf.cast(BP1,tf.float32)

    BN=N+P/N
    BN=tf.cast(BN,tf.float32)


    x_1 =BP*(y_true * -tf.math.log(y_pred + 1e-6))
    x_2 =BN*((1 - y_true) * -tf.math.log(1 - y_pred + 1e-6))

    cost=tf.add(x_1, x_2)
    cost_a=tf.reduce_mean(cost)
    return cost_a

