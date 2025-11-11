import tensorflow as tf

def mse(y_true, y_pred, valid_mask):
    valid_mask = tf.cast(valid_mask, tf.float32)[..., None]  # (B, T, 1)
    diff = tf.math.squared_difference(y_true, y_pred) * valid_mask  # (B, T, 2)

    # sum over coords/time for each sample
    per_sample_denom = tf.reduce_sum(valid_mask, axis=(1, 2)) + 1e-6   # (B,)
    per_sample_loss = tf.reduce_sum(diff, axis=(1, 2)) / per_sample_denom  # (B,)
    return tf.reduce_mean(per_sample_loss)

def ade(y_true, y_pred, valid_mask):
    # average displacement error over valid steps
    valid_mask = tf.cast(valid_mask, tf.float32)[..., None]

    dist = tf.norm(y_true - y_pred, axis=-1, keepdims=True)  # (B, 80, 1)
    dist = dist * valid_mask

    per_sample_tot = tf.reduce_sum(dist, axis=(1,2))
    per_sample_count = tf.reduce_sum(valid_mask, axis=(1,2)) + 1e-6

    return tf.reduce_mean(per_sample_tot / per_sample_count)

def fde(y_true, y_pred, valid_mask):
    valid_mask = tf.cast(valid_mask, tf.float32)

    # find last valid index per sample
    time_indices = tf.range(tf.shape(valid_mask)[1])[tf.newaxis, :]  # (1, T)

    # zero out invalid
    masked_times = valid_mask * tf.cast(time_indices, tf.float32)
    last_valid_idx = tf.cast(tf.math.reduce_max(masked_times, axis=1), tf.int32)  # (B,)

    # gather last valid true/pred
    batch_idx = tf.range(tf.shape(y_true)[0], dtype=tf.int32)
    idx = tf.stack([batch_idx, last_valid_idx], axis=1)  # (B, 2)

    last_true = tf.gather_nd(y_true, idx)  # (B, 2)
    last_pred = tf.gather_nd(y_pred, idx)  # (B, 2)

    return tf.reduce_mean(tf.norm(last_true - last_pred, axis=-1))
