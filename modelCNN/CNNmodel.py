import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0 = all, 1 = info, 2 = warning, 3 = error
import tensorflow as tf


PAST_STEPS = 10
FUTURE_STEPS = 80

def build_model():
    """
    Input: (B, 10, 2)
    Output: (B, 80, 2)
    """

    inputs = tf.keras.Input(shape=(PAST_STEPS, 2), name="past_xy")

    #Model Layers
    x = tf.keras.layers.Conv1D(64, kernel_size=3, padding="causal", activation="relu")(inputs)
    x = tf.keras.layers.Conv1D(64, kernel_size=3, padding="causal", activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(FUTURE_STEPS * 2)(x)

    outputs = tf.keras.layers.Reshape((FUTURE_STEPS, 2), name="future_xy")(x)

    model = tf.keras.Model(inputs, outputs, name="waymo_mlp_conv")
    return model


def masked_mse(y_true, y_pred, valid_mask):
    """
    y_true:    (B, 80, 2)
    y_pred:    (B, 80, 2)
    valid_mask:(B, 80)
    """
    valid_mask = tf.cast(valid_mask, tf.float32)
    valid_mask = tf.expand_dims(valid_mask, -1)     # (B, 80, 1)
    diff = tf.math.squared_difference(y_true, y_pred) * valid_mask
    denom = tf.reduce_sum(valid_mask) + 1e-6
    return tf.reduce_sum(diff) / denom


if __name__ == "__main__":
    from dataLoader import get_data

    ds = get_data(BATCH_SIZE=64)
    model = build_model()
    model.summary()

    # custom train step so we can use the mask
    optimizer = tf.keras.optimizers.Adam(1e-3)

    for step, (past, future, valid) in enumerate(ds):
        with tf.GradientTape() as tape:
            pred = model(past, training=True)
            loss = masked_mse(future, pred, valid)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 20 == 0:
            print(f"step {step}: loss {loss.numpy():.4f}")