import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0 = all, 1 = info, 2 = warning, 3 = error

import tensorflow as tf
from dataLoader import get_data
from metrics import mse, ade, masked_fde

# import models
from models.ConvMLP import ConvMLP

MODELS = [
    ConvMLP
]

DATA_DIR = "./data"

PAST_STEPS = 10
FUTURE_STEPS = 80

@tf.function
def train_step(model, optimizer, past, future, valid):
    with tf.GradientTape() as tape:
        pred = model(past, training=True)
        loss = mse(future, pred, valid)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    batch_ade = ade(future, pred, valid)
    batch_fde = masked_fde(future, pred, valid)
    
    return loss, batch_ade, batch_fde

def train_model(model_to_train, BATCH_SIZE=64, epochs=10):
    #get dataset
    ds = get_data(DATA_DIR, BATCH_SIZE)

    #build_model
    model = model_to_train(PAST_STEPS, FUTURE_STEPS)
    model.build(input_shape=(None, PAST_STEPS, 2))

    optimizer = tf.keras.optimizers.Adam(1e-3)

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_ade = 0.0
        epoch_fde = 0.0
        num_batches = 0

        for past, future, valid in ds:
            batch_loss, batch_ade, batch_fde = train_step(model, optimizer, past, future, valid)

            # add batch metrics
            epoch_loss += float(batch_loss.numpy())
            epoch_ade  += float(batch_ade.numpy())
            epoch_fde  += float(batch_fde.numpy())
            num_batches += 1

        # average over the epoch
        epoch_loss /= num_batches
        epoch_ade  /= num_batches
        epoch_fde  /= num_batches

        print(f"epoch {epoch+1}: loss {epoch_loss:.4f} | ADE {epoch_ade:.4f} | FDE {epoch_fde:.4f}")

        model_name = model_to_train.__name__
        os.makedirs("trained_models", exist_ok=True)
        save_path = f"trained_models/{model_name}.keras"

        model.save(save_path)

if __name__ == "__main__":
    for model in MODELS:
        train_model(model)