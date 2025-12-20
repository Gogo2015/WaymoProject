import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0 = all, 1 = info, 2 = warning, 3 = error

import datetime
import tensorflow as tf
from dataLoader import get_data
from metrics import mse, ade, fde
from losses import multimodal_loss

# import models
from models.ConvMLP import ConvMLP
from models.MultiModalConvMLP import MultiModalConvMLP

LOG_DIR = "logs"

MODELS = [
    #ConvMLP,
    MultiModalConvMLP
]

DATA_DIR = "./training_data"

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
    batch_fde = fde(future, pred, valid)

    return loss, batch_ade, batch_fde

@tf.function
def train_step_multimodal(model, optimizer, past, future, valid):
    with tf.GradientTape() as tape:
        pred_trajectories, confidences = model(past, training=True)
        loss, control_loss, intent_loss = multimodal_loss(future, pred_trajectories, confidences, valid)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss, control_loss, intent_loss

def train_model(model_to_train, BATCH_SIZE=64, epochs=10):
    #get dataset
    train_ds, val_ds = get_data(DATA_DIR, BATCH_SIZE)

    #build_model
    is_multimodal = (model_to_train == MultiModalConvMLP)
    
    if is_multimodal:
        model = model_to_train(num_modes=6, past_steps=PAST_STEPS, future_steps=FUTURE_STEPS)
    else:
        model = model_to_train(PAST_STEPS, FUTURE_STEPS)
    
    model.build(input_shape=(None, PAST_STEPS, 2))

    #Initialize for TensorBoard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(LOG_DIR, current_time, "train")
    val_log_dir = os.path.join(LOG_DIR, current_time, "val")

    train_writer = tf.summary.create_file_writer(train_log_dir)
    val_writer = tf.summary.create_file_writer(val_log_dir)

    optimizer = tf.keras.optimizers.Adam(1e-3)

    for epoch in range(epochs):
        epoch_loss, epoch_ade, epoch_fde = 0.0, 0.0, 0.0
        num_batches = 0

        for past, future, valid in train_ds:
            if is_multimodal:
                batch_loss, batch_ade, batch_fde = train_step_multimodal(model, optimizer, past, future, valid)
            else:
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

        # write train metrics
        with train_writer.as_default():
            tf.summary.scalar("loss", epoch_loss, step=epoch)
            tf.summary.scalar("ADE", epoch_ade, step=epoch)
            tf.summary.scalar("FDE", epoch_fde, step=epoch)

        if is_multimodal:
            print(f"epoch {epoch+1}: total {epoch_loss:.4f} | control {epoch_ade:.4f} | intent {epoch_fde:.4f}")
        else:
            print(f"epoch {epoch+1}: loss {epoch_loss:.4f} | ADE {epoch_ade:.4f} | FDE {epoch_fde:.4f}")

        val_loss, val_ade, val_fde = 0.0, 0.0, 0.0
        val_batches = 0

        for past, future, valid in val_ds:
            if is_multimodal:
                pred_traj, conf = model(past, training=False)
                loss, control, intent = multimodal_loss(future, pred_traj, conf, valid)
                val_loss += float(loss.numpy())
                val_ade += float(control.numpy())
                val_fde += float(intent.numpy())
            else:
                pred = model(past, training=False)
                val_loss += float(mse(future, pred, valid).numpy())
                val_ade  += float(ade(future, pred, valid).numpy())
                val_fde  += float(fde(future, pred, valid).numpy())
            
            val_batches += 1

        val_loss /= val_batches
        val_ade  /= val_batches
        val_fde  /= val_batches

        with val_writer.as_default():
            tf.summary.scalar("loss", val_loss, step=epoch)
            tf.summary.scalar("ADE", val_ade, step=epoch)
            tf.summary.scalar("FDE", val_fde, step=epoch)

        if is_multimodal:
            print(f"Val: total {val_loss:.4f} | control {val_ade:.4f} | intent {val_fde:.4f}")
        else:
            print(f"Val: loss {val_loss:.4f} | ADE {val_ade:.4f} | FDE {val_fde:.4f}")

        model_name = model_to_train.__name__
        os.makedirs("trained_models", exist_ok=True)
        save_path = f"trained_models/{model_name}.keras"

        model.save(save_path)

if __name__ == "__main__":
    for model in MODELS:
        train_model(model)