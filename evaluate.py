import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0 = all, 1 = info, 2 = warning, 3 = error

import tensorflow as tf
import numpy as np
from dataLoader import get_data
from models.ConvMLP import ConvMLP
from visualize import visualize_trajectory

PAST_STEPS = 10
FUTURE_STEPS = 80
DATA_DIR = "./test_data"

# Load test dataset
test_ds = get_data(DATA_DIR, BATCH_SIZE=1, training=False)

# Load trained model
model_path = "trained_models/ConvMLP.keras"
model = tf.keras.models.load_model(model_path)

# Generate 5 gifs
for i, (past, future, valid) in enumerate(test_ds.take(5)):
    pred_future = model(past, training=False).numpy()[0]      # (FUTURE_STEPS, 2)
    past_np = past.numpy()[0]
    true_future_np = future.numpy()[0]
    visualize_trajectory(past_np, true_future_np, pred_future, f"test_agent_{i+1}.gif")
