# Waymo Motion Prediction Model

## Project Overview

This project implements a trajectory prediction model using the Waymo Open Motion Dataset, aiming to predict an agent's future positions based on its past trajectory. The model takes the past 10 (x, y) coordinates of a single agent and predicts its next 80 positions (a future horizon of roughly 8 seconds at 10 Hz). This task is crucial for autonomous driving systems, as accurate motion forecasting of vehicles and pedestrians helps in planning and safety. The provided code serves as a baseline approach: it trains a simple deep learning model on a subset of the Waymo motion dataset and evaluates its performance using standard error metrics. The model works on individual agent tracks without any scene context or map information, demonstrating what can be achieved with purely historical trajectory data.

![Waymo Motion Visualization](.\gifs\ConvMLP\test_agent_2.gif)

Key features of the project include:

- **Data Loading & Preprocessing:** Reading Waymo Open Motion Dataset TFRecord files and preparing input-output trajectory pairs, with coordinate normalization (translation and rotation) for easier learning.
- **Model Architecture:** A lightweight convolutional neural network (Conv-MLP) that encodes past trajectory data and outputs future trajectory predictions.
- **Training Pipeline:** Code to train the model from scratch, including train/validation splitting, loss computation, and periodic logging.
- **Evaluation Metrics:** Calculation of Average Displacement Error (ADE) and Final Displacement Error (FDE) for quantitative evaluation of predictions.
- **Visualization:** Generation of animated GIFs to visually compare predicted trajectories against ground truth future trajectories.

---

![Waymo Motion Visualization](.\gifs\ConvMLP\test_agent_3.gif)
![Waymo Motion Visualization](.\gifs\ConvMLP\test_agent_4.gif)

![LossMetric](.\images\ConvMLP\losstrain.gif)
![ADEMetric](.\images\ConvMLP\ADEtrain.gif)
![FDEMetric](.\images\ConvMLP\FDEtrain.gif)


## Model Architecture

### ConvMLP Model

The core model (ConvMLP) is a simple 1D Convolution + MLP neural network implemented in TensorFlow/Keras. It consists of:

- **Convolutional Encoder:** Two 1D convolutional layers with causal padding (kernel size 3, 64 filters each) that process the sequence of past coordinates. These Conv1D layers act over the time dimension (the 10 past steps) to extract motion features (such as velocity or acceleration cues) from the sequence.
- **Flatten Layer:** The output of the conv layers (shape 10×64 per sequence) is flattened into a single vector representation.
- **MLP Decoder:** A fully connected network that takes the flattened encoding and produces the future trajectory. In this implementation, it has one hidden Dense layer of size 128 with ReLU activation, followed by a final Dense layer that outputs FUTURE_STEPS * 2 values (here 80*2 = 160 values). These 160 outputs are then reshaped into an 80×2 sequence representing predicted future (x, y) coordinates.

**Input/Output:**
- Input: shape (10, 2)
- Output: shape (80, 2)

Internally, the model uses causal convolutions, ensuring that at time t it only sees data from times ≤ t.

---

## Dataset Requirements & Preprocessing

### Waymo Open Motion Dataset

Download the Waymo Open Motion Dataset from the official site. TFRecords should be placed in:

./training_data/ # training + validation data
./test_data/ # testing data

### Preprocessing (dataLoader.py)

- Loads TFRecords with `tf.data.TFRecordDataset`
- Extracts:
  - `state/past/x`, `state/past/y` (length 10)
  - `state/future/x`, `state/future/y` (length 80)
  - `state/tracks_to_predict`
- Generates one training sample per agent marked for prediction
- **Normalization:**
  - Translate last past point → (0,0)
  - Rotate coordinates so final heading aligns with +x axis
- Applies `future_valid` mask
- Splits 90/10 into train/val
- Batches and prefetches

Each sample:

- Past trajectory: (10, 2)
- Future trajectory: (80, 2)
- Valid mask: (80,)

---

Directory structure:

```
Waymo-Motion-Prediction/
├── ConvMLP.py
├── dataLoader.py
├── train_model.py
├── evaluate.py
├── metrics.py
├── visualize.py
├── training_data/
└── test_data/
```

## Usage Instructions
### Training
Run:

```
python train_model.py
```

Loads + splits dataset

Trains ConvMLP for default epochs

Computes Loss (MSE), ADE, FDE

Saves model to trained_models/ConvMLP.keras

### TensorBoard:

```
tensorboard --logdir logs/
```

### Evaluation
Run evaluation script:

```
python evaluate.py
```

At the prompt, enter:

test

Outputs per-batch + averaged metrics: Loss, ADE, FDE

## Known Limitations
No use of map or scene context

Single-agent prediction

Strong straight-line bias

Basic architecture (no LSTM/GRU/Transformer)

Small training subset

Deterministic single-output trajectory

## Future Work
Add map + road graph context

Include neighboring agents

Use LSTMs, GRUs, Transformers, or GNNs

Multi-modal outputs (probabilistic trajectories)

Better loss functions and scenario filtering

Use more of the dataset/Use more compute in prep for larger models(Current focus, will use Google Compute Engine to use WOMD without having to download)