import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0 = all, 1 = info, 2 = warning, 3 = error
import tensorflow as tf

from _dataLoader import transform

DATA_DIR = "../data"
PAST_STEPS = 10
FUTURE_STEPS = 80
BATCH_SIZE = 64

state_features = {
    'state/future/valid':
        tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    'state/future/x':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/y':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/past/valid':
        tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    'state/past/x':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/y':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/tracks_to_predict':
        tf.io.FixedLenFeature([128], tf.int64, default_value=None),
}

features_description = {}
features_description.update(state_features)


def transform(single_example, features):
    """
    Input: 
    single_example: single example gotten from flat_map function

    Output:
    transformed: data split into (past, future, future_valid)
        - past: [N, 10, 2]
        - future: [N, 80, 2]

    """

    #Parse example
    parsed = tf.io.parse_single_example(single_example, features)

    #Retrieve features
    past_x, past_y = parsed['state/past/x'], parsed['state/past/y']
    future_x, future_y = parsed['state/future/x'], parsed['state/future/y']
    future_valid = parsed['state/future/valid']
    tracks = parsed['state/tracks_to_predict']

    #Get mask for agents that we want to predict
    mask = tracks > 0

    #Apply mask
    past_x = tf.boolean_mask(past_x, mask)
    past_y = tf.boolean_mask(past_y, mask)
    future_x = tf.boolean_mask(future_x, mask)
    future_y = tf.boolean_mask(future_y, mask)
    future_valid = tf.boolean_mask(future_valid, mask)

    #If zero tensor, return empty data
    if tf.shape(past_x)[0] == 0:
            return tf.data.Dataset.from_tensor_slices(
                (tf.zeros((0, PAST_STEPS, 2), tf.float32),
                 tf.zeros((0, FUTURE_STEPS, 2), tf.float32),
                 tf.zeros((0, FUTURE_STEPS), tf.int64))
            )
    
    #Translate agent's position so last point at origin
    anchor_x, anchor_y = past_x[:, -1], past_y[:, -1]
    past_x -= anchor_x[:, None]
    past_y -= anchor_y[:, None]
    fut_x -= anchor_x[:, None]
    fut_y -= anchor_y[:, None]

    #Rotate agent so motion is along +x axis
    dx = past_x[:, -1] - past_x[:, -2]
    dy = past_y[:, -1] - past_y[:, -2]
    yaw = tf.atan2(dy, dx)
    c, s = tf.cos(-yaw)[:, None], tf.sin(-yaw)[:, None]

    def rot(x, y):
        return x * c - y * s, x * s + y * c
    
    past_x, past_y = rot(past_x, past_y)
    future_x, future_y = rot(future_x, future_y)

    #Create tensors for returning
    past = tf.stack([past_x, past_y], axis=-1)
    future = tf.stack([future_x, future_y], axis=-1)
    return tf.data.Dataset.from_tensor_slices((past, future, future_valid))

def get_data():
    #Get dataset of all files in data folder
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)]
    dataset = tf.data.TFRecordDataset(files)

    dataset = dataset.flat_map(lambda x: transform(x, features_description))

    dataset = dataset.shuffle(2048).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return dataset




