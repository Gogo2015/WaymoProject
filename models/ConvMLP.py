import tensorflow as tf


PAST_STEPS = 10
FUTURE_STEPS = 80

class ConvMLP(tf.keras.Model):
    def __init__(self, activation="relu", **kwargs):
        super().__init__(**kwargs)

        self.input = tf.keras.Input(shape=(PAST_STEPS, 2), name="past_xy")

        #Conv Layers
        self.conv_layer_1 = tf.keras.layers.Conv1D(64, kernel_size=3, padding="causal", activation=activation)
        self.conv_layer_2 = tf.keras.layers.Conv1D(64, kernel_size=3, padding="causal", activation=activation)

        self.flatten = tf.keras.layers.Flatten()

        #Dense Layers
        self.dense_1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense_2 = tf.keras.layers.Dense(FUTURE_STEPS * 2)

        self.output = tf.keras.layers.Reshape((FUTURE_STEPS, 2), name="future_xy")


    def call(self, input):
        """
        Input: (B, 10, 2)
        Output: (B, 80, 2)
        """

        inputs = self.input(input)

        #Convolutional Encoder
        x = self.conv_layer_1(inputs)
        x = self.conv_layer_2(x)

        x = self.flatten(x)

        #MLP/Dense Decoder
        x = self.dense_1(x)
        x = self.dense_2(x)

        outputs = self.output(x)

        return outputs