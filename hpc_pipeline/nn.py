"""
Cannot run multiprocess...
All imports should not be in global scope. If they are, then each time I spawn a new process (for game) things get
messed up.
"""
import numpy as np
from utils import MAX_BATCH_SIZE, INPUT_SHAPE, LEARNING_RATE
import sys

#INPUT_SHAPE = (4, 4, 1)  # TODO: score??
CONV_SIZE = 32
FULLY_CON_SIZE = 64


class NeuralNetwork:

    def __init__(self):
        print("Building model")
        sys.stderr.flush()
        sys.stdout.flush()
        self.model = NeuralNetwork.build_nn()
        print("Built model")
        sys.stderr.flush()
        sys.stdout.flush()

    def preproccesing(self, data):
        x = np.expand_dims(data, axis=3)
        return x

    def query_model(self, data):
        #X = self.preproccesing(data)
        X = data
        prediction = self.model.predict(X, batch_size=MAX_BATCH_SIZE)
        # print(prediction)
        return prediction

    def fit(self, x, y, nbr_epochs=1):
        # Lets just retrain always...no evaluation
        # print(y)
        self.model.fit(x=x, y=y, batch_size=2048, epochs=nbr_epochs)  # TODO: batch_size?

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def save(self, path):
        self.model.save(path)

    @staticmethod
    def build_nn():
        from keras.layers import Input
        from keras.models import Model
        from keras.optimizers import Adam
        """
        Construct and compile new model
        """
        global INPUT_SHAPE
        x = Input(shape=INPUT_SHAPE)
        conv_out = x  # Two convolutional layers
        for i in range(1):
            conv_out = NeuralNetwork.conv_layer(conv_out)
        residual_output = conv_out  # Residual layers
        for i in range(10):  # Add ten residual layers
            residual_output = NeuralNetwork.res_layer(residual_output)
        policy_out = NeuralNetwork.policy_head(residual_output)
        value_out = NeuralNetwork.value_head(residual_output)

        model = Model(inputs=x, outputs=(policy_out, value_out))  #
        model.compile(loss=["categorical_crossentropy", "mean_squared_error"], loss_weights=[1.0, 1.0],
                      optimizer=Adam(lr=LEARNING_RATE), metrics=['accuracy'])

        return model

    @staticmethod
    def res_layer(y):
        from keras.layers import Conv2D, Activation, Add, BatchNormalization
        global CONV_SIZE
        h = Conv2D(CONV_SIZE, (2, 2), strides=(1, 1), padding="same")(y)
        h = BatchNormalization()(h)
        h = Activation("relu")(h)
        h = Conv2D(CONV_SIZE, (2, 2), strides=(1, 1), padding="same")(h)
        h = BatchNormalization()(h)
        h = Add()([h, y])  # Skip connection from previous layer that adds the input to the block
        return Activation("relu")(h)

    @staticmethod
    def conv_layer(y):
        from keras.layers import Conv2D, Activation, BatchNormalization
        global CONV_SIZE
        h = Conv2D(CONV_SIZE, (2, 2), strides=(1, 1))(y)
        h = BatchNormalization()(h)
        return Activation('relu')(h)

    @staticmethod
    def policy_head(y):
        from keras.layers import Conv2D, Activation, Dense, BatchNormalization, Flatten
        h = Conv2D(2, (1, 1), strides=(1, 1))(y)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = Flatten()(h)
        h = Dense(FULLY_CON_SIZE)(h)
        h = Activation('tanh')(h)
        h = Dense(4)(h)  # Four possible move probabilities
        return Activation('softmax', name='policy_out')(h)  # Four possible move probabilities

    @staticmethod
    def value_head(y):
        from keras.layers import Conv2D, Activation, Dense, BatchNormalization, Flatten
        h = Conv2D(1, (1, 1), strides=(1, 1))(y)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = Dense(FULLY_CON_SIZE)(h)

        h = Activation('relu')(h)
        h = Flatten()(h)
        h = Dense(1)(h)
        return Activation('sigmoid', name='value_out')(h)  # Score output


if __name__ == '__main__':
    model = NeuralNetwork()
    print('-------------')

