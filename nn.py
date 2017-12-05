import numpy as np
from numba import jit

import json
import os

MAX_POWER = 20  # Maximum power of two present in the board
INPUT_SHAPE = (MAX_POWER, 4, 4)
SCORE_SCALER = 100000
CONV_SIZE = 32
FULLY_CON_SIZE = 64
REG_COEF = 0.001/2

"""
TRAINING PROCEDURE SHOULD BE AS FOLLOWS:
1) Play N amount of games to create the training set
    - Store the game state of each move
    - Store the probabilities for each move
    - Store the end result of the game (total moves or end score)
2) Sample x amount of positions from the last N*20 games
3) Retrain the network based on the saved states
    - A custom loss function is used which takes into account the move probabilities and
      predicted end result (total moves or end score)
4) Play y games and see whether the new neural network achieves better results 
    - The new game results must be better by z percent so that 
      the new neural network is chosen

For ALPHA GO ZERO the values were:
N = 25000
x = 2048
y = 400

"""

@jit
def scale_score(score):
    return min(score/SCORE_SCALER, 1)
@jit
def descale_score(score_out):
    return score_out*SCORE_SCALER

@jit
def convert_game_to_bin(game):
    game = np.array(game)
    bin_rep = np.zeros(INPUT_SHAPE, dtype=bool)
    for i in range(MAX_POWER):  # For each power of twp
        bin_rep[i][game == i] = 1  # Assign True value
    return np.array(bin_rep)

@jit
def shuffle_train_data(x, y1, y2):
    p = np.random.permutation(x.shape[0])
    return x[p], y1[p], y2[p]

# Not used currently
def read_from_file():
    states_train = None
    scores_train = None
    probs_train = None

    dir_name = "games"
    total_games = len([name for name in os.listdir(dir_name) if os.path.isfile(dir_name + "/" + name)])

    ## Training data
    for i in range(1, total_games):
        with open("games/%d.txt" % i, "r") as train_file:
            train_dict = json.loads(train_file.read())
        file_states = np.asarray([convert_game_to_bin(x) for x in np.asarray(train_dict['states'])])
        score_val = scale_score(train_dict['score'])
        file_scores = np.asarray([score_val] * file_states.shape[0])
        probs_train_n = np.asarray(train_dict['probabilities'], dtype=float)
        probs_train_p = probs_train_n / np.sum(probs_train_n, axis=1, keepdims=True)

        if states_train is None:
            states_train = file_states
            scores_train = file_scores
            probs_train = probs_train_p
        else:
            states_train = np.concatenate([states_train, file_states])
            scores_train = np.concatenate([scores_train, file_scores])
            probs_train = np.concatenate([probs_train, probs_train_p])

    ## Test data
    with open("games/%d.txt" % total_games, "r") as train_file:
        train_dict = json.loads(train_file.read())
        file_states = np.asarray([convert_game_to_bin(x) for x in np.asarray(train_dict['states'])])
        score_val = train_dict['score']
        file_scores = np.asarray([score_val] * file_states.shape[0])
        probs_train_n = np.asarray(train_dict['probabilities'], dtype=float)
        probs_train_p = probs_train_n / np.sum(probs_train_n, axis=1, keepdims=True)

        states_test = file_states
        scores_test = file_scores
        probs_test = probs_train_p


def init_model():
    """
    Construct and compile new model
    """
    global INPUT_SHAPE
    x = Input(shape=INPUT_SHAPE)
    conv_out = x  # Two convolutional layers
    for i in range(1):
        conv_out = conv_layer(conv_out)
    residual_output = conv_out  # Residual layers
    for i in range(3):  # Add ten residual layers
        residual_output = res_layer(residual_output)
    policy_out = policy_head(residual_output)
    value_out = value_head(residual_output)

    model = Model(inputs=x, outputs=(policy_out, value_out))  #
    model.compile(loss=["mean_squared_error", "mean_squared_error"], optimizer=Adam(), metrics=['accuracy'])

    return model


def res_layer(y):
    global CONV_SIZE
    h = Conv2D(CONV_SIZE, (2, 2), strides=(1, 1), padding="same")(y)
    h = BatchNormalization()(h)
    h = Activation("relu")(h)
    h = Conv2D(CONV_SIZE, (2, 2), strides=(1, 1), padding="same")(h)
    h = BatchNormalization()(h)
    h = Add()([h, y])  # Skip connection from previous layer that adds the input to the block
    return Activation("relu")(h)

def conv_layer(y):
    global CONV_SIZE
    h = Conv2D(CONV_SIZE, (2, 2), strides=(1, 1))(y)
    h = BatchNormalization()(h)
    return Activation('relu')(h)

def policy_head(y):
    h = Conv2D(2, (1, 1), strides=(1, 1))(y)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    h = Flatten()(h)
    h = Dense(FULLY_CON_SIZE)(h)
    h = Activation('tanh')(h)
    h = Dense(4)(h)  # Four possible move probabilities
    return Activation('softmax', name='policy_out')(h)  # Four possible move probabilities

def value_head(y):
    h = Conv2D(1, (1, 1), strides=(1, 1))(y)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    h = Dense(FULLY_CON_SIZE)(h)

    h = Activation('relu')(h)
    h = Flatten()(h)
    h = Dense(1)(h)
    return Activation('sigmoid', name='value_out')(h) # Score output

#  Handles storing training data and sampling from it
class TrainingSet:
    def __init__(self, score_scaler=100000):
        self.max_games = 50  # Maximum running games
        self.cur_games = 0  # Current amount of games in training set
        self.counts = []  # Array of the amount of saved states in each game
        self.states = None  # All states of past n games (n <= max_games)
        self.scores = None  # All scores of past states
        self.probabilities = None  # All move probabilities of past states
        self.save_results = True
        self.score_scaler = score_scaler  # Score scaler

    @jit
    def scale_score(self, score):
        return min(score / self.score_scaler, 1)

    def add_results(self, results):  # Adds all games to the training set
        for result in results:
            if self.save_results:
                game.save_game_to_file(result[1], result[3], result[4])
            states, scores, probs = self.parse_game(result)
            self.add_game(states, scores, probs)

    def add_game(self, states, scores, probs):
        if self.states is None:  # If training set not initialised
            self.states = states
            self.scores = scores
            self.probabilities = probs
        else:
            self.states = np.concatenate([self.states, states])
            self.scores = np.concatenate([self.scores, scores])
            self.probabilities = np.concatenate([self.probabilities, probs])

            self.cur_games += 1
            self.counts.append(len(states))

            if self.max_games == self.cur_games:
                rem = self.counts.pop(0)  # Remove first game in training set
                self.states = self.states[rem:]
                self.scores = self.scores[rem:]
                self.probabilities = self.probs[rem:]

    def parse_game(self, result):
        tot_states = len(result[3])
        result_states = np.asarray([convert_game_to_bin(x) for x in np.asarray(result[3])])
        score_val = scale_score(result[1])
        result_scores = np.asarray([score_val] * result_states.shape[0])
        result_probs_n = np.asarray(result[4], dtype=float)
        result_probs_p = result_probs_n / np.sum(result_probs_n, axis=1, keepdims=True)

        return result_states, result_scores, result_probs_p

    def sample_train_set(self, amount):
        args = random.sample(range(len(self.states)), amount)
        return self.states[args], self.scores[args], self.probabilities[args]


import game
import multiprocessing
import random

INIT_GAMES = 10
SAMPLE_AMOUNT = 512
SIMS_PER_MOVE = 100
SCORE_SCALER = 100000


# https://stackoverflow.com/questions/46725323/keras-tensorflow-exception-while-predicting-from-multiple-threads
# Not used currently
class Predictor:
    def __init__(self, model_path):
        self.query_model = load_model(model_path)
        self.query_model.predict(np.zeros((1, 20, 4, 4)))  # Warm-up
        self.session = K.get_session()
        self.graph = tf.get_default_graph()
        self.graph.finalize()  # Finalize for thread-safety


    def preproccesing(self, data):
        game = np.array(data)
        bin_rep = np.zeros(INPUT_SHAPE, dtype=bool)
        for i in range(MAX_POWER):  # For each power of twp
            bin_rep[i][game == i] = 1  # Assign True value
        v = np.expand_dims(np.array(bin_rep), axis=0)
        print(v.shape)
        print(v)
        return v

    def query_model(self, data):
        X = self.preproccesing(data)
        with self.session.as_default():
            with self.graph.as_default():
                prediction = self.query_model.predict(X)
        print(prediction)
        return prediction


if __name__ == '__main__':
    #  Importing here avoids doing it twice when using multiprocessing
    import tensorflow as tf
    from keras import backend as K
    from keras.models import Model, load_model
    from keras.layers import Input, Conv2D, Activation, Add, Dense, BatchNormalization, Flatten, concatenate
    from keras.optimizers import Adam

    trainData = TrainingSet()

    p = multiprocessing.Pool(5)
    # Play 10 games from which 5 in parallel
    init_results = p.map(game.play_using_MCTS, [SIMS_PER_MOVE] * 10)
    trainData.add_results(init_results)  # Store results

    cur_model = init_model()
    cur_model.save("cur_model.h5")

    # When including model estimation in playing games, I wasn't able to play the games in
    # parallel. This makes the pipeline ridiculously slow.
    cur_results = []
    for i in range(10):
        cur_results.append(game.play_using_MCTS(SIMS_PER_MOVE, cur_model))
    trainData.add_results(cur_results)

    for iters in range(1000):
        cur_scores = [x[1] for x in cur_results]
        cur_mean_score = sum(cur_scores) / len(cur_scores)
        print("Current model mean score %d" % cur_mean_score)

        new_model = load_model("cur_model.h5")
        # Train new model with sampled data. Training is sufficiently fast.
        sample_states, sample_scores, sample_probs = trainData.sample_train_set(SAMPLE_AMOUNT)
        new_model.fit(x=sample_states, y=[sample_probs, sample_scores],
                      batch_size=SAMPLE_AMOUNT, epochs=1, validation_split=0.001)

        new_results = []
        for i in range(5):  # Play 5 games with the new model
            new_results.append(game.play_using_MCTS(SIMS_PER_MOVE, cur_model))

        new_scores = [x[1] for x in new_results]
        new_mean_score = sum(new_scores) / len(new_scores)
        print("New model mean score %d" % new_mean_score)
        print("Train data to states: %d scores: %d" % (len(trainData.states), len(trainData.scores)))
        # If new model has at least 5% better results
        if new_mean_score/cur_mean_score > 1.05:
            print("New model better than previous")
            cur_model = new_model
            cur_model.save("cur_model.h5")
            cur_results = new_results
            trainData.add_results(cur_results)

"""
1) Play 10 games without neural network and save the games
2) Initialise the neural network
   Loop:
3) Sample 512 states from each of the last 50 games
4) Train the neural network on the sampled states and save it separately
5) Evaluate the new and the previous network: play 5 games with each one (multi-threaded)
6) If the mean score for the new network is at least 5% bigger replace old network with 
   new and save it
7) Store the 5 games of the current network to the training set
8) Go to step 3)
"""