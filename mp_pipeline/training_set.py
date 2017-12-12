"""
Runs in own thread. Receives (state, mcts_prob, score) results from games.
Once enough games are received, samples data, formats and sends to NN.
Queues:
    1) q_training_set - for receiving results from games
    2) q_mode_train - sending data to NN (for training)
Raises event_can_train event for NN notification.
Handles calculation of v from score - currently compares to average of best batch.

Sends data for retraining after N games. Samples from 20 * N games.
"""
import multiprocessing as mp
import numpy as np
import random
from mp_pipeline.utils import MAX_GAMES, SAMPLE_AMOUNT


class TrainingSet(mp.Process):

    def __init__(self, q_training_set, q_model_train, event_can_train):
        mp.Process.__init__(self)
        self.daemon = True
        self.q_training_set = q_training_set
        self.q_model_train = q_model_train
        self.event_can_train = event_can_train
        self.nbr_games_between_training = MAX_GAMES
        self.sample_amount = SAMPLE_AMOUNT
        self.nbr_games_to_sample_from = self.nbr_games_between_training * 20

        self.game_counter = 0
        self.best_score = 0  # this is what we evaluate the current score against
        self.scores = []
        self.mcts_results = [[]]  # each el is list for one game [(s, p, score)]

    def run(self):
        while True:
            data = self.q_training_set.get()  # get game states

            if data[0] is None:
                if data[1] == 'KILL':
                    break
                elif data[1] == 'MAX_GAMES':
                    self.nbr_games_between_training = data[2]
                elif data[1] == 'SAMPLE_AMOUNT':
                    self.sample_amount = data[2]
                elif data[1] == 'CLEAR':
                    while not self.q_training_set.empty():
                        self.q_training_set.get()
                else:  # game data end
                    self.game_counter += 1
                    self.scores.append(data[1])

                    if self.game_counter % self.nbr_games_between_training == 0:
                        # update best score
                        cur_score = np.mean(self.scores)
                        if cur_score > self.best_score:
                            self.best_score = cur_score
                        self.scores.clear()
                        print('BEST SCORE:', self.best_score)

                        self.update_model()

                    # once we have enough games, we start overwriting data
                    if self.game_counter > self.nbr_games_to_sample_from:
                        self.game_counter = 0

                    if len(self.mcts_results) > self.game_counter:
                        self.mcts_results[self.game_counter] = []  # reserve space for next game data
                    else:
                        self.mcts_results.append([])
            else:
                self.mcts_results[self.game_counter].append(data)

        print('TERMINATING TRAINING SET')

    def get_v_from_score(self, score):  # TODO ???????
        # return score / self.best_score
        return score > self.best_score

    def update_model(self):
        # sample states
        n = len(self.mcts_results)
        total_samples = sum([len(el) for el in self.mcts_results])
        print('TOTAL SAMPLES TO CHOOSE FROM:', total_samples)

        # init model input/output
        x = np.empty((self.sample_amount, 4, 4, 1), dtype=np.float32)
        y = [np.empty((self.sample_amount, 4), dtype=np.float32), np.empty(self.sample_amount, dtype=np.float32)]

        # select randomly data to fill
        for i in range(self.sample_amount):
            game_idx = random.randrange(n)
            el_idx = random.randrange(len(self.mcts_results[game_idx]))
            state, p, score = self.mcts_results[game_idx][el_idx]
            v = self.get_v_from_score(score)
            x[i] = np.expand_dims(state, 2)

            # lets try one hot...this is stupidly slow at the moment TODO: FIX THIS
            best_move = np.argmax(p)
            p = np.zeros(4)
            p[best_move] = 1

            y[0][i] = p
            y[1][i] = v

        # check if network is ready to train
        while self.event_can_train.is_set():
            pass

        self.q_model_train.put((x, y))
        self.event_can_train.set()  # let model know it can train now
        print('SENT DATA TO TRAIN MODEL')
