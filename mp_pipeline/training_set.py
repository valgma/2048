"""
Runs in own thread. Receives (state, mcts_prob, score) results from games.
Once enough games are received, samples data, formats and sends to NN.
Queues:
    1) q_training_set - for receiving results from games
    2) q_mode_train - sending data to NN (for training)
Raises event_can_train event for NN notification.
Handles calculation of v from score - currently compares to average of best batch.

NB! simplified approach, each training batch is completely separate thing. Previous results get removed.
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

        self.game_counter = 0
        self.best_score = 1_000  # this is what we evaluate the current score against
        self.scores = []
        self.mcts_results = []  # (s, p, score)

    def run(self):
        while True:
            data = self.q_training_set.get()  # get game states

            if data[0] is None:  # game data end
                self.game_counter += 1
                self.scores.append(data[1])

                if self.game_counter >= MAX_GAMES:
                    self.update_model()
                    self.game_counter = 0
                    self.mcts_results.clear()

                    # update best score
                    cur_score = np.mean(self.scores)
                    if cur_score > self.best_score:
                        self.best_score = cur_score
                    self.scores.clear()
                    print('BEST SCORE:', self.best_score)
            else:
                self.mcts_results.append(data)

    def get_v_from_score(self, score):
        return score > self.best_score

    def update_model(self):
        # sample states
        n = len(self.mcts_results)
        print('TOTAL SAMPLES TO CHOOSE FROM:', n)

        # init model input/output
        x = np.empty((SAMPLE_AMOUNT, 4, 4, 1), dtype=np.float32)
        y = [np.empty((SAMPLE_AMOUNT, 4), dtype=np.float32), np.empty(SAMPLE_AMOUNT, dtype=np.float32)]

        # select randomly data to fill
        for i in range(SAMPLE_AMOUNT):
            el_idx = random.randrange(n)
            state, p, score = self.mcts_results[el_idx]
            v = self.get_v_from_score(score)
            x[i] = np.expand_dims(state, 2)
            y[0][i] = p
            y[1][i] = v

        self.q_model_train.put((x, y))
        self.event_can_train.set()  # let model know it can train now
        print('SENT DATA TO TRAIN MODEL')