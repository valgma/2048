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
from utils import MAX_GAMES, SAMPLE_AMOUNT, TRAIN_BATCHES, convert_game_to_bin
from datetime import datetime
import sys

class TrainingSet(mp.Process):

    def __init__(self, q_training_set, q_model_train, event_can_train, event_batch_ready, event_new_model_better):
        mp.Process.__init__(self)
        self.daemon = True
        self.q_training_set = q_training_set
        self.q_model_train = q_model_train
        self.event_can_train = event_can_train
        self.event_batch_ready = event_batch_ready
        self.event_new_model_better = event_new_model_better
        """ Currently worse games end sooner and therefore have greater influence on score. This might
            actually be quite good. (Model gets better, when these worse results don't appear)
            
            Additional problem might have to do with adding the results of worse models to the training set.
            
            Also negative overfitting might occur with 50 sample sets. (policy out accuracy goes to 100 percent)
            
        self.sim_games = [0] * NBR_CONCURRENT_GAMES  # Each thread current game count
        self.simulator_games = MAX_GAMES // NBR_CONCURRENT_GAMES
        self.nbr_games_between_training = int(MAX_GAMES // NBR_CONCURRENT_GAMES * NBR_CONCURRENT_GAMES)
        """
        self.nbr_games_between_training = MAX_GAMES
        self.sample_amount = SAMPLE_AMOUNT
        self.nbr_games_to_sample_from = self.nbr_games_between_training * 20

        self.game_counter = 0
        self.best_score = 0  # this is what we evaluate the current score against

        self.scores = []
        self.mcts_results = [[]]  # each el is list for one game [(s, p, score)]

        self.evaluating_new_model = False  # True when a new model needs to be evaluated

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
                    self.best_score = 0
                else:  # game data end
                    self.game_counter += 1
                    self.scores.append(data[1])

                    if self.game_counter % self.nbr_games_between_training == 0:
                        # update best score
                        cur_score = np.mean(self.scores)
                        print(str(datetime.now()), end=" ")
                        if cur_score > self.best_score:  # New weights performed better
                            print('New model better with score: %d' % cur_score)
                            self.best_score = cur_score
                            self.scores.clear()
                            self.update_model(True)
                        else:  # Old weights better
                            print('New model worse with score %d vs %d' % (cur_score, self.best_score))
                            self.scores.clear()
                            self.update_model(False)
                        sys.stdout.flush()

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
        sys.stdout.flush()

    def get_v_from_score(self, score):  # TODO ???????
        # return score / self.best_score
        return score > self.best_score

    def update_model(self, new_was_better):
        # sample states
        n = len(self.mcts_results)
        total_samples = sum([len(el) for el in self.mcts_results])
        print('Total samples: ', total_samples)

        # Initial training using samples from simple MCTS (done only once)
        initial_training = True if SAMPLE_AMOUNT < self.sample_amount else False

        if initial_training:
            eval_samples = self.sample_amount  # Train on all gathered samples
        else:
            eval_samples = total_samples if total_samples < self.sample_amount else self.sample_amount

        # init model input/output
        x = np.empty((eval_samples, 20, 4, 4), dtype=np.float32)
        y = [np.empty((eval_samples, 4), dtype=np.float32), np.empty(eval_samples, dtype=np.float32)]

        # check if network is ready to train
        while self.event_can_train.is_set():
            pass

        print("Training model on %d different batches of %d samples" % (TRAIN_BATCHES, eval_samples))
        sys.stdout.flush()
        if new_was_better:  # Notify NN to restore previous weights
            self.event_new_model_better.set()

        self.event_can_train.set()  # Let model know training starts
        for j in range(TRAIN_BATCHES):
            for i in range(eval_samples):  # Select random samples
                game_idx = random.randrange(n)
                el_idx = random.randrange(len(self.mcts_results[game_idx]))
                state, p, score = self.mcts_results[game_idx][el_idx]
                v = self.get_v_from_score(score)

                x[i] = convert_game_to_bin(state.reshape((4,4)))
                # lets try one hot...this is stupidly slow at the moment TODO: FIX THIS
                #best_move = np.argmax(p)
                #p = np.zeros(4)
                #p[best_move] = 1


                y[0][i] = p
                y[1][i] = v

            self.q_model_train.put((x, y))
            self.event_batch_ready.set()

        self.event_can_train.clear()

