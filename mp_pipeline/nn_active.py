"""
Runs the neural network. Handles move predictions for games and retraining of NN.
Saves model after each retraining.
NB! simplified approach, always trains with new data, does not evaluate the result.

Queues:
    1) q_model - games send their state here. nnActive picks them up
    2) q_games - dict for all of the game queues. This way we can send the result back.
    3) q_model_train - when event_can_train set then we pick training data from here and retrain model

This must run in main thread as tensorflow seems to not like running in separate threads...
"""
import numpy as np
import queue
from mp_pipeline.utils import MAX_BATCH_SIZE, MODELS_PATH


class NeuralNetActive:

    def __init__(self, q_model, q_games, model, event_can_train, q_model_train):
        self.q_model = q_model
        self.q_games = q_games
        self.model = model
        self.event_can_train = event_can_train
        self.q_model_train = q_model_train  # here we get sample data to retrain the model

        self.game_idxs = [0] * MAX_BATCH_SIZE
        self.batch = np.zeros((MAX_BATCH_SIZE, 4, 4), dtype=np.int32)
        self.model_ver = 0

    def run(self):

        i = 0
        while True:
            try:
                game_idx, game_state = self.q_model.get(False)  # simply fill batch
                self.game_idxs[i] = game_idx
                self.batch[i] = game_state
                i += 1

                if i >= MAX_BATCH_SIZE:
                    self.handle_batch(i)
                    i = 0

            except queue.Empty:
                # havent gotten a message in a while, empty
                self.handle_batch(i)
                i = 0
            except KeyboardInterrupt:
                break

    def handle_batch(self, batch_size):
        if batch_size > 0:
            p_batch, v_batch = self.predict()

            for i in range(batch_size):
                game_idx = self.game_idxs[i]
                p = p_batch[i]
                v = v_batch[i][0]
                self.q_games[game_idx].put((p, v))  # send p, v to simulator

        # after predicting a batch, check if we can retrain the model - do we have enough games
        if self.event_can_train.is_set():
            x, y = self.q_model_train.get()
            # print('GOT TRAINING DATA OF SHAPE', x.shape, y.shape)
            print('GOT TRAINING DATA OF LENGTH', len(x))
            self.model.fit(x, y)
            self.event_can_train.clear()
            self.model.save('%s/cur_model.h5' % MODELS_PATH)
            if self.model_ver % 10 == 0:  # save each 10th model...
                self.model.save('%s/model_ver_%d.h5' % (MODELS_PATH, self.model_ver))
            self.model_ver += 1

    def predict(self):
        # run NN
        p_batch, v_batch = self.model.query_model(self.batch)

        # p_batch.shape is (MAX_BATCH_SIZE, 4)
        # v_batch.shape is (MAX_BATCH_SIZE, 1)

        return p_batch, v_batch
