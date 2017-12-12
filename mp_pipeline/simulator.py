"""
Constantly plays a game using MCTS. When game finishes, starts a new one.
When reaches MCTS leaf, asks Neural Network to estimate move probabilities and expected score (normalised)
Runs in its own thread.
Uses 3 queues:
    1) q_model - for sending state to NN (to get evaluation)
    2) q_game - each game has its own for getting answer from NN
    3) q_training_set - once game finished, send (state, mcts_prob, score) to NN training set (for sampling)
"""
import numpy as np
import multiprocessing as mp
import psutil
import os
from mp_pipeline.game import Game
from mp_pipeline.utils import hs_state, copy_game, calc_move_scores, get_best_probs_from_count
from mp_pipeline.utils import MCTS_TREE_SIZE


class Simulator(mp.Process):

    def __init__(self, idx, q_game, q_model, q_training_set):
        mp.Process.__init__(self)
        self.idx = idx
        self.q_game = q_game
        self.q_model = q_model
        self.q_training_set = q_training_set

        self._tmp_memory = []  # for each move (state, predicted_p)
        self._tmp_score = 0

        # lower the process priority, to not allow this to freeze the computer
        p = psutil.Process(os.getpid())
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

    def run(self):
        print(self.idx, 'started')

        i = 0
        while True:
            i += 1
            cur_score = self.simulate()
            print(self.idx, 'game: %d, score: %.3f' % (i, cur_score))

    def get_prediction_from_nn(self, game):
        # send state to NN and wait for answer
        self.q_model.put((self.idx, game.state))
        p, v = self.q_game.get()

        return p, v

    def save_game_data(self):
        # send (s, p, score) to nn training
        for state, p in self._tmp_memory:
            data = state, p, self._tmp_score
            self.q_training_set.put(data)

        self.q_training_set.put((None, self._tmp_score))  # to notify end of data stream

        self._tmp_memory.clear()

    def get_next_move(self, game):
        return self.get_next_move_mcts(game, tree_size=MCTS_TREE_SIZE)

    def simulate(self):
        self._tmp_memory.clear()
        self._tmp_score = 0

        game = Game()

        can_play = True

        while can_play:
            move = self.get_next_move(game)
            can_play = game.move_full(move)

            if not can_play:
                for trial_move in range(4):
                    if trial_move != move:
                        can_play = game.move_full(trial_move)
                    if can_play:
                        break

            if not can_play:
                break

        self._tmp_score = game.score
        self.save_game_data()

        return game.score

    def get_next_move_mcts(self, game, tree_size):
        """For each move, build MCTS. pick best move
        """
        best_move = 0
        c_puct = 1.01  # exploration coef, bigger should indicate more exploration (randomness)

        # need to keep the tree somehow, use dict initially
        p, v = self.get_prediction_from_nn(game)
        tree = {(hs_state(game.state), a): (0, 0.0, 0.0, p[a]) for a in range(4)}  # all new leaves get expanded like this

        # now sim n games, when game reaches leaf -> simply evaluate game
        for _i in range(tree_size):
            path = []  # to keep track of current path, for updating
            cur_game = copy_game(game)

            cur_state_flat = hs_state(cur_game.state)
            # run till get to leaf
            while (cur_state_flat, 0) in tree:
                # evaluate which move to take based on current tree
                s_a = [tree[(cur_state_flat, a)] for a in range(4)]  # get current tree nodes for all a's
                move_scores = calc_move_scores(s_a, c_puct)

                next_move = np.argmax(move_scores)  # simple argmax her, TODO: check paper!
                can_play = cur_game.move_full(next_move)  # NB! can it be a problem if cannot play??

                if not can_play:
                    moves = np.argsort(move_scores)  # slow...
                    for a in range(3, -1, -1):
                        next_move = moves[a]
                        can_play = cur_game.move_full(next_move)
                        if can_play:
                            break

                if not can_play:
                    break
                else:
                    path.append((cur_state_flat, next_move))  # remember which move was chosen
                    cur_state_flat = hs_state(cur_game.state)

            # now we're in leaf - need to expand and get score
            # NB! this is place for NN estimation
            p, v = self.get_prediction_from_nn(cur_game)
            # expand leaf
            for a in range(4):
                tree[(cur_state_flat, a)] = (0, 0.0, 0.0, p[a])

            # update all path nodes in tree
            for s, a in path:
                n_a, w_a, q_a, p_a = tree[(s, a)]
                n_a += 1
                w_a += v
                q_a = w_a / n_a
                tree[(s, a)] = n_a, w_a, q_a, p_a

        # choose most visited action, TODO: check paper
        counts = [tree[(hs_state(game.state), a)][0] for a in range(4)]

        best_move, p = get_best_probs_from_count(counts)
        self._tmp_memory.append((np.copy(game.state), p))  # remember state and predicted move probabilities

        return best_move
