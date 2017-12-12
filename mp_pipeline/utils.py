import numpy as np
from numba import jit
import math

SAMPLE_AMOUNT = 500_000  # nbr of data points for training model
MAX_GAMES = 1_00  # nbr of games that are used for training NN: training states are sampled from these
MAX_BATCH_SIZE = 16  # batch size (nbr of moves) for running NN prediction
NBR_CONCURRENT_GAMES = 2*MAX_BATCH_SIZE  # seems reasonable...
MCTS_TREE_SIZE = 1_600
MODELS_PATH = 'mp_pipeline/models'


# has for dictionary that is used to keep the tree
def hs_state(state):
    return state.tostring()


@jit(nopython=True)
def calc_move_scores(s, c_puct):
    move_scores = np.zeros(4)

    n_s = 0
    for a in range(4):
        n_s += s[a][0]

    n_s_sqrt = math.sqrt(1 + n_s)
    for a in range(4):
        n_a, w_a, q_a, p_a = s[a]
        u_a = c_puct * p_a * n_s_sqrt / (1 + n_a)
        move_scores[a] = q_a + u_a

    return move_scores


@jit(nopython=True)
def get_best_probs_from_count(counts):
    p = np.zeros(4, dtype=np.float32)
    best_move = 0
    n_best = 0
    n_sum = 0.0
    for a in range(4):
        n_a = counts[a]
        n_sum += n_a

        if n_a > n_best:
            best_move = a
            n_best = n_a

    if n_sum > 0:  # when we can play...otherwise we can get division by zero if there is no move
        for a in range(4):
            p[a] = counts[a] / n_sum

    return best_move, p


def print_consts():
    print('SAMPLE_AMOUNT:', SAMPLE_AMOUNT)
    print('MAX_GAMES:', MAX_GAMES)
    print('MAX_BATCH_SIZE:', MAX_BATCH_SIZE)
    print('NBR_CONCURRENT_GAMES:', NBR_CONCURRENT_GAMES)
    print('MCTS_TREE_SIZE:', MCTS_TREE_SIZE)
