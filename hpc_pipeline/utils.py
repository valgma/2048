import numpy as np
from numba import jit
import math

SAMPLE_AMOUNT = 2048  # nbr of data points for training model
TRAIN_BATCHES = 100  # Amount of random sample batches to train on in one pass
MAX_GAMES = 25 #1_00  # nbr of games that are played before training NN: training states are sampled from these
MAX_BATCH_SIZE = 8  # batch size (nbr of moves) for running NN prediction
NBR_CONCURRENT_GAMES = 2*MAX_BATCH_SIZE  # seems reasonable...
MCTS_TREE_SIZE = 500
MODELS_PATH = '/gpfs/hpchome/earl/2048/models'

LEARNING_RATE = 0.001

MAX_POWER = 20  # Maximum power of two present in the board
INPUT_SHAPE = (MAX_POWER, 4, 4)

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

@jit
def convert_game_to_bin(data):
    game = np.array(data)
    bin_rep = np.zeros(INPUT_SHAPE, dtype=bool)
    for i in range(MAX_POWER):  # For each power of twp
        bin_rep[i][game == i] = 1  # Assign True value
    return np.array(bin_rep)

def print_consts():
    print('SAMPLE_AMOUNT:', SAMPLE_AMOUNT)
    print('TRAIN_BATCHES:', TRAIN_BATCHES)
    print('MAX_GAMES:', MAX_GAMES)
    print('MAX_BATCH_SIZE:', MAX_BATCH_SIZE)
    print('NBR_CONCURRENT_GAMES:', NBR_CONCURRENT_GAMES)
    print('MCTS_TREE_SIZE:', MCTS_TREE_SIZE)
    print('LEARNING_RATE:', LEARNING_RATE)

