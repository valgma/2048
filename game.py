import numpy as np
from numba import jit, njit, prange
import random
from time import time

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3


@jit(nopython=True)
def move_full(state, move, score=0, deterministic=False):
    """Changes current state (in place) according to move and generates new tile.
    :param state: 4x4 board with powers of 2
    :param move: UP, DOWN, LEFT, RIGHT
    :param score: previous score
    :param deterministic: if new tile loc is chosen deterministally
    :return: if move was successful, new score
    """
    move_done, score = move_simple(state, move, score)

    if not move_done:
        return move_done, score

    gen_new_tile(state, deterministic)

    return move_done, score


@jit(nopython=True)
def gen_new_tile(state, deterministic=False):
    if deterministic:
        for i in range(4):
            for j in range(4):
                if state[i, j] == 0:
                    state[i, j] = 2
                    return

    # zero_idx = np.where(state == 0)
    # zero_idx = np.ravel_multi_index(zero_idx, (4, 4))
    # idx = np.random.choice(zero_idx, 1)
    # idx = np.unravel_index(idx, (4, 4))
    # state[idx] = np.random.choice([2, 4], 1, p=[0.9, 0.1])

    # find locaations for zeros
    zero_idx = []
    for i in range(4):
        for j in range(4):
            if state[i, j] == 0:
                zero_idx.append((i, j))

    # choose randomly index and generate new value
    n = len(zero_idx)
    i, j = zero_idx[random.randrange(n)]
    new_block = 1 if random.random() < 0.9 else 2
    state[i, j] = new_block


@jit(nopython=True)
def move_simple(state, move, score=0):
    """
    Changes current state (in place) according to move.
    :param state: 4x4 board with powers of 2
    :param move: UP, DOWN, LEFT, RIGHT
    :param score: previous score
    :return: if move was successful, new score
    """
    move_done = False

    # same algo for all moves obviously, too lazy to figure out generalization
    if move == UP:
        for j in range(4):
            # first remove 0s
            zeros = 0
            for i in range(4):
                if state[i, j] == 0:
                    zeros += 1
                elif zeros > 0:
                    state[i - zeros, j] = state[i, j]
                    state[i, j] = 0
                    move_done = True
            # now mergers
            i = 0
            while True:
                if state[i, j] != 0:
                    if state[i + 1, j] == state[i, j]:
                        state[i, j] += 1
                        score += 2**state[i, j]
                        for k in range(i + 1, 3):  # move up
                            state[k, j] = state[k + 1, j]
                        state[3, j] = 0  # last row
                        move_done = True
                i += 1

                if i > 2:
                    break

    elif move == DOWN:
        for j in range(4):
            # first remove 0s
            zeros = 0
            for i in range(3, -1, -1):
                if state[i, j] == 0:
                    zeros += 1
                elif zeros > 0:
                    state[i + zeros, j] = state[i, j]
                    state[i, j] = 0
                    move_done = True
            # now mergers
            i = 3
            while True:
                if state[i, j] != 0:
                    if state[i - 1, j] == state[i, j]:
                        state[i, j] += 1
                        score += 2**state[i, j]
                        for k in range(i - 1, 0, -1):  # move down
                            state[k, j] = state[k - 1, j]
                        state[0, j] = 0  # first row
                        move_done = True
                i -= 1

                if i < 1:
                    break

    elif move == LEFT:
        for i in range(4):
            # first remove 0s
            zeros = 0
            for j in range(4):
                if state[i, j] == 0:
                    zeros += 1
                elif zeros > 0:
                    state[i, j - zeros] = state[i, j]
                    state[i, j] = 0
                    move_done = True
            # now mergers
            j = 0
            while True:
                if state[i, j] != 0:
                    if state[i, j + 1] == state[i, j]:
                        state[i, j] += 1
                        score += 2**state[i, j]
                        for k in range(j + 1, 3):  # move left
                            state[i, k] = state[i, k + 1]
                        state[i, 3] = 0  # right col
                        move_done = True
                j += 1

                if j > 2:
                    break

    elif move == RIGHT:
        for i in range(4):
            # first remove 0s
            zeros = 0
            for j in range(3, -1, -1):
                if state[i, j] == 0:
                    zeros += 1
                elif zeros > 0:
                    state[i, j + zeros] = state[i, j]
                    state[i, j] = 0
                    move_done = True
            # now mergers
            j = 3
            while True:
                if state[i, j] != 0:
                    if state[i, j - 1] == state[i, j]:
                        state[i, j] += 1
                        score += 2**state[i, j]
                        for k in range(j - 1, 0, -1):  # move right
                            state[i, k] = state[i, k - 1]
                        state[i, 0] = 0  # left col
                        move_done = True
                j -= 1

                if j < 1:
                    break

    return move_done, score


@jit(nopython=True)
def init_game():
    i, j = random.randrange(4), random.randrange(4)
    state = np.zeros((4, 4), dtype=np.int32)
    state[i, j] = 1

    return state


def interactive():
    moves = {'w': 'UP', 's': 'DOWN', 'a': 'LEFT', 'd': 'RIGHT'}
    state = init_game()

    print(state)
    score = 0

    while True:
        move = input('WASD: ')
        move = move.lower()
        if move == 'q':
            print('SCORE: %d' % score)
            break
        if move == 'w':
            succ, score = move_full(state, UP, score)
        if move == 's':
            succ, score = move_full(state, DOWN, score)
        if move == 'a':
            succ, score = move_full(state, LEFT, score)
        if move == 'd':
            succ, score = move_full(state, RIGHT, score)

        print('move: %s, succ: %r, score: %d' % (moves[move], succ, score))
        print(state)


@jit(nopython=True)
def sim_from_state(state, score, move_count, gen_tile=False):
    state = np.copy(state)
    can_play = True

    if gen_tile:
        gen_new_tile(state)

    while can_play:
        move = random.randrange(4)
        can_play, score = move_full(state, move, score)

        if not can_play:
            for trial_move in range(4):
                if trial_move != move:
                    can_play, score = move_full(state, trial_move, score)
                if can_play:
                    break

        if not can_play:
            break
        else:
            move_count += 1

    return state, score, move_count


@jit(nopython=True)
def sim_one_game():
    state = init_game()
    state, score, move_count = sim_from_state(state, 0, 0)

    return state, score, move_count


@jit(nopython=True)
def get_next_move_simpleMC(state, score, move_count, n=1000):
    """For each move, simulate n games and check which one gives best avg/total score
    """
    best_move = 0
    best_score = 0

    for move in range(4):
        move_state = np.copy(state)
        can_move, move_score = move_simple(move_state, move, score)
        if can_move:
            # NB! cannot forget to add new random tile
            total_move_score, total_move_count = sim_n_games_from_state(move_state, move_score, move_count + 1, n, True)

            # total_move_score = total_move_count  # NB! for testing move_count vs score

            if total_move_score > best_score:
                best_move = move
                best_score = total_move_score

    return best_move


# @jit(nopython=True)
@njit(parallel=True)
def sim_n_games_from_state(state, score, move_count, n, gen_tile=False):
    total_score, total_move_count = 0, 0

    for i in prange(n):
        _, _score, _move_count = sim_from_state(state, score, move_count, gen_tile)
        total_move_count += _move_count
        total_score += _score

    return total_score, total_move_count


def sim_n_games(n):
    state = init_game()
    t1 = time()
    total_score, total_move_count = sim_n_games_from_state(state, 0, 0, n)
    t2 = time()

    total_time = t2 - t1
    avg_move_count = total_move_count / n
    avg_score = total_score / n

    print('games: %d, time: %f, avg_score: %f, avg_move_count: %f' % (n, total_time, avg_score, avg_move_count))


def play_using_simpleMC(sims_per_move=100):
    state = init_game()
    move_count = 0
    score = 0
    can_play = True
    t1 = time()

    while can_play:
        move = get_next_move_simpleMC(state, score, move_count, sims_per_move)
        can_play, score = move_full(state, move, score)

        if not can_play:
            for trial_move in range(4):
                if trial_move != move:
                    can_play, score = move_full(state, trial_move, score)
                if can_play:
                    break

        if not can_play:
            break
        else:
            move_count += 1

        # if move_count % 100 == 0:
        #     t2 = time()
        #     print('move: %d, score: %d, time: %f' % (move_count, score, t2 - t1))
        #     t1 = t2

    return state, score, move_count


if __name__ == '__main__':
    # interactive()

    # call once to compile the functions
    state, score, move_count = sim_one_game()
    # print(state)
    # print('move_count: %d, score: %d' % (move_count, score))

    # sim_n_games(10000)

    # t1 = time()
    # state, score, move_count = play_using_simpleMC()
    # t2 = time()
    # print(state)
    # print('score: %d, move_count: %d, time: %f' % (score, move_count, t2 - t1))

    t1 = time()
    for sims_per_move in (100, 1000, 10000):
        for i in range(5):
            state, score, move_count = play_using_simpleMC(sims_per_move)
            t2 = time()
            best_tile = np.max(state)
            print('sims: %d; i: %d; best: %d; score: %d, time: %.3f' % (sims_per_move, i, best_tile, score, t2 - t1))
            t1 = time()