import numpy as np
import random
from numba import jitclass, jit
from numba import int32
import time

spec = [
    ('score', int32),        # a simple scalar field
    ('state', int32[:, :]),  # 2d array
]

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3


@jitclass(spec)
class Game:
    def __init__(self):
        self.state = np.zeros((4, 4), dtype=np.int32)
        self.score = 0
        # gen two tiles
        self._gen_tile()
        self._gen_tile()

    def _gen_tile(self):
        # find locations for zeros
        zero_idx = []
        for i in range(4):
            for j in range(4):
                if self.state[i, j] == 0:
                    zero_idx.append((i, j))

        # choose randomly index and generate new value
        n = len(zero_idx)
        i, j = zero_idx[random.randrange(n)]
        new_block = 1 if random.random() < 0.9 else 2
        self.state[i, j] = new_block

    def move_simple(self, move):
        move_done = False

        # same algo for all moves obviously, too lazy to figure out generalization
        if move == UP:
            for j in range(4):
                # first remove 0s
                zeros = 0
                for i in range(4):
                    if self.state[i, j] == 0:
                        zeros += 1
                    elif zeros > 0:
                        self.state[i - zeros, j] = self.state[i, j]
                        self.state[i, j] = 0
                        move_done = True
                # now mergers
                i = 0
                while True:
                    if self.state[i, j] != 0:
                        if self.state[i + 1, j] == self.state[i, j]:
                            self.state[i, j] += 1
                            self.score += 2**self.state[i, j]
                            for k in range(i + 1, 3):  # move up
                                self.state[k, j] = self.state[k + 1, j]
                            self.state[3, j] = 0  # last row
                            move_done = True
                    i += 1

                    if i > 2:
                        break

        elif move == DOWN:
            for j in range(4):
                # first remove 0s
                zeros = 0
                for i in range(3, -1, -1):
                    if self.state[i, j] == 0:
                        zeros += 1
                    elif zeros > 0:
                        self.state[i + zeros, j] = self.state[i, j]
                        self.state[i, j] = 0
                        move_done = True
                # now mergers
                i = 3
                while True:
                    if self.state[i, j] != 0:
                        if self.state[i - 1, j] == self.state[i, j]:
                            self.state[i, j] += 1
                            self.score += 2**self.state[i, j]
                            for k in range(i - 1, 0, -1):  # move down
                                self.state[k, j] = self.state[k - 1, j]
                            self.state[0, j] = 0  # first row
                            move_done = True
                    i -= 1

                    if i < 1:
                        break

        elif move == LEFT:
            for i in range(4):
                # first remove 0s
                zeros = 0
                for j in range(4):
                    if self.state[i, j] == 0:
                        zeros += 1
                    elif zeros > 0:
                        self.state[i, j - zeros] = self.state[i, j]
                        self.state[i, j] = 0
                        move_done = True
                # now mergers
                j = 0
                while True:
                    if self.state[i, j] != 0:
                        if self.state[i, j + 1] == self.state[i, j]:
                            self.state[i, j] += 1
                            self.score += 2**self.state[i, j]
                            for k in range(j + 1, 3):  # move left
                                self.state[i, k] = self.state[i, k + 1]
                            self.state[i, 3] = 0  # right col
                            move_done = True
                    j += 1

                    if j > 2:
                        break

        elif move == RIGHT:
            for i in range(4):
                # first remove 0s
                zeros = 0
                for j in range(3, -1, -1):
                    if self.state[i, j] == 0:
                        zeros += 1
                    elif zeros > 0:
                        self.state[i, j + zeros] = self.state[i, j]
                        self.state[i, j] = 0
                        move_done = True
                # now mergers
                j = 3
                while True:
                    if self.state[i, j] != 0:
                        if self.state[i, j - 1] == self.state[i, j]:
                            self.state[i, j] += 1
                            self.score += 2**self.state[i, j]
                            for k in range(j - 1, 0, -1):  # move right
                                self.state[i, k] = self.state[i, k - 1]
                            self.state[i, 0] = 0  # left col
                            move_done = True
                    j -= 1

                    if j < 1:
                        break

        return move_done

    def move_full(self, move):
        move_done = self.move_simple(move)

        if move_done:
            self._gen_tile()

        return move_done


@jit(nopython=True)
def sim_till_end(game):
    can_play = True

    while can_play:
        move = random.randrange(4)
        can_play = game.move_full(move)

        if not can_play:
            for trial_move in range(4):
                if trial_move != move:
                    can_play = game.move_full(trial_move)
                if can_play:
                    break

        if not can_play:
            break


# @jit(nopython=True)
def sim_n_games(n):
    total_score = 0
    for i in range(n):
        game = Game()
        sim_till_end(game)
        total_score += game.score

    avg_score = total_score / n

    return avg_score


if __name__ == '__main__':
    n = 1_00_000
    sim_n_games(10)

    t1 = time.time()
    score = sim_n_games(n)
    t2 = time.time()

    print('games: %d, time: %f, score: %f' % (n, t2 - t1, score))