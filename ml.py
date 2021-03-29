"""
This file implements the simple Q-Learning stuff
"""
from game import init_board, move, choice, is_lost, place_two, Directions, score, print_board

from jitpickler import jitpickle

from numba import int32, float32, int64, njit
from numba.experimental import jitclass
from numba.core import types
from numba.typed import Dict, List

from math import ceil

import random

from numpy import log2
import numpy as np

BOARD_SIZE = 2
MAX_VAL = 2 ** (BOARD_SIZE ** 2)
EPISODES = 10000
PRINT = True

@njit
def Config():
    return {
        'ALPHA'            : 0.7031166889352032,
        'ALPHA_DECAY_TO'   : 0.22411328519976909,
        'GAMMA'            : 0.6610130093415523,
        'EPSILON'          : 0.16414504850955716,
        'EPSILON_DECAY_TO' : 0.6267082491289307,
        'DECAY_VALUE'      : 0.9998646374156888,
    }

@njit
def create_index(state, action, board_size=None):
    """
    creates an index for the Q table.

    The board state matrix with entries ((a1,...,an),(b1,...,bn),...) 
    is converted into a number:

    log2(a1) | log2(a2) | ... | log2(b1) | ...

    the matrix should only hold powers of 2, "|" is binary concatenation
    """
    if board_size == None:
        board_size = BOARD_SIZE

    bits = ceil(log2(log2(2 ** (board_size ** 2))) + 1)

    state = np.copy(state)

    # normalize table (rotate it to always be the same move)
    if action == Directions.RIGHT:
        state = np.fliplr(state)
    elif action == Directions.UP or action == Directions.DOWN:
        state = state.T
        if action == Directions.DOWN:
            state = np.fliplr(state)

    hash = 0

    for row in state:
        for val in row:
            hash = hash << bits
            if val != 0:
                hash = hash + int(log2(val))

    return hash

@njit
def q_create():
    return Dict.empty(
        key_type=types.int64,
        value_type=types.float32
    )
@njit
def q_get(q, state, action: Directions):
    a = action + 0
    return q.get(create_index(state, a), 0)

@njit
def q_set(q, state, a, val):
    if is_lost(state):
        print("error, trying to set a terminal state!")
        return
    q[create_index(state, a)] = val
@njit
def q_get_actions(q, state):
    return [
        (Directions.UP, q_get(q, state, Directions.UP)),
        (Directions.DOWN, q_get(q, state, Directions.DOWN)),
        (Directions.LEFT, q_get(q, state, Directions.LEFT)),
        (Directions.RIGHT, q_get(q, state, Directions.RIGHT))
    ]
@njit
def q_get_max_action(q, state):
    opts = q_get_actions(q, state)
    max = opts[0]
    for o in opts[1:]:
        if o[1] > max[1]:
            max = o
    return max

# these ugly methods are used by the optimized evolutionary algorith
# to save on time
@njit
def q_set_index(q, ind, val):
    q[ind] = val
@njit
def q_get_index(q, ind):
    return q.get(ind, 0)

@njit
def decay(val, target=0, rate=0.99):
    """
    this method decreases a value val by rate
    """
    return ((val - target) * rate) + target

@njit
def r(old_state, action, new_state):
    """
    this is the reward function, it returns the score of the new state
    """
    return score(new_state)

@njit
def epsilon_greedy_choice(q, state, eps):
    """
    performs an epsilon greedy search for a move to perform at a given state
    """
    if random.random() > eps:  # probability 1-eps
        return q_get_max_action(q, state)[0]
    else:
        return choice(q_get_actions(q, state))[0]

@njit(fastmath=True)
def jitsum(list):
    ttl = 0
    for x in list:
        ttl += x
    return x

@njit
def q_learning(conf: Config):
    """
    run the Q-Learning algorithm for a specific config
    """
    episode = 0
    alpha = conf['ALPHA']
    gamma = conf['GAMMA']
    epsilon = conf['EPSILON']

    q = q_create()

    points_sum = 0
    batch_size = 100  # number of episodes for which to print avg score

    score_hist = List.empty_list(types.int64)
    decision_hist = List.empty_list(types.float64)

    while episode < EPISODES:
        board = init_board(BOARD_SIZE)

        decisions = List()

        while not is_lost(board):
            # do reinforcement learning stuff here:
            old_state = np.copy(board)
            a = epsilon_greedy_choice(q, old_state, epsilon)
            decisions.append(0 if q_get(q, board, a) == 0 else 1)

            # if it's not a valid move, retry
            done, board = move(board, a)
            if not done:
                continue
            # this is not optimal, as it introduces a couple of different
            # possible board states after each move but we could not find
            # a better way to do this.
            place_two(board)
            new_state = board
            r_val = r(old_state, a, new_state)
            max_action_val = q_get_max_action(q, new_state)[1]
            qsa = q_get(q, old_state, a)
            # math
            q_set(q, old_state, a, qsa + alpha * (r_val + gamma * max_action_val - qsa))

        # decay the value every 50th episode
        if episode % 50 == 0:
            epsilon = decay(epsilon, conf['EPSILON_DECAY_TO'], conf['DECAY_VALUE'])
        if episode % 50 == 0:
            alpha = decay(alpha, conf['ALPHA_DECAY_TO'], conf['DECAY_VALUE'])

        score_hist.append(score(board))
        decision_hist.append(jitsum(decisions) / len(decisions))

        #points_sum += score(board)
        episode += 1
        #if episode % batch_size == 0:
            #print(episode)
            #print("alpha:", alpha)
            #print("epsilon:", epsilon)
            #print("points:", points_sum / batch_size)
            #points_sum = 0
        #    print(f"=========== episode {episode} ===========")
        #    print(f"   \u03B1 = {alpha:.03f}       \u03B5 = {epsilon:.03f} ")
        #    print(f" |Q| = {len(q)}    points = {points_sum / batch_size:.03f}")
        #    print()
        #    


    # save decision and score history as json
    # when the latex file is built, graphs are generated from this data
    return decision_hist, score_hist, q


def run(c):
    decision_hist, score_hist, q = q_learning(c)

    decision_hist = list(decision_hist)
    score_hist = list(score_hist)

    import json
    with open(f"decisions-{BOARD_SIZE}-large.json", 'w') as f:
        json.dump(decision_hist, f)
    with open(f"scores-{BOARD_SIZE}-large.json", 'w') as f:
        json.dump(score_hist, f)

    # if the Q-Table is saved, it can be loaded later on
    # for further improvements or just to watch it play
    # the Q-table must always be used with the same 
    # BOARD_SIZE, MAX_VAL parameter
    # otherwise the dictionary keys cannot be calculated
    # correctly
    if input("save this Q? ") in ('Y', 'y'):
        import pickle
        with open(f'q{BOARD_SIZE}.pickle', 'wb') as f:
            pickle.dump(dict(q), f)
        print("saved.")

    # watch the q-table play a game 
    if input("you wanna try it? ") in ('y', 'Y'):
        while True:
            print('\x1bc')
            play_from_q(q)


def move_to_str(dir: Directions):
    if dir == Directions.UP:
        return '↑'
    elif dir == Directions.RIGHT:
        return '→'
    elif dir == Directions.DOWN:
        return '↓'
    elif dir == Directions.LEFT:
        return '←'
    return 'unknown'


def play_from_q(q):
    """
    observe a game being played when we always play
    the top move from the Q-Table
    """
    b = init_board(BOARD_SIZE)
    while not is_lost(b):
        print_board(b)
        m = q_get_max_action(q, b)
        print()
        print(f"selected move: {move_to_str(m[0])} with confidence {m[1]:.02f}")
        input("Continue?")
        move(b, m[0])
        place_two(b)
        print('\x1bc')
    print_board(b)
    print()
    print("you lost anyway, score is ", score(b))


def load_q():
    """
    loads a Q-Table from a file
    """
    import pickle
    with open(f'q{BOARD_SIZE}.pickle', 'rb') as f:
        return Dict(pickle.load(f))


if __name__ == '__main__':
    if input("play stored q game? ") in ('y', 'Y'):
        play_from_q(load_q())
    else:
        import timeit

        print(timeit.timeit('q_learning(Config())', globals=globals(), number=1))
        print(timeit.timeit('q_learning(Config())', globals=globals(), number=100))
        print(timeit.timeit('q_learning(Config())', globals=globals(), number=100))
