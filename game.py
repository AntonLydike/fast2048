"""
This file holds our reimplementation of the 2048 game with
slightly altered rules as per the requirements of teh exercise.

We don't need a game object, as all required information is already
in the board state. Therefore our game object is the board as a 2d
array. The format is board[row][col]
"""
import random as rng
from math import log10, ceil, log2
from enum import IntEnum

import numpy as np
from numba import njit
from numba.typed import Dict, List

MAX_VAL = 1024


# define possible moves
class Directions(IntEnum):
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4

ALL_DIRECTIONS = List(Directions)

@njit
def init_board(size):
    """
    Initializes a board and places two initial twos
    """
    board = np.zeros((size,size), dtype='int32')
    place_two(board)
    place_two(board)
    return board


### CONTROLER

@njit
def choice(data):
    return data[rng.randint(0,len(data)-1)]

@njit
def place_two(board):
    spots = get_free_spots(board)
    if len(spots) == 0:
        return False
    pos = choice(spots)
    board[pos[0],pos[1]] = 2

@njit
def move_left(board):
    size = len(board)
    changed = False

    for x in range(size):
        b = np.copy(board[x])
        l = np.nonzero(b)[0]
        for i in range(len(l) - 1):
            if b[l[i]] == b[l[i + 1]]:
                b[l[i]] *= 2
                b[l[i + 1]] = 0

        l = np.nonzero(b)[0]
        for i in range(size):
            val = 0
            if i < len(l):
                val = b[l[i]]
            
            if board[x,i] != val:
                changed = True
            board[x,i] = val
    return changed, board

@njit
def move_right(board):
    val, board = move_left(np.fliplr(board))
    return val, np.fliplr(board)

@njit
def move_up(board):
    val, board = move_left(board.T)
    return val, board.T

@njit
def move_down(board):
    board = np.fliplr(board.T)
    val, board = move_left(board)
    return val, np.fliplr(board).T

@njit
def move(board, dir: Directions):
    if dir == Directions.UP:
        return move_up(board)
    elif dir == Directions.RIGHT:
        return move_right(board)
    elif dir == Directions.DOWN:
        return move_down(board)
    elif dir == Directions.LEFT:
        return move_left(board)
    return False, board



### OBSERVER

@njit
def is_lost(board):
    s = len(board)

    for x in range(s - 1):
        for y in range(s - 1):
            if board[x,y] == 0 \
                    or board[x,y + 1] == 0 or board[x,y + 1] == board[x,y] \
                    or board[x + 1,y] == 0 or board[x + 1,y] == board[x,y]:
                return False
    for i in range(s - 1):
        if board[i,s - 1] == 0 or board[i,s - 1] == board[i + 1,s - 1] or \
                board[s - 1,i] == 0 or board[s - 1,i] == board[s - 1,i + 1]:
            return False
    if board[s - 1,s - 1] == 0:
        return False
    return True

@njit
def get_free_spots(board):
    return np.vstack(np.where(board == 0)).T

@njit
def print_board(board):
    padd = ceil(log10(MAX_VAL))
    for row in board:
        print(" ".join([str(x).rjust(padd) for x in row]))

@njit
def score(board):
    ttl = 0
    for val in board.flat:
        if val == 0:
            continue
        ttl += np.log2(val/2) * val
    return int(ttl)

@njit
def copy_board(board):
    return np.copy(board)

@njit
def test():
    runs = 1
    for size in (4,):
        scores = np.zeros(runs)
        for i in range(runs):
            b = init_board(size)
            while not is_lost(b):
                done, b = move(b, rng.randint(1,4))
                if done:
                    place_two(b)
            scores[i] = score(b)
        # print(size)
        # print(f"scores: {scores}")
        # print("avg:", np.sum(scores) / runs)


if __name__ == '__main__':
    

    import timeit
    import sys
    print("testing first run")
    print(timeit.timeit('test()', globals=globals(), number=1))
    print("testing 1000 runs")
    print(timeit.timeit('test()', globals=globals(), number=1000))
    print("testing 1000 runs")
    print(timeit.timeit('test()', globals=globals(), number=1000))

    sys.exit(0)

    # this code is just straigh up copied from stackoverflow
    class _Getch:
        """Gets a single character from standard input.  Does not echo to the
    screen."""

        def __init__(self):
            try:
                self.impl = _GetchWindows()
            except ImportError:
                self.impl = _GetchUnix()

        def __call__(self):
            return self.impl()


    class _GetchUnix:
        def __init__(self):
            import tty, sys

        def __call__(self):
            import sys, tty, termios
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch


    class _GetchWindows:
        def __init__(self):
            import msvcrt

        def __call__(self):
            import msvcrt
            return msvcrt.getch()


    getch = _Getch()

    from time import sleep

    dirs = {
        'w': Directions.UP,
        'a': Directions.LEFT,
        's': Directions.DOWN,
        'd': Directions.RIGHT
    }

    print("Welcome to 2048 - CLI edition")
    size = int(input("Select your board size: "))

    while size not in range(2, 8):
        size = int(input("That is not a good size, please pick a new one: "))

    MAX_VAL = 2 ** (size ** 2)

    print()
    print("Play the game using your wasd keys.")
    print("Press q to restart")
    print()
    print("Ready?")
    input()

    while True:
        b = init_board(size)

        while not is_lost(b):
            #print('\x1bc')
            print(f" {score(b)} ")
            print()
            print_board(b)
            c = getch()
            while c not in dirs:
                if c == "q":
                    b = np.eye(1)
                    break

                print("invalid move!")
                sleep(1)
                c = getch()
            else:
                done, b = move(b, dirs[c])
                if done:
                    place_two(b)
        print("YOU LOST")
        input()
