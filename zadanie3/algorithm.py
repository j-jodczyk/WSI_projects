# wylacznie implementacja - nie trzeba dokumentacji
# rozbicie na funkcje
# Zaimplementuj algorytm MIN-MAX w grze kółko i krzyżyk
# na planszy 3x3. Program powinien grać sam ze sobą i
# wizualizować kolejne stany gry na terminalu.
# Pamiętaj, że implementacja musi być wykonana samodzielnie.
# Brak zrozumienia dostarczonego kodu rozwiązania równowazne jest plagiatowi.

import numpy as np

# max - o
# min - x

# TODO czy zakladamy, ze gra moze byc nierozstrzygnieta


class State:
    def __init__(self):
        self.board = np.empty(3, 3)

    def is_terminal(self):
        # check rows:
        for i in range(3):
            if len(set(self.board[i])) == 1:
                return True

        # check column:
        for j in range(3):
            if self.board[0][j] == self.board[1][j] == self.board[2][j]:
                return True

        # check diagonal:
        if self.board[0][0] == self.board[1][1] == self.board[2][2]:
            return True

        # check antidiagonal:
        if self.board[2][0] == self.board[1][1] == self.board[0][2]:
            return True

        # check if no moves can be made:
        if not any(None in row for row in self.board):
            return True

        return False

    def successors(self):
        pass


class Move:
    pass


class Player:
    # polityka grania
    pass


class Game:
    def __init__(self):
        self.begin_state = State()
        self.curr_state = State()

    def heuristic(self, state):
        points = np.array([[3, 2, 3], [2, 4, 2], [3, 2, 3]])
        state_points = 0
        for i in range(3):
            for j in range(3):
                if state[i][j] == "x":
                    state_points -= points[i][j]
                elif state[i][j] == "o":
                    state_points += points[i][j]
        return state_points

    def game(self):
        if self.curr_state.is_terminal():
            return self.heuristic(self.curr_state)
        curr_state_succ = self.curr_state.successors()
