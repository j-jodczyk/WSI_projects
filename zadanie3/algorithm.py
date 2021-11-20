# wylacznie implementacja - nie trzeba dokumentacji
# rozbicie na funkcje
# Zaimplementuj algorytm MIN-MAX w grze kółko i krzyżyk
# na planszy 3x3. Program powinien grać sam ze sobą i
# wizualizować kolejne stany gry na terminalu.
# Pamiętaj, że implementacja musi być wykonana samodzielnie.
# Brak zrozumienia dostarczonego kodu rozwiązania równowazne jest plagiatowi.

import numpy as np
import copy

# max - o
# min - x


# TODO czy zakladamy, ze gra moze byc nierozstrzygnieta


class IncorrectPlayerNameError(Exception):
    def __init__(self, name):
        super().__init__('Name of Player must be "x" or "o"')
        self.name = name


class State:
    def __init__(self, board=np.empty((3, 3), str)):
        self.board = board

    def is_terminal(self):

        # check rows:
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] and self.board[i][0] in ['x', 'o']:
                return True

        # check column:
        for j in range(3):
            if self.board[0][j] == self.board[1][j] == self.board[2][j] and self.board[0][j] in ['x', 'o']:
                return True

        # check diagonal:
        if self.board[0][0] == self.board[1][1] == self.board[2][2] and self.board[0][0] in ['x', 'o']:
            return True

        # check antidiagonal:
        if self.board[2][0] == self.board[1][1] == self.board[0][2] and self.board[2][0] in ['x', 'o']:
            return True

        # check if no moves can be made:
        if not any('' in row for row in self.board):
            return True

        return False


    def heuristic(self):
        points = np.array([[3, 2, 3], [2, 4, 2], [3, 2, 3]])
        state_points = 0
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == "x":
                    state_points -= points[i][j]
                elif self.board[i][j] == "o":
                    state_points += points[i][j]
        return state_points

    def successors(self, player):
        empty_spots = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == '':
                    empty_spots.append((i, j))
        return[State(self.newboard(coord, player)) for coord in empty_spots]

    def newboard(self, coord, player):
        n_board = copy.deepcopy(self.board)
        n_board[coord[0]][coord[1]] = player.name
        return n_board



class Move:
    pass


class Player:
    def __init__(self, name):
        if name != 'x' and name != 'o':
            raise IncorrectPlayerNameError(name)
        self.name = name
        self.opponent = None

    def set_opponent(self):
        self.opponent = Player('x') if self.name == 'o' else Player('o')
    # polityka grania


class Game:
    def __init__(self, starting_player):
        self.begin_state = State()
        self.curr_state = State()
        self.starting_player = starting_player
        self.starting_player.set_opponent()
        self.curr_player = starting_player

    def set_curr_state(self, new_state):
        self.curr_state = new_state

    def switch_player(self):
        if self.curr_player.name == "x":
            tmp_player = self.curr_player
            self.curr_player = self.curr_player.opponent
            self.curr_player.opponent = tmp_player

    def gameplay(self, depth):
        while not self.curr_state.is_terminal():
            self.set_curr_state(self.Minmax(self.curr_state, depth)[0])
            self.switch_player
            print(Game)

    def Minmax(self, state, depth):
        payoff = {}
        if state.is_terminal() or depth==0:
            return state.heuristic()
        state_successors = state.successors(self.curr_player)
        for u in state_successors:
            payoff[u] = self.Minmax(u, depth-1)
        if self.curr_player.name == 'x':
            return max(payoff, key=payoff.get)
        else:
            return min(payoff, key=payoff.get)

    def create_line(array):
        for i in range(len(array)):
            if array[i] == None:
                array[i] = ' '
        return f' {array[0]} | {array[1]} | {array[2]} '

    def __str__(self):
        c_board = copy.deepcopy(self.curr_state.board)
        return self.create_line(c_board[0])+\
               '---|---|---'+\
               self.create_line(c_board[1])+\
               '---|---|---'+\
               self.create_line(c_board[2])


x = Player('x')
g = Game(x)
g.gameplay(3)