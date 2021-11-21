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
        self.curr_playing = None
        self.value = None

    def set_value(self, new_value):
        self.value = new_value

    def set_curr_playing(self, player):
        self.curr_playing = player.name

    def switch_player(self):
        self.curr_playing = 'x' if self.curr_playing =='o' else 'o'

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

    def successors(self, player_name):
        empty_spots = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == '':
                    empty_spots.append((i, j))
        return[State(self.newboard(coord, player_name)) for coord in empty_spots]

    def newboard(self, coord, player_name):
        n_board = copy.deepcopy(self.board)
        n_board[coord[0]][coord[1]] = player_name
        return n_board

    def create_line(self, array):
        for i in range(len(array)):
            if array[i] == '':
                array[i] = ' '
        return f' {array[0]} | {array[1]} | {array[2]} \n'

    def __str__(self):
        c_board = copy.deepcopy(self.board)
        result= self.create_line(c_board[0])+\
                '---|---|---\n'+\
                self.create_line(c_board[1])+\
                '---|---|---\n'+\
                self.create_line(c_board[2])
        return result



class Move:
    def __init__(self, game, player, state):
        self.game = game
        self.player = player
        self.state = state


class Player:
    def __init__(self, name):
        if name != 'x' and name != 'o':
            raise IncorrectPlayerNameError(name)
        self.name = name
        self.opponent = None

    def set_opponent(self):
        self.opponent = Player('x') if self.name == 'o' else Player('o')


class Game:
    def __init__(self, starting_player):
        self.curr_state = State()
        self.curr_state.set_curr_playing(starting_player)
        self.curr_player = starting_player
        self.curr_player.set_opponent()

    def set_curr_state(self, new_state):
        self.curr_state = new_state

    def switch_player(self):
        tmp_player = self.curr_player
        self.curr_player = self.curr_player.opponent
        self.curr_player.opponent = tmp_player

    def Minmax(self, state, depth, isMin):
        if isMin:
            if state.is_terminal() or depth==0:
                state.set_value(state.heuristic())
                return state
            state_successors = state.successors('x')
            for u in state_successors:
                u.set_value(self.Minmax(u, depth-1, not isMin).value)
            return min(state_successors, key=lambda t:t.value)
        else:
            if state.is_terminal() or depth==0:
                state.set_value(state.heuristic())
                return state
            state_successors = state.successors('o')
            for u in state_successors:
                u.set_value(self.Minmax(u, depth-1, not isMin).value)
            return max(state_successors, key=lambda t:t.value)



def gameplay(game, depth):
    while not game.curr_state.is_terminal():
        print(game.curr_state)
        isMin = True if game.curr_player.name == 'x' else False
        game.set_curr_state(game.Minmax(game.curr_state, depth, isMin))
        game.switch_player()
    print(game.curr_state)



x = Player('o')
g = Game(x)
gameplay(g, 6)