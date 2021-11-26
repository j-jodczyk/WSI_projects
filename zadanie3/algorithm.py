# wylacznie implementacja - nie trzeba dokumentacji
# rozbicie na funkcje
# Zaimplementuj algorytm MIN-MAX w grze kółko i krzyżyk
# na planszy 3x3. Program powinien grać sam ze sobą i
# wizualizować kolejne stany gry na terminalu.
# Pamiętaj, że implementacja musi być wykonana samodzielnie.
# Brak zrozumienia dostarczonego kodu rozwiązania równowazne jest plagiatowi.

import numpy as np
import copy

from numpy.core.fromnumeric import transpose

# max - o
# min - x


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

    def check_row(self, row):
        if len(set(row)) == 1:
            return row[0]

    def who_wins(self): # do zmiany
        winner = ''
        # check rows:
        for row in self.board:
            winner = self.check_row(row)

        # check column:
        transposed = np.transpose(self.board)
        for column in transposed:
            winner = self.check_row(column)

        # check diagonal:
        if len(set(self.board[n][n] for n in range(len(self.board))))==1:
            winner = self.board[0][0]

        # check antidiagonal:
        if len(set(self.board[n][len(self.board)-n-1] for n in range(len(self.board))))==1:
            winner = self.board[0][len(self.board)-1]

        if winner != '':
            return winner

        return None

    def is_terminal(self):

        if self.who_wins():
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


class Player:
    def __init__(self, name):
        if name != 'x' and name != 'o':
            raise IncorrectPlayerNameError(name)
        self.name = name
        self.opponent = None

    def set_opponent(self, player):
        self.opponent = player

    def Minmax(self, state, depth, isMin):
        if state.is_terminal() or depth==0:
                state.set_value(state.heuristic())
                return state
        if isMin: #'x'
            state_successors = state.successors('x')
            for u in state_successors:
                u.set_value(self.Minmax(u, depth-1, not isMin).value)
            return min(state_successors, key=lambda t:t.value)
        else: #'o'
            state_successors = state.successors('o')
            for u in state_successors:
                u.set_value(self.Minmax(u, depth-1, not isMin).value)
            return max(state_successors, key=lambda t:t.value)


# przeniesc logike gry do player
# gra przekazuje playerowi obecny stan
# player ocenia stan zwraca grze nowy stan
# gra waliduje stan, zmienia playera
class Game:
    def __init__(self, starting_players):
        self.curr_state = State()
        self.curr_state.set_curr_playing(starting_players[0])
        self.curr_player = starting_players[0]
        self.curr_player.set_opponent(starting_players[1])

    def set_curr_state(self, new_state):
        self.curr_state = new_state

    def switch_player(self):
        tmp_player = self.curr_player
        self.curr_player = self.curr_player.opponent
        self.curr_player.opponent = tmp_player

    def gameplay(self, depth):
        while not self.curr_state.is_terminal():
            print("current state of the game")
            print(self.curr_state)
            isMin = True if self.curr_player.name == 'x' else False
            playersMove = self.curr_player.Minmax(self.curr_state, depth, isMin)
            self.set_curr_state(playersMove)
            self.switch_player()
        print(f'winner: {self.curr_state.who_wins()}')
        print("current state of the game")
        print(self.curr_state)




x = Player('x')
o = Player('o')
g = Game([x, o])
g.gameplay(7)