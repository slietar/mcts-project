from dataclasses import dataclass
import numpy as np
import random

QUARTER_BOARD_LENGTH = 6
HALF_BOARD_LENGTH = QUARTER_BOARD_LENGTH * 2
FULL_BOARD_LENGTH = HALF_BOARD_LENGTH * 2
PLAYER_PIECE_COUNT = 15

@dataclass
class Backgammon:
    board: np.ndarray
    p0_turn: bool = True

    def __init__(self):
        self.board = np.array([
            0,  # captured pieces of player 0 (always >= 0)
            2, 0, 0, 0, 0, -5,
            0, -3, 0, 0, 0, 5,
            -5, 0, 0, 0, 3, 0,
            5, 0, 0, 0, 0, -2,
            0,  # captured pieces of player 1 (always <= 0)
        ])
        assert self.check_integrity()

    def check_integrity(self):
        return (
            (np.maximum(self.board, 0).sum() == PLAYER_PIECE_COUNT)
            and (np.maximum(-self.board, 0).sum() == PLAYER_PIECE_COUNT)
            and (self.board[0] >= 0)
            and (self.board[-1] <= 0)
            and not (self.p0_won() and self.p0_turn)
            and not (self.p1_won() and not self.p0_turn)
        )

    def p0_won(self):
        return (self.board[:-(QUARTER_BOARD_LENGTH + 1)] <= 0).all()

    def p1_won(self):
        return (self.board[(QUARTER_BOARD_LENGTH + 1):] >= 0).all()

    def is_finished(self):
        if self.p0_won():
            return 0
        elif self.p1_won():
            return 1
        else:
            return None

    def legal_moves(self, distance: int):
        start_columns = np.arange(len(self.board))

        if self.p0_turn:
            return (
                ((start_columns + distance) < FULL_BOARD_LENGTH + 1)
                & (self.board > 0)
                & (self.board[np.minimum(start_columns + distance, len(self.board) - 1)] >= -1)
                & ((self.board[0] <= 0) | (start_columns == 0))
            )
        else:
            return (
                ((start_columns - distance) >= 1)
                & (self.board < 0)
                & (self.board[np.maximum(start_columns - distance, 0)] <= 1)
                & ((self.board[-1] >= 0) | (start_columns == FULL_BOARD_LENGTH + 1))
            )

    def play(self, start_column: int, distance: int):
        assert self.check_integrity()
        assert self.legal_moves(distance)[start_column]

        if self.p0_turn:
            end_column = start_column + distance
            self.board[start_column] -= 1
            if self.board[end_column] == -1:
                self.board[end_column] = 1
                self.board[-1] -= 1
            else:
                self.board[end_column] += 1
        else:
            end_column = start_column - distance
            self.board[start_column] += 1
            if self.board[end_column] == 1:
                self.board[end_column] = -1
                self.board[0] += 1
            else:
                self.board[end_column] -= 1

        self.p0_turn = not self.p0_turn
        assert self.check_integrity()

    def print(self):
        COLOR_BRIGHT_BLACK = '\033[90m'
        COLOR_RED = '\033[31m'
        COLOR_RESET = '\033[0m'
        output = ''
        line_count = 5
        width = HALF_BOARD_LENGTH * 2 + 3
        output += f'{COLOR_BRIGHT_BLACK}{"x" * self.board[0]}{COLOR_RESET}{" " * (width - self.board[0] + self.board[-1])}{COLOR_RED}{"x" * -self.board[-1]}{COLOR_RESET}\n\n'
        for second_half in [False, True]:
            for line in range(line_count):
                for col in range(HALF_BOARD_LENGTH):
                    value = self.board[(HALF_BOARD_LENGTH + 1 + col) if second_half else (HALF_BOARD_LENGTH - col)]
                    threshold = ((line_count - line) if second_half else (line + 1))
                    extra = (abs(value) > line_count) and (threshold == line_count)
                    symbol = f'{abs(value) - line_count + 1}'.rjust(2) if extra else ' x'
                    if value >= threshold:
                        output += f'{COLOR_BRIGHT_BLACK}{symbol}{COLOR_RESET}'
                    elif value <= -threshold:
                        output += f'{COLOR_RED}{symbol}{COLOR_RESET}'
                    else:
                        output += '  '
                    if col == QUARTER_BOARD_LENGTH - 1:
                        output += ' |'
                if line < line_count - 1:
                    output += '\n'
            if not second_half:
                output += '\n'
        print(output)

# Game execution
b = Backgammon()
while not b.is_finished():
    distances = np.random.permutation(6) + 1
    for distance in distances:
        legal_moves = b.legal_moves(distance).nonzero()[0]
        if len(legal_moves) == 0:
            continue
        move = random.choice(legal_moves)
        b.play(move, distance)
        b.print()
        print('---')
        break
    else:
        break
print('Game over!')