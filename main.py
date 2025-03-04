from dataclasses import dataclass
import time

import numpy as np


xor = lambda x, axis = None: np.bitwise_xor.reduce(x, axis=axis)

QUARTER_BOARD_LENGTH = 6
HALF_BOARD_LENGTH = QUARTER_BOARD_LENGTH * 2
FULL_BOARD_LENGTH = HALF_BOARD_LENGTH * 2
PLAYER_PIECE_COUNT = 15

PIECE_ENCODINGS = np.random.randint(0, 1 << 64 - 1, (FULL_BOARD_LENGTH + 2, PLAYER_PIECE_COUNT * 2 + 1))
TURN_P0_ENCODING = np.random.randint(0, 1 << 64 - 1)

@dataclass(slots=True)
class Backgammon:
  board: np.ndarray
  hash: int
  turn_p0: bool

  def __init__(self):
    self.board = np.array([
      0, # captured pieces of player 0 (always >= 0)
      2, 0, 0, 0, 0, -5,
      0, -3, 0, 0, 0, 5,
      -5, 0, 0, 0, 3, 0,
      5, 0, 0, 0, 0, -2,
      0, # captured pieces of player 1 (always <= 0)
    ])

    self.turn_p0 = True
    self.hash = self.compute_hash()

    assert self.check_integrity()

  def check_integrity(self):
    return (
          (np.maximum(self.board, 0).sum() == PLAYER_PIECE_COUNT)
      and (np.maximum(-self.board, 0).sum() == PLAYER_PIECE_COUNT)
      and (self.board[0] >= 0)
      and (self.board[-1] <= 0)
      and not (self.p0_won() and self.turn_p0)
      and not (self.p1_won() and not self.turn_p0)
    )

  def compute_hash(self):
    return xor(PIECE_ENCODINGS[np.arange(len(self.board)), self.board]) ^ (TURN_P0_ENCODING * self.turn_p0)

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

    if self.turn_p0:
      end_columns = np.minimum(start_columns + distance, len(self.board) - 1)
      old_end_column_values = self.board[end_columns]
      new_end_column_values = np.maximum(old_end_column_values + 1, 1)
      capturing = old_end_column_values < 0

      h = (
          TURN_P0_ENCODING
        ^ PIECE_ENCODINGS[start_columns, self.board]
        ^ PIECE_ENCODINGS[end_columns, old_end_column_values]
        ^ PIECE_ENCODINGS[end_columns, new_end_column_values]
        ^ (PIECE_ENCODINGS[-1, self.board[-1] - 1] * capturing)
      )

      return (
          ((start_columns + distance) < FULL_BOARD_LENGTH + 1)
        & (self.board > 0)
        & (old_end_column_values >= -1)
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

    if self.turn_p0:
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

    self.turn_p0 = not self.turn_p0
    self.hash = self.compute_hash()

    assert self.check_integrity()

  def print(self):
    COLOR_BRIGHT_BLACK = '\033[90m'
    COLOR_RED = '\033[31m'
    COLOR_RESET = '\033[0m'

    # p0 = black
    # p1 = read

    output = ''
    line_count = 5

    width = HALF_BOARD_LENGTH * 2 + 3
    output += f'{COLOR_BRIGHT_BLACK}{'o' * self.board[0]}{COLOR_RESET}{' ' * (width - self.board[0] + self.board[-1])}{COLOR_RED}{'o' * -self.board[-1]}{COLOR_RESET}\n\n'

    for second_half in [False, True]:
      for line in range(line_count):
        for col in range(HALF_BOARD_LENGTH):
          value = self.board[
            (HALF_BOARD_LENGTH + 1 + col) if second_half else (HALF_BOARD_LENGTH - col)
          ]

          threshold = ((line_count - line) if second_half else (line + 1))
          extra = (abs(value) > line_count) and (threshold == line_count)
          symbol = f'{abs(value) - line_count + 1}'.rjust(2) if extra else ' o'

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


b = Backgammon()


while not (b.p0_won() or b.p1_won()):
  distances = np.random.permutation(6) + 1

  for distance in distances:
    legal_moves = b.legal_moves(distance).nonzero()[0]

    if len(legal_moves) == 0:
      continue

    move = legal_moves[[np.random.randint(len(legal_moves))]]
    b.play(move, distance)
    print('\n' * 5)
    print(b.compute_hash())
    b.print()
    time.sleep(0.3)
    break
  else:
    break
