from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
import time
from typing import Optional, Protocol

import numpy as np


xor = lambda x, axis = None: np.bitwise_xor.reduce(x, axis=axis)

QUARTER_BOARD_LENGTH = 6
HALF_BOARD_LENGTH = QUARTER_BOARD_LENGTH * 2
FULL_BOARD_LENGTH = HALF_BOARD_LENGTH * 2
PLAYER_PIECE_COUNT = 15
BOARD_SIZE = FULL_BOARD_LENGTH + 2
MAX_DISTANCE = 6

PIECE_ENCODINGS = np.random.randint(0, 1 << 64 - 1, (FULL_BOARD_LENGTH + 2, PLAYER_PIECE_COUNT * 2 + 1))
P0_TURN_ENCODING = np.random.randint(0, 1 << 64 - 1)


class Strategy(Protocol):
  def play(self, game: 'Game', distance: int) -> Optional[int]:
    ...


@dataclass(slots=True)
class Game:
  board: np.ndarray = field(default_factory=(lambda: np.array([
    0, # captured pieces of player 0 (always >= 0)
    2, 0, 0, 0, 0, -5,
    0, -3, 0, 0, 0, 5,
    -5, 0, 0, 0, 3, 0,
    5, 0, 0, 0, 0, -2,
    0, # captured pieces of player 1 (always <= 0)
  ])))
  hash: int = field(init=False)
  length: int = 0
  turn_p0: bool = True

  def __post_init__(self):
    self.hash = self.compute_hash()
    assert self.check_integrity()

  @property
  def normalized_board(self):
    return self.board[::(1 if self.turn_p0 else -1)] * (1 if self.turn_p0 else -1)

  def check_integrity(self):
    return (
          (np.maximum(self.board, 0).sum() == PLAYER_PIECE_COUNT)
      and (np.maximum(-self.board, 0).sum() == PLAYER_PIECE_COUNT)
      and (self.board[0] >= 0)
      and (self.board[-1] <= 0)
      and not (self.p0_won() and self.turn_p0)
      and not (self.p1_won() and not self.turn_p0)
      and (self.hash == self.compute_hash())
    )

  def compute_hash(self):
    return xor(PIECE_ENCODINGS[
      np.arange(BOARD_SIZE),
      self.board,
    ]) ^ (P0_TURN_ENCODING if self.turn_p0 else 0)

  def copy(self):
    return deepcopy(self)

  def p0_won(self):
    return (self.board[:-(QUARTER_BOARD_LENGTH + 1)] <= 0).all()

  def p1_won(self):
    return (self.board[(QUARTER_BOARD_LENGTH + 1):] >= 0).all()

  def p0_win_count(self):
    if self.p0_won():
      return 0
    elif self.p1_won():
      return 1
    else:
      return None

  def legal_moves(self, distance: int):
    assert 1 <= distance <= MAX_DISTANCE

    start_columns = np.arange(len(self.board))
    old_start_column_values = self.board[start_columns]

    if self.turn_p0:
      end_columns = np.minimum(start_columns + distance, len(self.board) - 1)
      old_end_column_values = self.board[end_columns]

      new_start_column_values = old_start_column_values - 1
      new_end_column_values = np.maximum(old_end_column_values + 1, 1)

      capturing = old_end_column_values < 0
      capture_hash = PIECE_ENCODINGS[-1, self.board[-1]] ^ PIECE_ENCODINGS[-1, self.board[-1] - 1]

      legal = (
          ((start_columns + distance) < FULL_BOARD_LENGTH + 1)
        & (self.board > 0)
        & (old_end_column_values >= -1)
        & ((self.board[0] <= 0) | (start_columns == 0))
      )

    else:
      end_columns = np.maximum(start_columns - distance, 0)
      old_end_column_values = self.board[end_columns]

      new_start_column_values = old_start_column_values + 1
      new_end_column_values = np.minimum(old_end_column_values - 1, -1)

      capturing = old_end_column_values > 0
      capture_hash = PIECE_ENCODINGS[0, self.board[0]] ^ PIECE_ENCODINGS[0, self.board[0] + 1]

      legal = (
          ((start_columns - distance) >= 1)
        & (self.board < 0)
        & (old_end_column_values <= 1)
        & ((self.board[-1] >= 0) | (start_columns == FULL_BOARD_LENGTH + 1))
      )

    hashes = (
        self.hash
      ^ PIECE_ENCODINGS[start_columns, old_start_column_values]
      ^ PIECE_ENCODINGS[start_columns, new_start_column_values]
      ^ PIECE_ENCODINGS[end_columns, old_end_column_values]
      ^ PIECE_ENCODINGS[end_columns, new_end_column_values]
      ^ (capture_hash * capturing)
      ^ P0_TURN_ENCODING
    )

    return legal * hashes


  def play(self, start_column: int, distance: int):
    assert self.check_integrity()
    # assert self.legal_moves(distance)[start_column] != 0
    if __debug__ and (self.legal_moves(distance)[start_column] == 0):
      print(start_column, distance)
      print(self.legal_moves(distance))
      self.print()
      raise ValueError('Invalid move')

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
    self.length += 1

    assert self.check_integrity()

  def play_skip(self):
    assert self.check_integrity()

    self.turn_p0 = not self.turn_p0
    self.hash ^= P0_TURN_ENCODING
    self.length += 1

    assert self.check_integrity()

  def run_strategies(self, p0_strategy: Strategy, p1_strategy: Strategy):
    while True:
      p0_win_count = self.p0_win_count()

      if p0_win_count is not None:
        return p0_win_count

      distance = np.random.randint(MAX_DISTANCE) + 1

      if self.turn_p0:
        move = p0_strategy.play(self, distance)
      else:
        move = p1_strategy.play(self, distance)

      if move is not None:
        self.play(move, distance)
      else:
        self.play_skip()

  def print(self):
    COLOR_BRIGHT_BLACK = '\033[90m'
    COLOR_RED = '\033[31m'
    COLOR_RESET = '\033[0m'

    # p0 = black
    # p1 = read

    output = ''
    line_count = 5

    width = HALF_BOARD_LENGTH * 2 + 3
    output += (
    f"{COLOR_BRIGHT_BLACK}{'o' * self.board[0]}{COLOR_RESET}"
    + ' ' * (width - self.board[0] + self.board[-1])
    + f"{COLOR_RED}{'o' * -self.board[-1]}{COLOR_RESET}\n\n"
    )

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

  def translate_win_fraction(self, p0_win_count: int | np.ndarray):
    return p0_win_count if self.turn_p0 else 1 - p0_win_count

  def transpose(self):
    self.board = -self.board[::-1].copy()
    self.turn_p0 = not self.turn_p0

    assert self.check_integrity()


if __name__ == '__main__':
  # print({
  #   d: b.legal_moves(d).sum() for d in range(1, 7)
  # })

  b = Game()
  step = 0

  while not (b.p0_won() or b.p1_won()):
    distances = np.random.permutation(6) + 1

    for distance in distances:
      legal_moves = b.legal_moves(distance).nonzero()[0]

      if len(legal_moves) == 0:
        continue

      move = legal_moves[[np.random.randint(len(legal_moves))]]
      b.play(move, distance)
      # print('\n' * 5)
      # print(b.compute_hash())
      # b.print()
      # time.sleep(0.3)
      step += 1
      break
    else:
      break

  b.print()

  print(step, b.p0_win_count())
