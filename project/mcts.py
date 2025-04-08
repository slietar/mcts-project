from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .game import MAX_DISTANCE, Game, Strategy


@dataclass(slots=True)
class RandomStrategy:
  def play(self, game: Game, distance: int):
    legal_moves = game.legal_moves(distance).nonzero()[0]

    if len(legal_moves) == 0:
      return None

    move_index = np.random.randint(len(legal_moves))
    return legal_moves[move_index]

RANDOM_STRATEGY = RandomStrategy()


@dataclass(slots=True)
class FlatStrategy:
  # Stores the score (= probability that player 0 wins) of each move
  cache: dict[int, float] = field(default_factory=dict, init=False, repr=False)
  playout_count: int = 10

  def play(self, game: Game, distance: int):
    move_hashes = game.legal_moves(distance)
    move_columns = move_hashes.nonzero()[0]

    # if len(move_columns) > 0 and move_columns[0] == 18:
    #   game.print()
    #   raise ValueError('Invalid move')

    # game.print()

    if len(move_hashes) == 0:
      return None

    best_score = -1
    best_move_column: Optional[int] = None

    for move_column, move_hash in zip(move_columns, move_hashes[move_columns]):
      if move_hash in self.cache:
        move_score = self.cache[move_hash]
      else:
        game_moved = game.copy()
        game_moved.play(move_column, distance)

        move_score = 0.0

        for _ in range(self.playout_count):
          game_copy = game_moved.copy()
          p0_win_count = game_copy.run_strategies(RANDOM_STRATEGY, RANDOM_STRATEGY)
          move_score += p0_win_count

        move_score /= self.playout_count
        self.cache[move_hash] = move_score

      if not game.turn_p0:
        move_score = 1 - move_score

      if move_score > best_score:
        best_score = move_score
        best_move_column = move_column

    return best_move_column



# if __name__ == '__main__':
#   game = Game()

#   distance = 1
#   legal_moves = game.legal_moves(distance)

#   game.print()

#   # print(game.hash)
#   # print('Board', game.board.reshape(2, -1), sep='\n')

#   for i in legal_moves.nonzero()[0]:
#     game_copy = game.copy()
#     # print(game_copy.board)
#     game_copy.play(i, distance)
#     # print(i, game_copy.hash, legal_moves[i])
#     assert game_copy.hash == legal_moves[i]
#     # print(game_copy.board)
#     # game_copy.print()

#     # print('Board', game_copy.board.reshape(2, -1), sep='\n')
#     # break


def measure(strategy1: Strategy, strategy2: Strategy, /, *, repeat_count: int = 1000):
  strategy1_win_count = 0

  for _ in range(repeat_count):
    game = Game()
    p0_win_count = game.run_strategies(
      deepcopy(strategy1),
      deepcopy(strategy2),
    )

    strategy1_win_count += 1 - p0_win_count

  return strategy1_win_count / repeat_count


if __name__ == '__main__':
  print(measure(FlatStrategy(), FlatStrategy(), repeat_count=1))
