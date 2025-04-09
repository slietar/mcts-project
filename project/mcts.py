import concurrent.futures
import math
from copy import deepcopy
from dataclasses import dataclass, field
from time import time
from typing import Optional

import numpy as np

from .game import Game, Strategy
from .gpu import GPUPlayoutEngine, PlayoutEngine


@dataclass(slots=True)
class RandomStrategy:
  def copy(self):
    return RandomStrategy()

  def play(self, game: Game, distance: int):
    legal_moves = game.legal_moves(distance).nonzero()[0]

    if len(legal_moves) == 0:
      return None

    move_index = np.random.randint(len(legal_moves))
    return legal_moves[move_index]

RANDOM_STRATEGY = RandomStrategy()


generator = np.random.default_rng(0)

@dataclass(slots=True)
class FlatStrategy:
  # Stores the score (= probability that player 0 wins) of each move
  cache: dict[int, float] = field(default_factory=dict, init=False, repr=False)
  playout_count: int = 1000
  playout_engine: PlayoutEngine = field(default_factory=GPUPlayoutEngine, init=False, repr=False)

  def copy(self):
    return FlatStrategy(playout_count=self.playout_count)

  def play(self, game: Game, distance: int):
    move_hashes = game.legal_moves(distance)
    move_columns = move_hashes.nonzero()[0]

    best_score = -1
    best_move_column: Optional[int] = None

    for move_column, move_hash in zip(move_columns, move_hashes[move_columns]):
      if move_hash in self.cache:
        move_score = self.cache[move_hash]
      else:
        game_moved = game.copy()
        game_moved.play(move_column, distance)

        # move_score = 0.0

        # for _ in range(self.playout_count):
        #   game_copy = game_moved.copy()
        #   p0_win_count = game_copy.run_strategies(RANDOM_STRATEGY, RANDOM_STRATEGY)
        #   move_score += p0_win_count

        move_score = self.playout_engine(game_moved, generator=generator, playout_count=self.playout_count)

        # move_score /= self.playout_count
        self.cache[move_hash] = move_score

      if not game.turn_p0:
        move_score = 1 - move_score

      if move_score > best_score:
        best_score = move_score
        best_move_column = move_column

    return best_move_column


@dataclass(slots=True)
class Node:
  game: Game
  parent_hash: Optional[int]
  p0_win_count: int = 0
  simulation_count: int = 0

@dataclass(slots=True)
class UCTStrategy:
  exploration_budget: int = 10
  exploration_constant: float = math.sqrt(2)
  nodes: dict[int, Node] = field(default_factory=dict, init=False, repr=False)

  def play(self, game: Game, distance: int):
    move_hashes = game.legal_moves(distance)

    if not game.hash in self.nodes:
      self.nodes[game.hash] = Node(game, parent_hash=None)

    for _ in range(self.exploration_budget):
      current_node = self.nodes[game.hash]
      ancestors = [current_node]

      while True:
        current_move_hashes = current_node.game.legal_moves(distance)

        def get_uct_score(node: Node):
          return (
              node.game.translate_win_fraction(node.p0_win_count) / node.simulation_count
            + self.exploration_constant * math.sqrt(math.log(current_node.simulation_count) / node.simulation_count)
          )

        max_uct_score_move_column = -1
        max_uct_score = -1

        for move_column, move_hash in enumerate(current_move_hashes):
          if move_hash == 0:
            continue

          child_node = self.nodes.get(move_hash)

          if child_node is None:
            child_game = current_node.game.copy()
            child_game.play(move_column, distance)

            child_node = Node(child_game, parent_hash=current_node.game.hash)
            self.nodes[move_hash] = child_node

            ancestors.append(child_node)
            max_uct_score_move_column = move_column
            break

          ancestors.append(child_node)
          child_uct_score = get_uct_score(child_node)

          if child_uct_score > max_uct_score:
            max_uct_score = child_uct_score
            max_uct_score_move_column = move_column

        selected_node = self.nodes[move_hashes[max_uct_score_move_column]]
        game_copy = selected_node.game.copy()
        p0_win_count = game_copy.run_strategies(RANDOM_STRATEGY, RANDOM_STRATEGY)

        selected_node.simulation_count += 1
        selected_node.p0_win_count += p0_win_count

    children_nodes = [self.nodes[move_hash] for move_hash in move_hashes if move_hash in self.nodes]
    p0_win_fractions = np.array([child_node.p0_win_count / child_node.simulation_count for child_node in children_nodes])

    return np.argmax(game.translate_win_fraction(p0_win_fractions))


def measure(strategy1: Strategy, strategy2: Strategy, /, *, repeat_count: int = 1000):
  strategy1_win_count = 0

  for _ in range(repeat_count):
    game = Game()

    s1 = strategy1.copy()
    s2 = s1 if strategy1 is strategy2 else strategy2.copy()

    p0_win_count = game.run_strategies(s1, s2)

    strategy1_win_count += p0_win_count

    print('Length', game.length)
    game.print()

  return strategy1_win_count / repeat_count


def measure_game(strategy1: Strategy, strategy2: Strategy, /):
  return Game().run_strategies(strategy1, strategy2)

def measure_multithreaded(strategy1: Strategy, strategy2: Strategy, /, *, repeat_count: int = 1000):
  p0_win_count = 0

  with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = []

    for _ in range(repeat_count):
      s1 = strategy1.copy()
      s2 = s1 if strategy1 is strategy2 else strategy2.copy()

      futures.append(executor.submit(measure_game, s1, s2))

    for future in concurrent.futures.as_completed(futures):
      p0_win_count += future.result()

      # print(future.result())

  return p0_win_count / repeat_count


if __name__ == '__main__':
  t0 = time()
  # print('Strategy 1 win fraction:', measure_multithreaded(FlatStrategy(), RandomStrategy(), repeat_count=10))
  print('Strategy 1 win fraction:', measure(FlatStrategy(), RandomStrategy(), repeat_count=1))
  t1 = time()
  print('Execution time:', t1 - t0)
