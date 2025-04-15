import concurrent.futures
import math
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from time import time
from typing import Optional

import numpy as np
from tqdm import tqdm

from .game import BOARD_SIZE, MAX_DISTANCE, Game, Strategy
from .gpu import GPUBackend, GPUPlayoutEngine, PlayoutEngine


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


# engine = GPUPlayoutEngine()

@dataclass(slots=True)
class Node:
  game: Game
  parent_hash: Optional[int]

  # Positive if p0 wins
  amaf: np.ndarray = field(default_factory=(lambda: np.zeros((BOARD_SIZE, MAX_DISTANCE), dtype=np.int32)), init=False, repr=False)
  children_simulation_count: int = 0
  p0_win_count: int = 0
  simulation_count: int = 0

@dataclass(slots=True)
class UCTStrategy:
  amaf_proportion: float = 0.5
  expansion_count: int = 10
  exploration_constant: float = math.sqrt(2)
  playout_count: int = 10

  nodes: dict[int, Node] = field(default_factory=dict, init=False, repr=False)

  def copy(self):
    return UCTStrategy()

  def play(self, game: Game, distance: int):
    if not game.hash in self.nodes:
      self.nodes[game.hash] = Node(game, parent_hash=None)

    # print('Root node', game.hash)

    for _ in range(self.expansion_count):
      current_node = self.nodes[game.hash]
      ancestor_nodes = list[tuple[Node, float]]()

      while current_node.game.p0_win_count() is None:
        current_distance = distance if not ancestor_nodes else np.random.randint(MAX_DISTANCE) + 1
        current_move_hashes = current_node.game.legal_moves(current_distance)

        def get_uct_score(node: Node):
          # print(node.game.hash)
          # if current_node.children_simulation_count == 0:
          #   pprint([*ancestor_nodes, current_node, node])
          #   print(current_node.game.p0_win_count())
          #   print(node.game.p0_win_count())
          #   print(current_move_hashes)

          # if node.p0_win_count is not None:
          #   return -1

          if current_node.children_simulation_count == 0:
            return -1

          return (
              node.game.translate_win_fraction(node.p0_win_count) / node.simulation_count
            + self.exploration_constant * math.sqrt(math.log(current_node.children_simulation_count) / node.simulation_count)
          )

        created_child_node = False
        max_score_move_column: Optional[int] = None # None = no move possible, only skip is possible
        max_score: float = -1 # inf = exact choice

        for move_column, move_hash in enumerate(current_move_hashes):
          if move_hash == 0:
            continue

          child_node = self.nodes.get(move_hash)

          if child_node is None:
            child_game = current_node.game.copy()
            child_game.play(move_column, current_distance)

            child_node = Node(child_game, parent_hash=current_node.game.hash)
            self.nodes[child_game.hash] = child_node

            created_child_node = True
            max_score_move_column = move_column
            max_score = math.inf

            break

          # UCT score of current player
          child_uct_score = get_uct_score(child_node)
          child_amaf_score = child_node.amaf[move_column, current_distance - 1] / child_node.simulation_count * (1 if child_node.game.turn_p0 else -1)

          child_score = child_amaf_score * self.amaf_proportion + child_uct_score * (1 - self.amaf_proportion)
          # print(child_uct_score, child_amaf_score)

          if child_score > max_score:
            max_score = child_score # type: ignore
            max_score_move_column = move_column

        # ancestor_weight = 1 / len(current_move_hashes.nonzero()[0]) if current_node.game.turn_p0 != game.turn_p0 else 1
        ancestor_weight = 0.0

        # If no move is possible i.e. current_move_hashes is all zeros
        if max_score_move_column is None:
          child_game = current_node.game.copy()
          # child_game.print()
          child_game.play_skip()
          child_node = self.nodes.get(child_game.hash)

          if child_node is None:
            child_node = Node(child_game, parent_hash=current_node.game.hash)
            self.nodes[child_game.hash] = child_node
            created_child_node = True

          ancestor_nodes.append((current_node, ancestor_weight))
          current_node = child_node
        else:
          ancestor_nodes.append((current_node, ancestor_weight))
          current_node = self.nodes[current_move_hashes[max_score_move_column]]

        if created_child_node:
          break

      amaf_change = np.zeros((BOARD_SIZE, MAX_DISTANCE), dtype=np.int32)
      current_p0_win_count =  current_node.game.p0_win_count()

      if current_p0_win_count is not None:
        simulation_count = 10_000
        p0_win_count = current_p0_win_count

        # print('!')
        # pprint([*ancestor_nodes, current_node])
      else:
        p0_win_count = 0
        simulation_count = self.playout_count

        for _ in range(simulation_count):
          game_copy = current_node.game.copy()
          playout_p0_win_count, playout_counts = game_copy.playout_amaf()

          p0_win_count += playout_p0_win_count
          amaf_change += playout_counts * (playout_p0_win_count * 2 - 1)

        # print(f'Simulation count: {simulation_count}, P0 win count: {p0_win_count}')

      current_node.amaf += amaf_change
      current_node.simulation_count += simulation_count
      current_node.p0_win_count += p0_win_count

      for ancestor_index, (ancestor_node, _) in enumerate(ancestor_nodes):
        ancestor_node.children_simulation_count += simulation_count

        if ancestor_index != 0:
          ancestor_node.amaf += amaf_change
          ancestor_node.p0_win_count += p0_win_count
          ancestor_node.simulation_count += simulation_count

      # print(len(self.nodes))
      # pprint(len(ancestor_nodes))
      # pprint(current_node)

      # pprint([*ancestor_nodes, current_node])
      # pprint()

    move_hashes = game.legal_moves(distance)

    p0_win_fractions = np.array([
      ((child_node := self.nodes[move_hash]).p0_win_count / child_node.simulation_count)
      if (move_hash != 0) and (move_hash in self.nodes) else -1
      for move_hash in move_hashes
    ])

    # print(move_hashes)
    # print(p0_win_fractions)
    # print(p0_win_fractions)

    win_fractions = game.translate_win_fraction(p0_win_fractions)
    # print(win_fractions)
    move = np.argmax(win_fractions)

    if move_hashes[move] == 0:
      return None

    return move


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
  # np.random.seed(0)

  g = Game()
  s1 = UCTStrategy()

  for _ in tqdm(range(1000)):
    if g.p0_win_count() is not None:
      break

    distance = np.random.randint(5) + 1

    if g.turn_p0:
      move = s1.play(g, distance)
    else:
      move = RANDOM_STRATEGY.play(g, distance)

    if move is not None:
      g.play(move, distance)
    else:
      g.play_skip()

    g.print()

  # g.print()

  # t0 = time()
  # print('Strategy 1 win fraction:', measure(UCTStrategy(), RandomStrategy(), repeat_count=1))
  # t1 = time()
  # print('Execution time:', t1 - t0)
