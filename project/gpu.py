from pathlib import Path
from time import time_ns
from typing import Protocol

import numpy as np
import wgpu

from .game import BOARD_SIZE, MAX_DISTANCE, Game


MAX_PLAYOUT_COUNT = 65_535
MAX_PLAYOUT_LENGTH = 600
RANDOM_NUM_COUNT_PER_MOVE = 2


class PlayoutEngine(Protocol):
  def __call__(self, game: Game, *, generator: np.random.Generator, playout_count: int) -> float:
    ...


class GPUBackend:
  def __init__(self):
    self.device = wgpu.utils.get_default_device()
    shader = self.device.create_shader_module(code=(Path(__file__).parent / 'shader.wgsl').read_text())

    binding_layouts = [
        {
            "binding": 0,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.read_only_storage,
            },
        },
        {
            "binding": 1,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.read_only_storage,
            },
        },
        {
            "binding": 2,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.storage,
            },
        },
        {
            "binding": 3,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.storage,
            },
        },
    ]

    self.bind_group_layout = self.device.create_bind_group_layout(entries=binding_layouts)
    self.pipeline_layout = self.device.create_pipeline_layout(bind_group_layouts=[self.bind_group_layout])

    self.compute_pipeline = self.device.create_compute_pipeline(
        layout=self.pipeline_layout,
        compute=dict(
          entry_point='main',
          module=shader,
        ),
    )

  @classmethod
  def get(cls):
    if not hasattr(cls, '_instance'):
      cls._instance = cls()

    return cls._instance


class GPUPlayoutEngine:
  def __init__(self):
    self._backend = GPUBackend.get()

    self._buffer0 = self._backend.device.create_buffer(
      size=(MAX_PLAYOUT_COUNT * MAX_PLAYOUT_LENGTH * RANDOM_NUM_COUNT_PER_MOVE * 4),
      usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )

    self._buffer1 = self._backend.device.create_buffer(
      size=4,
      usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )

    self._buffer2 = self._backend.device.create_buffer(
        size=(BOARD_SIZE * MAX_PLAYOUT_COUNT * 4),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
    )

    self._buffer3 = self._backend.device.create_buffer(
        size=(12 + BOARD_SIZE * 4),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
    )

    bindings = [
        {
            "binding": 0,
            "resource": {"buffer": self._buffer0, "offset": 0, "size": self._buffer0.size},
        },
        {
            "binding": 1,
            "resource": {"buffer": self._buffer1, "offset": 0, "size": self._buffer1.size},
        },
        {
            "binding": 2,
            "resource": {"buffer": self._buffer2, "offset": 0, "size": self._buffer2.size},
        },
        {
            "binding": 3,
            "resource": {"buffer": self._buffer3, "offset": 0, "size": self._buffer3.size},
        },
    ]

    self._bind_group = self._backend.device.create_bind_group(layout=self._backend.bind_group_layout, entries=bindings)

  def __call__(self, game: Game, *, generator: np.random.Generator, playout_count: int):
    target_game = game.transpose() if not game.turn_p0 else game

    max_playout_length = MAX_PLAYOUT_LENGTH - target_game.length


    # Write buffers

    boards_arr = (
      target_game.board
        .astype(dtype=np.int32)
        .copy()
        [None, :]
        .repeat(playout_count, axis=0)
        .ravel()
    )

    settings_arr = np.array([max_playout_length], dtype=np.uint32)

    random_arr = generator.random(
      dtype=np.float32,
      size=(playout_count, RANDOM_NUM_COUNT_PER_MOVE * max_playout_length)
    ).ravel()

    self._backend.device.queue.write_buffer(self._buffer0, 0, random_arr.tobytes())
    self._backend.device.queue.write_buffer(self._buffer1, 0, settings_arr.tobytes())
    self._backend.device.queue.write_buffer(self._buffer2, 0, boards_arr.tobytes())
    self._backend.device.queue.write_buffer(self._buffer3, 0, b'\0' * self._buffer3.size)


    # Submit the compute pass

    command_encoder = self._backend.device.create_command_encoder()

    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(self._backend.compute_pipeline)
    compute_pass.set_bind_group(0, self._bind_group)
    compute_pass.dispatch_workgroups(playout_count, 1, 1)
    compute_pass.end()

    self._backend.device.queue.submit([command_encoder.finish()])


    # Read results

    out = self._backend.device.queue.read_buffer(self._buffer3, 0, 12 + BOARD_SIZE * 4).cast('i')
    stats = np.frombuffer(out, dtype=np.int32)

    # Completed playout fraction
    # print(stats[0] / playout_count)

    # print(stats[0])
    # print(f'{stats=}')
    # print(stats[1] / stats[0])

    # print(stats[2:])
    # g = Game(stats[3:], turn_p0=(stats[2] % 2 == 0))

    # if not game.turn_p0:
    #   g = g.transpose()

    # g.print()

    # print('Average playout length:', stats[2] / stats[0])

    p0_win_fraction = stats[1] / stats[0]

    if not game.turn_p0:
      p0_win_fraction = 1 - p0_win_fraction

    # t1 = time_ns()
    # print((t1 - t0) / 1e6 / playout_count, 'ms/playout')

    return p0_win_fraction


class CPUPlayoutEngine:
  def __call__(self, game: Game, *, generator: np.random.Generator, playout_count: int):
    max_playout_length = MAX_PLAYOUT_LENGTH - game.length
    random_arr = generator.random(
      dtype=np.float32,
      size=(playout_count, RANDOM_NUM_COUNT_PER_MOVE * max_playout_length)
    ).ravel()

    completed_game_count = 0
    p0_win_count = 0
    total_length = 0

    for playout_id in range(playout_count):
      current_game = game.copy()

      # 2x slower
      # current_game.run_strategies(RANDOM_STRATEGY, RANDOM_STRATEGY)
      # continue

      playout_random_index = playout_id * RANDOM_NUM_COUNT_PER_MOVE * max_playout_length

      for current_length in range(max_playout_length):
        step_random_index = playout_random_index + current_length * RANDOM_NUM_COUNT_PER_MOVE

        distance = int(random_arr[step_random_index] * MAX_DISTANCE) + 1

        legal_moves = current_game.legal_moves(distance).nonzero()[0]

        if len(legal_moves) == 0:
          current_game.play_skip()
          continue

        move_index = int(random_arr[step_random_index + 1] * len(legal_moves))
        move = legal_moves[len(legal_moves) - 1 - move_index] if not current_game.turn_p0 else legal_moves[move_index]
        # print('->', distance, move_index, legal_moves)
        # move = legal_moves[move_index]
        current_game.play(move, distance)
        # print(distance, legal_moves)

        current_game_p0_win_count = current_game.p0_win_count()

        if current_game_p0_win_count is not None:
          completed_game_count += 1
          p0_win_count += current_game_p0_win_count
          total_length += current_game.length
          break
      # else:
      #   print('Failed to find a winner')

      # current_game.print()
      # break

    # current_game.print()
    # x = current_game.copy()
    # x.transpose()
    # x.print()

    # print(p0_win_count / playout_count)
    # print(p0_win_count)
    # print(total_length / p0_win_count)

    # print('Average playout length:', total_length / completed_game_count)

    return p0_win_count / completed_game_count


if __name__ == '__main__':
  start_game = Game()
  # start_game.play_skip()
  playout_count = 1000

  # print(start_game.board)
  # start_game.print()


  # GPU
  if True:
    print('GPU')

    generator = np.random.default_rng(0)
    playout_engine_gpu = GPUPlayoutEngine()

    t0 = time_ns()
    x = playout_engine_gpu(start_game, generator=generator, playout_count=playout_count)
    t1 = time_ns()

    print(f'Time: {((t1 - t0) / 1e6 / playout_count):.4f} ms/playout')
    print(f'Win fraction: {x}')


  # CPU
  if True:
    print('\nCPU')

    generator = np.random.default_rng(0)
    playout_engine_cpu = CPUPlayoutEngine()

    t2 = time_ns()
    x = playout_engine_cpu(start_game, generator=generator, playout_count=playout_count)
    t3 = time_ns()

    print(f'Time: {((t3 - t2) / 1e6 / playout_count):.4f} ms/playout')
    print(f'Win fraction: {x}')





  # print(x)


  # generator = np.random.default_rng(0)
  # playout_engine_gpu = GPUPlayoutEngine()


  # async def main():
  #   t0 = time_ns()
  #   repeat = 5

  #   async with asyncio.TaskGroup() as group:
  #     for _ in range(repeat):
  #       group.create_task(
  #         asyncio.to_thread(lambda: playout_engine_gpu(start_game, generator=generator, playout_count=playout_count))
  #       )

  #   t1 = time_ns()
  #   print(f'Time: {((t1 - t0) / 1e6 / playout_count / repeat):.4f} ms/playout')

  # import asyncio
  # asyncio.run(main())

  # # def run(_):
  # #   playout_engine_gpu(start_game, generator=generator, playout_count=playout_count)

  # # if __name__ == '__main__':
  # #   import concurrent.futures

  # #   with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
  # #     for future in executor.map(run, range(5)):
  # #       pass
