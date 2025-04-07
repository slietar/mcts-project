from pathlib import Path
from time import time_ns
from typing import Protocol

import numpy as np
import wgpu

from .game import BOARD_SIZE, MAX_DISTANCE, Game


MAX_PLAYOUT_COUNT = 65_536
MAX_PLAYOUT_LENGTH = 100
RANDOM_NUM_COUNT_PER_MOVE = 2


class PlayoutEngine(Protocol):
  def __call__(self, game: Game, *, generator: np.random.Generator, playout_count: int) -> float:
    ...


class GPUPlayoutEngine:
  def __init__(self):
    self._device = wgpu.utils.get_default_device()
    shader = self._device.create_shader_module(code=(Path(__file__).parent / 'shader.wgsl').read_text())

    self._buffer0 = self._device.create_buffer(
      size=(MAX_PLAYOUT_COUNT * MAX_PLAYOUT_LENGTH * RANDOM_NUM_COUNT_PER_MOVE * 4),
      usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )

    self._buffer1 = self._device.create_buffer(
      size=4,
      usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )

    self._buffer2 = self._device.create_buffer(
        size=(BOARD_SIZE * MAX_PLAYOUT_COUNT * 4),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
    )

    self._buffer3 = self._device.create_buffer(
        size=(8 + BOARD_SIZE * 4),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )

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

    bind_group_layout = self._device.create_bind_group_layout(entries=binding_layouts)
    pipeline_layout = self._device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
    self._bind_group = self._device.create_bind_group(layout=bind_group_layout, entries=bindings)

    # Create and run the pipeline
    self._compute_pipeline = self._device.create_compute_pipeline(
        layout=pipeline_layout,
        compute=dict(
          entry_point='main',
          module=shader,
        ),
    )

  def __call__(self, game: Game, *, generator: np.random.Generator, playout_count: int):
    max_playout_length = MAX_PLAYOUT_LENGTH - game.length

    assert playout_count <= 65_536


    # Write buffers

    boards_arr = (
      game.board
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

    self._device.queue.write_buffer(self._buffer0, 0, random_arr.tobytes())
    self._device.queue.write_buffer(self._buffer1, 0, settings_arr.tobytes())
    self._device.queue.write_buffer(self._buffer2, 0, boards_arr.tobytes())


    # Submit the compute pass

    command_encoder = self._device.create_command_encoder()

    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(self._compute_pipeline)
    compute_pass.set_bind_group(0, self._bind_group)
    compute_pass.dispatch_workgroups(playout_count, 1, 1)
    compute_pass.end()

    self._device.queue.submit([command_encoder.finish()])


    # Read results

    out = self._device.queue.read_buffer(self._buffer3, 0, 8 + BOARD_SIZE * 4).cast('i')
    stats = np.frombuffer(out, dtype=np.int32) #.reshape((playout_count, -1))

    # print(stats[0])
    # print(f'{stats=}')
    print(stats[1] / stats[0])

    # b = Backgammon(stats[2:], turn_p0=False)
    # b.print()


class CPUPlayoutEngine:
  def __call__(self, game: Game, *, generator: np.random.Generator, playout_count: int):
    max_playout_length = MAX_PLAYOUT_LENGTH - game.length
    random_arr = generator.random(
      dtype=np.float32,
      size=(playout_count, RANDOM_NUM_COUNT_PER_MOVE * max_playout_length)
    ).ravel()

    p0_win_count = 0
    total_length = 0

    for playout_id in range(playout_count):
      current_game = game.copy()
      current_random_index = playout_id * RANDOM_NUM_COUNT_PER_MOVE * max_playout_length

      for _ in range(max_playout_length):
        distance = int(random_arr[current_random_index] * MAX_DISTANCE) + 1
        current_random_index += 1

        legal_moves = current_game.legal_moves(distance).nonzero()[0]

        if len(legal_moves) == 0:
          current_game.length += 1
          continue

        move_index = int(random_arr[current_random_index] * len(legal_moves))
        current_random_index += 1

        move = legal_moves[move_index]

        current_game.play(move, distance)
        current_game.turn_p0 = True

        winner = current_game.winner()

        if winner is not None:
          p0_win_count += 1
          total_length += current_game.length
          break

      # current_game.print()

    # print(p0_win_count / playout_count)
    # print(p0_win_count)
    print(total_length / p0_win_count)


start_game = Game()
playout_count = 200


# GPU
print('GPU')

generator = np.random.default_rng(0)
playout_engine_gpu = GPUPlayoutEngine()

t0 = time_ns()
playout_engine_gpu(start_game, generator=generator, playout_count=playout_count)
t1 = time_ns()

print(f'Time: {((t1 - t0) / 1e6 / playout_count):.4f} ms/playout')


# CPU
print('\nCPU')

generator = np.random.default_rng(0)
playout_engine_cpu = CPUPlayoutEngine()

t2 = time_ns()
x = playout_engine_cpu(start_game, generator=generator, playout_count=playout_count)
t3 = time_ns()
print(f'Time: {((t3 - t2) / 1e6 / playout_count):.4f} ms/playout')

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
