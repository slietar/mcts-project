from pathlib import Path
from time import time_ns

import numpy as np
import wgpu
from wgpu.utils.compute import compute_with_buffers

from .game import Backgammon


playout_count = 65_536 * 2
playout_length = 30

random_arr = np.random.rand(100, playout_count, playout_length).astype(np.float32).ravel()

settings_arr = np.array([
  playout_length,
], dtype=np.uint32)

default_game = Backgammon()
initial_boards_arr = default_game.board.astype(dtype=np.int32).copy()[None, :].repeat(playout_count, axis=0).ravel()

out_arr = np.array([0.0], dtype=np.float32)


# # print(initial_boards_arr.reshape((2, -1)))

# final_boards = np.frombuffer(output[2], dtype=np.int32).reshape((playout_count, -1))


# for board in final_boards:
#   b = Backgammon(board, turn_p0=False)
#   # print(b.legal_moves(4).nonzero()[0])
#   b.print()

#   print(f'Integrity: {b.check_integrity()}')


device = wgpu.utils.get_default_device()

adapters = wgpu.gpu.enumerate_adapters_sync()
shader = device.create_shader_module(code=(Path(__file__).parent / 'shader.wgsl').read_text())

buffer0 = device.create_buffer_with_data(
  data=random_arr,
  usage=wgpu.BufferUsage.STORAGE,
)

buffer1 = device.create_buffer_with_data(
  data=settings_arr,
  usage=wgpu.BufferUsage.STORAGE,
)

buffer2 = device.create_buffer_with_data(
    data=initial_boards_arr,
    usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
)

buffer3 = device.create_buffer(
    size=out_arr.nbytes,
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
        "resource": {"buffer": buffer0, "offset": 0, "size": buffer0.size},
    },
    {
        "binding": 1,
        "resource": {"buffer": buffer1, "offset": 0, "size": buffer1.size},
    },
    {
        "binding": 2,
        "resource": {"buffer": buffer2, "offset": 0, "size": buffer2.size},
    },
    {
        "binding": 3,
        "resource": {"buffer": buffer3, "offset": 0, "size": buffer3.size},
    },
]

bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
bind_group = device.create_bind_group(layout=bind_group_layout, entries=bindings)

# Create and run the pipeline
compute_pipeline = device.create_compute_pipeline(
    layout=pipeline_layout,
    compute={"module": shader, "entry_point": "main"},
)


t0 = time_ns()

command_encoder = device.create_command_encoder()

compute_pass = command_encoder.begin_compute_pass()
compute_pass.set_pipeline(compute_pipeline)
compute_pass.set_bind_group(0, bind_group)
compute_pass.dispatch_workgroups(playout_count % 65_536, 2, 1)
compute_pass.end()

device.queue.submit([command_encoder.finish()])

# Read result
# result = buffer2.read_data().cast("i")

out = device.queue.read_buffer(buffer2).cast('i')
result = np.frombuffer(out, dtype=np.int32).reshape((playout_count, -1))
# print(result)

t1 = time_ns()
print(f'Time: {((t1 - t0) / 1e6 / playout_count):.4f} ms/playout')


# CPU

t2 = time_ns()

for _ in range(playout_count):
  game = default_game.copy()

  for _ in range(playout_length):
    distance = np.random.randint(1, 7)
    moves = game.legal_moves(4).nonzero()[0]

    if len(moves) > 0:
      game.play(moves[np.random.randint(len(moves))], 4)

t3 = time_ns()
print(f'Time: {((t3 - t2) / 1e6 / playout_count):.4f} ms/playout')
