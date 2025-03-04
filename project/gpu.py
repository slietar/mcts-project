from pathlib import Path

import numpy as np
from wgpu.utils.compute import compute_with_buffers

from .game import Backgammon


playout_count = 1
playout_length = 5

random_arr = np.random.rand(100, playout_count, playout_length).astype(np.float32).ravel()

settings_arr = np.array([
  playout_length,
], dtype=np.uint32)

initial_boards_arr = np.array([
    0,
    2, 0, 0, 0, 0, -5,
    0, -3, 0, 0, 0, 5,
    -5, 0, 0, 0, 3, 0,
    5, 0, 0, 0, 0, -2,
    0,
], dtype=np.int32)[None, :].repeat(playout_count, axis=0).ravel()

out_arr = np.array([0.0], dtype=np.float32)


output = compute_with_buffers(
    input_arrays={
      0: random_arr,
      1: settings_arr,
      2: initial_boards_arr,
      3: out_arr,
    },
    output_arrays={
      2: (len(initial_boards_arr), 'i'),
      3: (len(out_arr), 'f'),
    },
    shader=(Path(__file__).parent / 'shader.wgsl').read_text(),
    n=(playout_count, 1, 1),
)

final_boards = np.frombuffer(output[2], dtype=np.int32).reshape((playout_count, -1))

# print(final_boards)
# print(f'{final_boards[0, 0]:028b}')

# print(random_arr[0])
# print(np.frombuffer(output[3], dtype=np.float32))


b = Backgammon()
# print(b.legal_moves(4).nonzero()[0])
b.print()

b.board = final_boards[0, :]
b.turn_p0 = False

b.print()
print(b.check_integrity())
