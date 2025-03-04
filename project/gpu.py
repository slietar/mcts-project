from pathlib import Path
import numpy as np
from wgpu.utils.compute import compute_with_buffers


from .game import Backgammon

playout_count = 1
playout_length = 5

random_arr = np.random.randint(0, 6, (playout_count, playout_length), dtype=np.uint32).ravel()

settings_arr = np.array([
  playout_length
], dtype=np.uint32)

initial_boards_arr = np.array([
    0,
    2, 0, 0, 0, 0, -5,
    0, -3, 0, 0, 0, 5,
    -5, 0, 0, 0, 3, 0,
    5, 0, 0, 0, 0, -2,
    0,
], dtype=np.int32)[None, :].repeat(playout_count, axis=0).ravel()


output = compute_with_buffers(
    input_arrays={
      0: random_arr,
      1: settings_arr,
      2: initial_boards_arr,
    },
    output_arrays={2: (len(initial_boards_arr), "i")},
    shader=(Path(__file__).parent / 'shader.wgsl').read_text(),
    n=(playout_count, 1, 1),
)

# print(output)

# # get output

final_boards = np.frombuffer(output[2], dtype=np.int32).reshape((playout_count, -1))

print(final_boards)

# # check that results are the same as numpy, we can expect 7 decimal precision
# all_close = np.allclose(A @ B, C)
# assert all_close
# print(f"np.allclose():\n {all_close}\n")
# print(f"AB - C:\n{A @ B - C}\n")
# diff_norms = np.linalg.norm(A @ B - C, ord="fro") / np.linalg.norm(A @ B, ord="fro")
# print(f"||AB - C||_F - ||AB||_F:\n{diff_norms}\n")
# print(f"C:\n{C}")


b = Backgammon()
b.print()

b.board = final_boards[0, :]
b.turn_p0 = True

b.print()
print(b.check_integrity())
