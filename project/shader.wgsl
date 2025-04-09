const blockSize: u32 = 16;
const quarterBoardLength: u32 = 6;
const halfBoardLength: u32 = quarterBoardLength * 2;
const fullBoardLength: u32 = halfBoardLength * 2;
const boardSize: u32 = fullBoardLength + 2;
const playerPieceCount: u32 = 15;
const maxDistance: u32 = 6;
const randomNumCountPerMove: u32 = 2;


struct Settings {
  maxPlayoutLength: u32,
}

struct Stats {
  sumP0Win: atomic<u32>,
  sumPlayoutLength: atomic<u32>,
  out: array<i32>,
}

@group(0) @binding(0)
var<storage, read> random: array<f32>;

@group(0) @binding(1)
var<storage, read> settings: Settings;

@group(0) @binding(2)
var<storage, read_write> initialBoards: array<i32>;

@group(0) @binding(3)
var<storage, read_write> stats: Stats;


// n = 0 -> returns the index of the first one bit
fn indexOfNthOneBit(val: u32, n: u32) -> u32 {
  var shiftedVal = val;
  var index = 0u;

  for (var oneIndex = 0u; oneIndex <= n; oneIndex += 1u) {
    let advance = countTrailingZeros(shiftedVal) + 1u;
    index += advance;
    shiftedVal >>= advance;
  }

  return index - 1u;
}


@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let playoutId = gid.x;

  let playoutRandomIndex = settings.maxPlayoutLength * playoutId * randomNumCountPerMove;
  var turnP0 = true;

  var board: array<i32, boardSize>;

  for (var index = 0u; index < boardSize; index += 1u) {
    board[index] = initialBoards[playoutId * boardSize + index];
  }


  var packedEmpty = 0u;
  var packedP0 = 0u;
  var packedMany = 0u;

  for (var column = 0u; column < boardSize; column += 1u) {
    let value = board[column];

    if value == 0 {
      packedEmpty |= 1u << column;
    }

    if value >= 1 {
      packedP0 |= 1u << column;
    }

    if abs(value) > 1 {
      packedMany |= 1u << column;
    }
  }


  for (var currentLength = 0u; currentLength < settings.maxPlayoutLength; currentLength += 1u) {
    let stepRandomIndex = playoutRandomIndex + currentLength * randomNumCountPerMove;
    let distance = u32(random[stepRandomIndex] * f32(maxDistance)) + 1u;
    // let distance = 4u;

    var startColumn: u32; // Sentinel value for no move possible = boardSize
    var endColumn: u32;

    if board[0] > 0 {
      startColumn = 0u;
      endColumn = startColumn + distance;

      if board[endColumn] < -1 {
        startColumn = boardSize;
      }
    } else {
      let packedLegal =
          packedP0
        & (((1u << (boardSize - 1u)) - 1u) >> distance)
        & (
            (packedEmpty >> distance)
          | (packedP0 >> distance)
          | ~(packedMany >> distance)
        );

      let moveCount = countOneBits(packedLegal);

      if moveCount > 0u {
        let moveIndex = u32(random[stepRandomIndex + 1] * f32(moveCount));

        startColumn = indexOfNthOneBit(packedLegal, moveIndex);
        endColumn = startColumn + distance;

        board[startColumn] -= 1;

        if board[endColumn] == -1 {
          board[endColumn] = 1;
          board[boardSize - 1u] -= 1;
        } else {
          board[endColumn] += 1;
        }
      } else {
        startColumn = boardSize;
      }
    }

    if startColumn != boardSize {
      let startMask = 1u << startColumn;
      let endMask = 1u << endColumn;

      if board[startColumn] > 1 {
        packedEmpty &= ~startMask;
        packedP0 |= startMask;
        packedMany |= startMask;
      } else if board[startColumn] > 0 {
        packedEmpty &= ~startMask;
        packedP0 |= startMask;
        packedMany &= ~startMask;
      } else {
        packedEmpty |= startMask;
        packedP0 &= ~startMask;
        packedMany &= ~startMask;
      }

      packedEmpty &= ~endMask;
      packedP0 |= endMask;

      if board[endColumn] > 1 {
        packedMany |= endMask;
      } else {
        packedMany &= ~endMask;
      }
    }


    // Check if P0 has won

    let fourthQuarterMask = (((1u << quarterBoardLength) - 1u) << (quarterBoardLength * 3u + 1u));
    let wonP0 = (packedP0 & fourthQuarterMask) == packedP0;

    if wonP0 {
      if turnP0 {
        atomicAdd(&stats.sumP0Win, 1u);
      }

      atomicAdd(&stats.sumPlayoutLength, currentLength + 1u);
      break;
    }

    turnP0 = !turnP0;


    // Transpose board

    for (var column = 0u; column < boardSize; column += 1u) {
      let value = board[column];

      board[column] = -board[boardSize - 1u - column];
      board[boardSize - 1u - column] = value;
    }

    packedP0 = ~packedEmpty & ~packedP0;
    packedP0 = reverseBits(packedP0) >> (32u - boardSize);
    packedMany = reverseBits(packedMany) >> (32u - boardSize);
    packedEmpty = reverseBits(packedEmpty) >> (32u - boardSize);

    // 0000 0123 (source)
    // 3210 0000 reverseBits
    // 0032 1000 >> 2
  }

  // for (var index = 0u; index < boardSize; index += 1u) {
  //   stats.out[index] = board[index];
  // }
}
