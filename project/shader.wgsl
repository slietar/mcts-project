const blockSize: u32 = 16;
const quarterBoardLength: u32 = 6;
const halfBoardLength: u32 = quarterBoardLength * 2;
const fullBoardLength: u32 = halfBoardLength * 2;
const boardSize: u32 = fullBoardLength + 2;
const playerPieceCount: u32 = 15;

// @group(0) @binding(0)
// var<storage, read> pieceEncodings: array<u32, (boardSize * (playerPieceCount * 2) + 1)>;

struct Settings {
  playoutLength: u32,
  // turnP0Encoding: u32,
}

@group(0) @binding(0)
var<storage, read> random: array<f32>;

@group(0) @binding(1)
var<storage, read> settings: Settings;

@group(0) @binding(2)
var<storage, read_write> initialBoards: array<i32>;

@group(0) @binding(3)
var<storage, read_write> outArr: array<f32, 1>;


@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let playoutId = gid.x;

  var currentRandomIndex = settings.playoutLength * playoutId;
  var turnP0 = true;

  var board: array<i32, boardSize>;

  for (var index = 0u; index < boardSize; index += 1u) {
    board[index] = initialBoards[playoutId * boardSize + index];
    // initialBoards[playoutId * boardSize + index] = -1;
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


  for (var playoutIndex = 0; playoutIndex < 1; playoutIndex += 1) {
    let distance = u32(random[currentRandomIndex] * 6.0) + 1u;
    // let distance = 4u;
    currentRandomIndex += 1u;

    if turnP0 {
      if board[0] > 0 {
        // ...
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
          // let moveIndex = moveCount - 1u;
          let moveIndex = u32(random[currentRandomIndex] * f32(moveCount));
          currentRandomIndex += 1u;

          var packedCurrent = packedLegal;
          var startColumn = 0u;

          for (var currentMoveIndex = 0u; currentMoveIndex <= moveIndex; currentMoveIndex += 1u) {
            let advance = countTrailingZeros(packedCurrent) + 1u;
            // initialBoards[currentMoveIndex] = i32(startColumn + advance - 1u);
            startColumn += advance;
            packedCurrent >>= advance;
          }

          startColumn -= 1u;
          let endColumn = startColumn + distance;

          board[startColumn] -= 1;
          board[endColumn] += 1;

          let startMask = 1u << startColumn;
          let endMask = 1u << endColumn;

          // packedEmpty = board[startColumn] > 0 ? packedEmpty & ~startMask : packedEmpty | startMask;

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
      }
    }
  }

  for (var index = 0u; index < boardSize; index += 1u) {
    initialBoards[playoutId * boardSize + index] = board[index];
  }
}
