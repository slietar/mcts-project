const blockSize: u32 = 16;
const quarterBoardLength: u32 = 6;
const halfBoardLength: u32 = quarterBoardLength * 2;
const fullBoardLength: u32 = halfBoardLength * 2;
const playerPieceCount: u32 = 15;

// @group(0) @binding(0)
// var<storage, read> pieceEncodings: array<u32, ((fullBoardLength + 2) * (playerPieceCount * 2) + 1)>;

struct Settings {
  playoutLength: u32,
  // turnP0Encoding: u32,
}

@group(0) @binding(0)
var<storage, read> random: array<u32>;

@group(0) @binding(1)
var<storage, read> settings: Settings;

@group(0) @binding(2)
var<storage, read_write> initialBoards: array<i32, (fullBoardLength + 2)>;


@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let playoutId = gid.x;

  var currentRandomIndex = settings.playoutLength * playoutId;
  var turnP0 = true;

  var board: array<i32, (fullBoardLength + 2)>;

  for (var index = 0u; index < fullBoardLength + 2; index += 1u) {
    board[index] = initialBoards[playoutId * (fullBoardLength + 2u) + index];
  }

  loop {
    let distance = random[currentRandomIndex];
    currentRandomIndex += 1u;

    for (var startColumnAttemptIndex = 0u; startColumnAttemptIndex < fullBoardLength + 2u; startColumnAttemptIndex += 1u) {
      let startColumn = random[currentRandomIndex];
      currentRandomIndex += 1u;

      if turnP0 {
        let endColumn = startColumn + distance;

        if !(
          (endColumn < fullBoardLength + 1) &&
          (board[startColumn] > 0) &&
          (board[endColumn] >= -1) &&
          (board[0] <= 0 || startColumn == 0)
        ) {
          continue;
        }

        board[startColumn] -= 1;

        if board[endColumn] < 0 {
          board[endColumn] = 1;
          board[fullBoardLength - 1] -= 1;
        } else {
          board[endColumn] += 1;
        }

        break;
      }
    }

    break;
  }

  for (var index = 0u; index < fullBoardLength + 2; index += 1u) {
    initialBoards[playoutId * (fullBoardLength + 2u) + index] = board[index];
  }
}
